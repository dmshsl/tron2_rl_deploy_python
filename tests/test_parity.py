"""Replay real Isaac traces without Isaac or the SDK; missing traces fail closed."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from scipy.spatial.transform import Rotation

from tron2_deploy.policy_contract import load_contract
from tron2_deploy.policy_observation import PolicyMemory, SdkState, build_actor_input
from tron2_deploy.policy_targets import action_targets, torque_preclip

TRACE_DIR = Path(os.environ.get("TRON2_TRACE_DIR", "/home/eunwon/tron2_ws/artifacts/basecampaign/traces"))
MODELS = Path(__file__).resolve().parents[1] / "controllers/model"
CASES = ("base_flat", "base_stairs", "base_blind_flat", "base_blind_stairs", "base_tilted")


@pytest.mark.parametrize("case", CASES)
def test_deploy_parity_when_replaying_isaac(case: str) -> None:
    # Given: independent Isaac tensors and SDK-equivalent input streams.
    model_dir = MODELS / ("WF_TRON2A_BASE_BLIND" if "blind" in case else "WF_TRON2A_BASE")
    contract_path = Path(os.environ.get("TRON2_PARITY_CONTRACT", model_dir / "contract.yaml"))
    contract = load_contract(contract_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    encoder = ort.InferenceSession(str(model_dir / "encoder.onnx"), options, providers=["CPUExecutionProvider"])
    policy = ort.InferenceSession(str(model_dir / "policy.onnx"), options, providers=["CPUExecutionProvider"])
    for session in (encoder, policy):
        assert session.get_inputs()[0].type == session.get_outputs()[0].type == "tensor(float)"
        assert session.get_modelmeta().custom_metadata_map["tron2.internal_precision"] == "float64"
    maxima: dict[str, float] = {}

    def compare(block: str, actual: np.ndarray, expected: np.ndarray) -> None:
        delta = float(np.max(np.abs(actual - expected), initial=0))
        maxima[block] = max(maxima.get(block, 0.), delta)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0,
                                   err_msg=f"{case} step={step} block={block}")

    with np.load(TRACE_DIR / f"{case}.npz", allow_pickle=False) as trace:
        assert trace["q"].shape == (300, 10)
        assert trace["raw_scan"].shape == (300, 121 if "blind" in case else 231)
        assert trace["policy_scan"].shape == (300, 0 if "blind" in case else 231)
        np.testing.assert_allclose(contract.imu.imu_to_base_rpy, trace["imu_mount_rpy"], atol=1e-8)
        if case == "base_tilted":
            angles = Rotation.from_quat(trace["base_quat"][0, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)
            np.testing.assert_allclose(angles[:2], [15., 15.], atol=1e-5, rtol=0)
        mem = PolicyMemory.empty(contract)
        for step in range(300):
            if trace["reset"][step]:
                mem = PolicyMemory.empty(contract)
            state = SdkState(trace["q"][step], trace["dq"][step], trace["imu_quat"][step],
                             trace["imu_gyro"][step], step * contract.control.policy_dt)
            compare("ema_before", mem.vel_error_ema, trace["ema_before"][step])
            compare("applied_last_action", mem.last_action, trace["last_action"][step])
            # When: replay the same state through the real pure builder and ONNX models.
            actor, history = build_actor_input(state, mem, trace["policy_scan"][step],
                                               trace["commands"][step], contract)
            # Then: name the observation block before checking neural-network outputs.
            offset = contract.encoder_output_size
            for term in contract.policy_terms:
                end = offset + term.size
                compare(term.name, actor[offset:end], trace["actor_input"][step, offset:end])
                offset = end
            compare("commands", actor[offset:], trace["commands_obs"][step])
            compare("policy", actor[contract.encoder_output_size:offset], trace["policy"][step])
            compare("obsHistory", history, trace["obsHistory"][step])
            latent = encoder.run(None, {encoder.get_inputs()[0].name: history})[0]
            compare("encoder", latent, trace["actor_input"][step, :contract.encoder_output_size])
            actor[:contract.encoder_output_size] = latent
            compare("actor_input", actor, trace["actor_input"][step])
            actions = policy.run(None, {policy.get_inputs()[0].name: actor})[0]
            compare("raw_actions", actions, trace["raw_actions"][step])
            targets = action_targets(actions, contract)
            compare("joint_targets_q", targets.q, trace["target_q"][step])
            compare("joint_targets_dq", targets.dq, trace["target_dq"][step])
            torque = targets.kp * (targets.q - state.q) + targets.kd * (targets.dq - state.dq)
            maxima["torque_report_only"] = max(maxima.get("torque_report_only", 0.),
                float(np.max(np.abs(torque - trace["torque_unclipped"][step]))))
            clipped = torque_preclip(targets, state, contract)
            maxima["torque_preclip_target_change_report_only"] = max(
                maxima.get("torque_preclip_target_change_report_only", 0.),
                float(np.max(np.abs(np.concatenate((clipped.q - targets.q, clipped.dq - targets.dq))))))
            ema_term = next(t for t in contract.policy_terms if t.name == "vel_error_ema")
            ema_offset = sum(t.size for t in contract.policy_terms[:contract.policy_terms.index(ema_term)])
            ema = actor[contract.encoder_output_size + ema_offset:contract.encoder_output_size + ema_offset + 2]
            compare("ema_state", ema, trace["ema_after"][step])
            # Replay uses the action actually applied to the recorded plant, not a counterfactual rollout.
            mem = PolicyMemory(history.reshape(contract.history_length, -1), trace["raw_actions"][step], ema.copy(), latent)
    print(f"PARITY {case} " + " ".join(f"{key}={value:.9g}" for key, value in maxima.items()))


def test_zero_default_pose_contract_fails_on_joint_pos(tmp_path: Path) -> None:
    # Given: a temporary contract copy with only the default pose deliberately broken.
    contract = load_contract(MODELS / "WF_TRON2A_BASE/contract.yaml")
    broken_joints = contract.joints.model_copy(update={"default_position": (0.,) * 10})
    broken = contract.model_copy(update={"joints": broken_joints})
    copy = tmp_path / "zero-pose.yaml"
    copy.write_text(broken.model_dump_json())
    # `tron2_deploy` is a namespace package: importing it needs its parent
    # directory (the superproject root) on sys.path, which is only true by
    # accident when the OUTER pytest happens to be invoked from there. Pin
    # it explicitly via PYTHONPATH so this subprocess resolves the import
    # regardless of the outer invocation's current working directory.
    superproject_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = str(superproject_root) if not existing_pythonpath else \
        f"{superproject_root}{os.pathsep}{existing_pythonpath}"
    environment = dict(os.environ, TRON2_PARITY_CONTRACT=str(copy), PYTHONPATH=pythonpath)
    # When: rerun the actual replay test against the temporary contract.
    result = subprocess.run([sys.executable, "-m", "pytest", str(Path(__file__).resolve()),
                             "-q", "-k", "base_flat", "--tb=short"],
                            env=environment, capture_output=True, text=True, timeout=120, check=False)
    # Then: the test must fail at the joint-position observation, not a downstream block.
    assert result.returncode == 1, result.stdout + result.stderr
    assert "step=0 block=joint_pos" in result.stdout, result.stdout
    print("NEGATIVE_CONTROL expected_exit=1 actual_exit=1 block=joint_pos; original contract unchanged")
