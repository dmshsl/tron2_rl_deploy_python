"""Replay Isaac StandUp traces through the deploy stand-up builder + ONNX; missing traces fail closed.

Trace: tron2_rl/standup/standup_trace.py (deterministic play cfg, 4 envs x 300 steps, auto resets).
Model: controllers/model/WF_TRON2A_STANDUP (or TRON2_STANDUP_MODEL_DIR), trace from
TRON2_STANDUP_TRACE; both must come from the same checkpoint (sha256 cross-checked).
"""

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from tron2_deploy.policy_observation import SdkState
from tron2_deploy.standup_contract import OBSERVATION_BLOCKS, StandupPolicyContract, load_standup_contract
from tron2_deploy.standup_policy import StandupMemory, incremental_target, standup_observation, wheel_torque

MODEL = Path(os.environ.get("TRON2_STANDUP_MODEL_DIR",
                            Path(__file__).resolve().parents[1] / "controllers/model/WF_TRON2A_STANDUP"))
TRACE = Path(os.environ.get("TRON2_STANDUP_TRACE",
                            "/home/eunwon/tron2_ws/artifacts/basecampaign/traces/standup_pose_a.npz"))


def replay(contract: StandupPolicyContract, session: ort.InferenceSession, trace) -> dict[str, float]:
    maxima: dict[str, float] = {}

    def compare(block: str, actual: np.ndarray, expected: np.ndarray, where: str) -> None:
        delta = float(np.max(np.abs(np.asarray(actual, dtype=np.float64) - expected), initial=0))
        maxima[block] = max(maxima.get(block, 0.), delta)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0, err_msg=f"{where} block={block}")

    positions = list(contract.position_sdk)
    envs, steps = trace["q"].shape[:2]
    for env in range(envs):
        memory = None
        for step in range(steps):
            where = f"env={env} step={step}"
            state = SdkState(trace["q"][env, step], trace["dq"][env, step], trace["imu_quat"][env, step],
                             trace["imu_gyro"][env, step], step * contract.policy_dt)
            if trace["reset"][env, step]:
                # Deploy start(): q_tgt latches the measured pose, last action zero (env _reset_idx).
                memory = StandupMemory(state.q[positions].astype(np.float32).copy(), np.zeros(10, dtype=np.float32))
            assert memory is not None
            compare("target_before", memory.target, trace["target_before"][env, step], where)
            compare("last_action", memory.last_action, trace["last_action"][env, step], where)
            actor = standup_observation(state, memory, contract)
            offset = 0
            for name, size in OBSERVATION_BLOCKS:
                compare(name, actor[offset:offset + size], trace["actor_input"][env, step, offset:offset + size], where)
                offset += size
            actions = session.run(None, {session.get_inputs()[0].name: actor})[0]
            compare("raw_actions", actions, trace["raw_actions"][env, step], where)
            clipped = np.clip(actions, -1., 1.).astype(np.float32)
            target = incremental_target(memory.target, state.q[positions].astype(np.float32), clipped[:8], contract)
            compare("target_after", target, trace["target_after"][env, step], where)
            compare("wheel_torque", wheel_torque(clipped[8:], contract), trace["wheel_torque"][env, step], where)
            # Continue from the plant's applied values (replay, not a counterfactual rollout).
            memory = StandupMemory(trace["target_after"][env, step].astype(np.float32),
                                   trace["applied_action"][env, step].astype(np.float32))
    return maxima


@pytest.fixture(scope="module")
def recorded():
    contract = load_standup_contract(MODEL / "contract.yaml")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL / "policy.onnx"), options, providers=["CPUExecutionProvider"])
    with np.load(TRACE, allow_pickle=False) as data:
        trace = {key: data[key] for key in data.files}
    return contract, session, trace


def test_standup_deploy_parity_when_replaying_isaac(recorded):
    # Given the shipped stand-up contract/ONNX and an Isaac trace of the same checkpoint.
    contract, session, trace = recorded
    meta = session.get_modelmeta().custom_metadata_map
    assert session.get_inputs()[0].type == session.get_outputs()[0].type == "tensor(float)"
    assert meta["tron2.internal_precision"] == "float64" and meta["tron2.policy"] == "standup"
    assert meta["tron2.checkpoint_sha256"] == contract.checkpoint_sha256 == str(trace["checkpoint_sha256"])
    assert list(trace["sdk_names"]) == list(contract.sdk_names)
    np.testing.assert_array_equal(trace["imu_mount_rpy"], contract.imu_to_base_rpy)
    assert trace["q"].shape[1] >= 300 and trace["reset"][:, 1:].any()  # includes automatic env resets
    # When every step is rebuilt by the deploy code, then every block matches within 1e-5.
    maxima = replay(contract, session, trace)
    print("STANDUP_PARITY " + " ".join(f"{key}={value:.9g}" for key, value in maxima.items()))
    assert max(maxima.values()) < 1e-5


def test_standup_parity_negative_control_fails_on_joint_pos(recorded):
    # Given a contract whose standing pose is deliberately wrong, the replay fails at joint_pos first.
    contract, session, trace = recorded
    broken = contract.model_copy(update={"standing": (0.,) * 8})
    with pytest.raises(AssertionError, match="block=joint_pos"):
        replay(broken, session, trace)
