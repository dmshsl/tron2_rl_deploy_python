"""Command clamp, software cap and slew limiting for every command source."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tron2_deploy.command_shaping import CommandLimits, CommandShaper, limits_from_contract
from tron2_deploy.policy_contract import DeploymentContract, load_contract
from tron2_deploy.policy_observation import SdkState

MODELS = Path(__file__).resolve().parents[1] / "controllers/model"
sys.path.insert(0, str(MODELS.parent))
from PolicyWheelfootController import PolicyWheelfootController

HW = CommandLimits(low=(-3., 0., -2.), high=(3., 0., 2.), cap=(1., 0., 1.),
                   accel=(1.5, 1.5, 3.), decel=(3., 3., 6.))
DT = 0.002


def run(shaper: CommandShaper, command, seconds: float, t0: float = 0.) -> tuple[np.ndarray, float]:
    out, t = shaper.output, t0
    for _ in range(round(seconds / DT)):
        t += DT
        out = shaper(np.asarray(command, dtype=np.float32), t)
    return out, t


def test_nan_and_inf_components_become_zero() -> None:
    shaper = CommandShaper(HW)
    np.testing.assert_array_equal(shaper.clamp(np.array([np.nan, np.inf, -np.inf])), [0., 0., 0.])


@pytest.mark.parametrize(("raw", "expected"), [((10., 5., 10.), (1., 0., 1.)), ((-10., -5., -10.), (-1., 0., -1.)),
                                               ((0.4, 0., -0.3), (0.4, 0., -0.3))])
def test_default_cap_and_trained_range_clamp(raw, expected) -> None:
    np.testing.assert_allclose(CommandShaper(HW).clamp(np.array(raw)), expected)


def test_cap_is_adjustable_but_never_exceeds_the_trained_range() -> None:
    shaper = CommandShaper(HW)
    assert shaper.set_cap(vx=2.5) == (2.5, 1.)
    np.testing.assert_allclose(shaper.clamp(np.array([9., 0., 9.])), [2.5, 0., 1.])
    assert shaper.set_cap(vx=99., wz=99.) == (3., 2.)
    np.testing.assert_allclose(shaper.clamp(np.array([9., 0., -9.])), [3., 0., -2.])
    assert shaper.set_cap(vx=0.) == (0., 2.)
    with pytest.raises(ValueError):
        shaper.set_cap(vx=-1.)
    with pytest.raises(ValueError):
        shaper.set_cap(wz=float("nan"))


def test_cap_from_constructor_is_clipped_to_the_range() -> None:
    assert CommandShaper(replace(HW, cap=(5., 0., 5.))).limits.cap == (3., 0., 2.)


def test_slew_limits_acceleration_and_braking() -> None:
    shaper = CommandShaper(HW)
    shaper.set_cap(vx=3.)
    out, t = run(shaper, (3., 0., 0.), 1.0)
    assert out[0] == pytest.approx(1.5, abs=5e-3)  # 1.5 m/s^2 (first tick has dt = 0)
    out, t = run(shaper, (3., 0., 0.), 1.1, t)
    assert out[0] == pytest.approx(3.0, abs=1e-6)
    out, t = run(shaper, (0., 0., 0.), 0.5, t)
    assert out[0] == pytest.approx(1.5, abs=2e-3)  # 3 m/s^2 braking
    out, _ = run(shaper, (0., 0., 0.), 0.6, t)
    assert out[0] == 0.


def test_reversal_brakes_through_zero_then_accelerates() -> None:
    shaper = CommandShaper(HW)
    out, t = run(shaper, (1., 0., 0.), 1.0)
    assert out[0] == pytest.approx(1.0)
    trace = []
    for _ in range(round(1.5 / DT)):
        t += DT
        trace.append(shaper(np.array([-1., 0., 0.]), t)[0])
    trace = np.array(trace)
    zero = np.argmax(trace <= 0)
    assert zero * DT == pytest.approx(1.0 / 3.0, abs=2 * DT)  # decel phase
    assert trace[-1] == pytest.approx(-1.0, abs=1e-6)
    assert np.all(np.abs(np.diff(trace)) <= 3.0 * DT + 1e-6)


def test_yaw_rate_slews_independently() -> None:
    shaper = CommandShaper(HW)
    out, _ = run(shaper, (0., 0., 1.), 0.2)
    assert out[2] == pytest.approx(0.6, abs=1e-2) and out[0] == 0.


def test_lateral_acceleration_limit_reduces_yaw_rate_only() -> None:
    shaper = CommandShaper(replace(HW, cap=(3., 0., 2.), lateral_accel_max=3.))
    np.testing.assert_allclose(shaper.clamp(np.array([3., 0., 2.])), [3., 0., 1.])
    np.testing.assert_allclose(shaper.clamp(np.array([1., 0., 2.])), [1., 0., 2.])
    np.testing.assert_allclose(shaper.clamp(np.array([0., 0., -2.])), [0., 0., -2.])


def test_first_call_and_reset_start_from_rest() -> None:
    shaper = CommandShaper(HW)
    assert np.all(shaper(np.array([1., 0., 1.]), 100.) == 0.)
    run(shaper, (1., 0., 1.), 2., 100.)
    shaper.reset()
    assert np.all(shaper(np.array([1., 0., 1.]), 500.) == 0.)


def test_backwards_timestamps_do_not_move_the_output() -> None:
    shaper = CommandShaper(HW)
    shaper(np.array([1., 0., 0.]), 1.)
    assert shaper(np.array([1., 0., 0.]), 0.5)[0] == 0.


@pytest.mark.parametrize("bad", [dict(low=(0.1, 0., -1.)), dict(accel=(0., 1., 1.)), dict(cap=(-1., 0., 1.)),
                                 dict(lateral_accel_max=0.)])
def test_invalid_limits_are_rejected(bad) -> None:
    with pytest.raises(ValueError):
        replace(HW, **bad)


def test_contract_section_defaults_and_validation() -> None:
    contract = load_contract(MODELS / "WF_TRON2A_BASE/contract.yaml")
    limits = limits_from_contract(contract)
    assert limits.cap[0] == contract.commands.default_cap[0] and limits.cap[2] == contract.commands.default_cap[1]
    assert (limits.low[0], limits.high[0]) == tuple(contract.command_ranges[0])
    data = contract.model_dump(mode="json")
    data["commands"]["default_cap"] = [contract.command_ranges[0][1] + 0.5, 1.0]
    with pytest.raises(ValueError):
        DeploymentContract.model_validate(data)


@pytest.fixture
def controller():
    model = MODELS / "WF_TRON2A_BASE_BLIND"
    return PolicyWheelfootController(model, load_contract(model / "contract.yaml"))


def test_controller_shapes_every_command_and_nan_does_not_latch(controller) -> None:
    contract = controller.contract
    state = SdkState(np.array(contract.joints.default_position), np.zeros(10), np.array([1., 0., 0., 0.]),
                     np.zeros(3), 0.)
    controller.prepare(state)
    t = contract.control.power_on_seconds + 0.02
    seen = []
    for i in range(100):
        command = np.array([np.nan, 0., 0.]) if i == 10 else np.array([5., 3., -5.])
        controller.step(replace(state, timestamp=t + i * contract.control.publish_dt), command)
        seen.append(controller.shaped_command.copy())
    assert not controller.safety.reason
    seen = np.array(seen)
    assert np.all(np.abs(seen[:, 0]) <= contract.commands.default_cap[0] + 1e-6)
    assert np.all(seen[:, 1] == 0.)
    assert np.all(np.abs(np.diff(seen[:, 0])) <= contract.commands.decel[0] * contract.control.publish_dt + 1e-6)
    assert seen[-1, 0] > 0 and seen[-1, 2] < 0
    assert controller.set_command_cap(vx=0.2) == (0.2, contract.commands.default_cap[1])
    controller.stop()
    assert np.all(controller.shaped_command == 0.)


def test_environment_cap_override_is_clipped_to_the_trained_range(monkeypatch) -> None:
    monkeypatch.setenv("TRON2_COMMAND_CAP", "9:0.5")
    model = MODELS / "WF_TRON2A_BASE_BLIND"
    contract = load_contract(model / "contract.yaml")
    controller = PolicyWheelfootController(model, contract)
    reach = max(-contract.command_ranges[0][0], contract.command_ranges[0][1])
    assert controller.command_shaper.limits.cap[0] == reach and controller.command_shaper.limits.cap[2] == 0.5


def test_heading_hold_steers_back_only_at_zero_yaw_command() -> None:
    from tron2_deploy.command_shaping import HeadingHold, yaw_of

    hold = HeadingHold(gain=1.0, max_rate=0.3)
    assert np.all(HeadingHold(0.0, 0.3)(np.zeros(3, np.float32), 0.0, 0.5) == 0.0)
    assert hold(np.zeros(3, np.float32), 0.0, 0.1)[2] == 0.0          # first call arms ref = 0.1
    assert np.isclose(hold(np.zeros(3, np.float32), 0.0, 0.15)[2], -0.05, atol=1e-6)
    assert hold(np.zeros(3, np.float32), 0.0, 1.5)[2] == np.float32(-0.3)  # clipped
    turning = np.array([0.0, 0.0, 0.4], np.float32)
    assert hold(turning, 0.4, 2.0)[2] == np.float32(0.4)              # operator turning: unchanged, ref follows
    assert hold(np.zeros(3, np.float32), 0.0, 2.0)[2] == 0.0
    assert np.isclose(hold(np.zeros(3, np.float32), 0.0, 2.0 - 2 * np.pi + 0.02)[2], -0.02, atol=1e-5)  # wrapped
    hold.reset()
    assert hold(np.zeros(3, np.float32), 0.0, -1.0)[2] == 0.0
    assert np.isclose(yaw_of(np.array([np.cos(0.25), 0.0, 0.0, np.sin(0.25)])), 0.5)
