"""Run Base policies on localhost only; --dry-run validates without importing SDK.

For Base, set TRON2_HEIGHT_SCAN_FACTORY=module:factory; factory(contract, mode)
returns HeightScanProvider. GT is supplied by todo 11; perception factories return
PerceptionHeightScanProvider(contract, frame_source). Schedule times start after
the three-second preparation, in physical m/s and rad/s (not joystick units).
"""

import argparse
import csv
import importlib
import os
import sys
from contextlib import ExitStack
from itertools import pairwise
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, TypeAdapter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "controllers"))

from PolicyWheelfootController import PolicyWheelfootController

from tron2_deploy.height_scan_provider import HeightScanProvider
from tron2_deploy.policy_contract import DeploymentContract, load_contract
from tron2_deploy.policy_observation import PolicyInputError, Vector
from tron2_deploy.policy_sdk import SdkAdapter


class ScheduleEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    t_start: float
    vx: float
    vy: float
    wz: float


def load_schedule(path: Path | None) -> tuple[ScheduleEntry, ...]:
    if path is None:
        return ()
    entries = TypeAdapter(list[ScheduleEntry | tuple[float, float, float, float]]).validate_python(
        yaml.safe_load(path.read_text()))
    result = tuple(entry if isinstance(entry, ScheduleEntry) else
                   ScheduleEntry(t_start=entry[0], vx=entry[1], vy=entry[2], wz=entry[3]) for entry in entries)
    if not result or result[0].t_start != 0 or any(b.t_start <= a.t_start for a, b in pairwise(result)):
        raise PolicyInputError("schedule must start at zero with strictly increasing t_start")
    return result


def schedule_command(schedule: tuple[ScheduleEntry, ...], elapsed: float) -> Vector:
    selected = schedule[0]
    for entry in schedule:
        if entry.t_start > elapsed:
            break
        selected = entry
    return np.array([selected.vx, selected.vy, selected.wz], dtype=np.float32)


def height_provider(contract: DeploymentContract, mode: str) -> HeightScanProvider | None:
    if contract.policy == "base_blind":
        return None
    factory_path = os.environ.get("TRON2_HEIGHT_SCAN_FACTORY", "")
    module, separator, name = factory_path.partition(":")
    if not separator:
        raise PolicyInputError("Base requires TRON2_HEIGHT_SCAN_FACTORY=module:factory (contract, mode) returning HeightScanProvider; no fabricated scan fallback")
    provider = getattr(importlib.import_module(module), name)(contract, mode)
    if not callable(getattr(provider, "get_scan", None)):
        raise PolicyInputError("height factory must return HeightScanProvider.get_scan(state)")
    return provider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("base", "base_blind"), required=True)
    parser.add_argument("--robot_ip", default="127.0.0.1")
    parser.add_argument("--autostart", action="store_true")
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--timeout", type=float, default=60.)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--height_scan", choices=("gt", "perception"), default="perception")
    parser.add_argument("--imu_offset_rpy", nargs=3, type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.robot_ip != "127.0.0.1":
        parser.error("this execution scope forbids real-robot connections; use 127.0.0.1")
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    model_dir = Path(__file__).resolve().parent / "controllers/model" / f"WF_TRON2A_{args.policy.upper()}"
    contract = load_contract(model_dir / "contract.yaml")
    if args.imu_offset_rpy is not None:
        data = contract.model_dump()
        data["imu"]["imu_to_base_rpy"] = args.imu_offset_rpy
        data["imu"]["calibration"] = "OVERRIDE"
        contract = DeploymentContract.model_validate(data)
    schedule = load_schedule(args.schedule)
    for entry in schedule:
        values = (entry.vx, entry.vy, entry.wz)
        if any(not low <= value <= high for value, (low, high) in zip(values, contract.command_ranges, strict=True)):
            parser.error("schedule velocity is outside the generated training command ranges")
    print(yaml.safe_dump(contract.model_dump(mode="json"), sort_keys=False), flush=True)
    provider = None if args.dry_run else height_provider(contract, args.height_scan)
    controller = PolicyWheelfootController(model_dir, contract, provider)
    if args.dry_run:
        print(f"DRY_RUN PASS actor={contract.actor_size} encoder={contract.encoder_input_size}; SDK not imported")
        return 0

    from limxsdk import datatypes
    from limxsdk.robot import Robot, RobotType

    robot = Robot(RobotType.Tron2)
    if not robot.init(args.robot_ip):
        raise PolicyInputError("SDK localhost initialization failed")
    adapter = SdkAdapter(robot, datatypes.RobotCmd)
    try:
        deadline = monotonic() + min(5., args.timeout)
        if not adapter.ready_state.wait(max(0., deadline - monotonic())) or not adapter.ready_imu.wait(max(0., deadline - monotonic())):
            raise PolicyInputError("no SDK state and IMU within startup timeout")
        with ExitStack() as stack:
            writer = None
            if args.csv:
                stream = stack.enter_context(args.csv.open("w", newline=""))
                writer = csv.writer(stream)
                header = ["t", "policy_t", "vx", "vy", "wz"] + [
                    f"{prefix}_{i}" for prefix, count in (("q", 10), ("dq", 10), ("quat", 4),
                    ("gyro", 3), ("action", 10), ("target_q", 10), ("target_dq", 10)) for i in range(count)]
                if provider is not None:
                    header += [f"scan_{i}" for i in range(contract.scan.count)]
                writer.writerow(header)
            begin = monotonic()
            next_tick, last_prepare = begin, False
            autostart = args.autostart
            while monotonic() - begin < args.timeout:
                now = monotonic()
                state, joy = adapter.sample(now)
                if joy.stop:
                    controller.stop()
                    autostart = False
                elif autostart or (joy.prepare and not last_prepare and now - joy.timestamp <= 0.5):
                    controller.prepare(state)
                    autostart = False
                last_prepare = joy.prepare
                policy_time = 0. if controller.started_at is None else max(0., now - controller.started_at - contract.control.power_on_seconds)
                commands = np.zeros(3, dtype=np.float32)
                if schedule:
                    commands = schedule_command(schedule, policy_time)
                elif now - joy.timestamp <= 0.5:
                    ranges = np.array(contract.command_ranges, dtype=np.float32)
                    axes = np.clip(joy.axes, -1., 1.)
                    commands = np.where(axes >= 0, axes * ranges[:, 1], -axes * ranges[:, 0])
                targets = controller.step(state, commands)
                adapter.publish(targets)
                if writer is not None:
                    row = [now - begin, policy_time, *commands, *state.q, *state.dq, *state.quat,
                          *state.gyro, *controller.mem.last_action, *targets.q, *targets.dq]
                    if provider is not None:
                        row += (list(controller.last_scan) if controller.last_scan is not None
                               else [float("nan")] * contract.scan.count)
                    writer.writerow(row)
                next_tick += contract.control.publish_dt
                delay = next_tick - monotonic()
                if delay > 0:
                    sleep(delay)
                else:
                    next_tick = monotonic()
    finally:
        adapter.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
