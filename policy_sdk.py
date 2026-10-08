"""SDK callback adapter with copied samples, startup gates, and a stale-data watchdog."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic, time_ns
from typing import Protocol

import numpy as np

from tron2_deploy.policy_observation import PolicyInputError, SdkState, Vector
from tron2_deploy.policy_targets import JointTargets


class StateMessage(Protocol):
    stamp: int
    q: Sequence[float]
    dq: Sequence[float]


class ImuMessage(Protocol):
    stamp: int
    quat: Sequence[float]
    gyro: Sequence[float]


class JoyMessage(Protocol):
    buttons: Sequence[int]
    axes: Sequence[float]


class CommandMessage(Protocol):
    mode: list[float]
    q: list[float]
    dq: list[float]
    tau: list[float]
    Kp: list[float]
    Kd: list[float]
    motor_names: list[str]


class RobotInterface(Protocol):
    def subscribeRobotState(self, callback: Callable[[StateMessage], None]) -> None: ...
    def subscribeImuData(self, callback: Callable[[ImuMessage], None]) -> None: ...
    def subscribeSensorJoy(self, callback: Callable[[JoyMessage], None]) -> None: ...
    def publishRobotCmd(self, command: CommandMessage) -> None: ...


@dataclass(frozen=True, slots=True)
class JoyState:
    prepare: bool
    stop: bool
    axes: Vector
    timestamp: float


class SdkAdapter:
    def __init__(self, robot: RobotInterface, command_factory: Callable[[], CommandMessage],
                 clock: Callable[[], float] = monotonic) -> None:
        self.robot, self.command = robot, command_factory()
        self.emergency = command_factory()
        self.clock = clock
        self.lock = Lock()
        self.closed, self.tripped = Event(), Event()
        self.fault = ""
        self.last_publish: float | None = None
        self.startup_deadline = self.clock() + 5.
        self.ready_state, self.ready_imu = Event(), Event()
        self.q = self.dq = np.zeros(10, dtype=np.float32)
        self.quat = np.array([1., 0., 0., 0.], dtype=np.float32)
        self.gyro = np.zeros(3, dtype=np.float32)
        self.state_time = self.imu_time = -np.inf
        self.state_stamp = self.imu_stamp = 0
        self.joy = JoyState(False, False, np.zeros(3, dtype=np.float32), -np.inf)
        for command in (self.command, self.emergency):
            command.mode = [0.] * 10
            command.q, command.dq, command.tau = [0.] * 10, [0.] * 10, [0.] * 10
            command.Kp, command.Kd = [0.] * 10, [0.] * 10
            command.motor_names = [""] * 10
        robot.subscribeRobotState(self.on_state)
        robot.subscribeImuData(self.on_imu)
        robot.subscribeSensorJoy(self.on_joy)
        self.watchdog = Thread(target=self._watchdog, name="policy-output-watchdog", daemon=True)
        self.watchdog.start()

    def _valid_stamp(self, stamp: int, previous: int) -> bool:
        if stamp <= previous or abs(time_ns() - stamp) > 250_000_000:
            self.fault = "SDK source timestamp stale, regressing, or not Unix nanoseconds"
            return False
        return True

    def on_state(self, message: StateMessage) -> None:
        with self.lock:
            if not self._valid_stamp(message.stamp, self.state_stamp):
                return
            self.q = np.array(message.q, dtype=np.float32)
            self.dq = np.array(message.dq, dtype=np.float32)
            self.state_time, self.state_stamp = self.clock(), message.stamp
            self.ready_state.set()

    def on_imu(self, message: ImuMessage) -> None:
        with self.lock:
            if not self._valid_stamp(message.stamp, self.imu_stamp):
                return
            self.quat = np.array(message.quat, dtype=np.float32)
            self.gyro = np.array(message.gyro, dtype=np.float32)
            self.imu_time, self.imu_stamp = self.clock(), message.stamp
            self.ready_imu.set()

    def on_joy(self, message: JoyMessage) -> None:
        if len(message.buttons) < 5 or len(message.axes) < 3:
            return
        with self.lock:
            l1 = bool(message.buttons[4])
            self.joy = JoyState(l1 and bool(message.buttons[1] or message.buttons[3]),
                                l1 and bool(message.buttons[0]),
                                np.array([message.axes[1], message.axes[0], message.axes[2]], dtype=np.float32),
                                self.clock())

    def sample(self, now: float) -> tuple[SdkState, JoyState]:
        with self.lock:
            if self.fault or self.tripped.is_set():
                raise PolicyInputError(self.fault or "output watchdog tripped; restart required")
            if now - min(self.state_time, self.imu_time) > 0.25:
                raise PolicyInputError("SDK state/IMU stale for more than 0.25 seconds")
            if abs(self.state_stamp - self.imu_stamp) > 50_000_000:
                raise PolicyInputError("SDK state/IMU source timestamp skew exceeds 0.05 seconds")
            return SdkState(self.q, self.dq, self.quat, self.gyro, now), self.joy

    def publish(self, targets: JointTargets) -> None:
        if self.tripped.is_set():
            raise PolicyInputError("output watchdog tripped; restart required")
        self.command.q, self.command.dq = targets.q.tolist(), targets.dq.tolist()
        self.command.Kp, self.command.Kd = targets.kp.tolist(), targets.kd.tolist()
        self.last_publish = self.clock()
        self.robot.publishRobotCmd(self.command)

    def _watchdog(self) -> None:
        """Independent of inference/scan acquisition; use a separate immutable stop packet."""
        while not self.closed.wait(0.05):
            now = self.clock()
            startup_expired = self.last_publish is None and now > self.startup_deadline
            if startup_expired or self.tripped.is_set() or (self.last_publish is not None and (self.fault or
                    now - min(self.last_publish, self.state_time, self.imu_time) > 0.25)):
                self.tripped.set()
                self.robot.publishRobotCmd(self.emergency)

    def release(self) -> None:
        """Send zero gains/torques on every exit, including invalid or stale sensors."""
        self.closed.set()
        self.robot.publishRobotCmd(self.emergency)
        self.watchdog.join(timeout=0.5)
