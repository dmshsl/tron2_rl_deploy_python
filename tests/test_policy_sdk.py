from threading import Event
from time import monotonic, monotonic_ns, time_ns
from types import SimpleNamespace

import numpy as np
import pytest
from tron2_deploy.policy_observation import PolicyInputError
from tron2_deploy.policy_sdk import SdkAdapter
from tron2_deploy.policy_targets import JointTargets


class FakeRobot:
    def __init__(self):
        self.released = Event()

    def subscribeRobotState(self, callback):
        self.state_callback = callback

    def subscribeImuData(self, callback):
        self.imu_callback = callback

    def subscribeSensorJoy(self, callback):
        self.joy_callback = callback

    def publishRobotCmd(self, command):
        if np.all(np.array(command.Kp) == 0) and np.all(np.array(command.Kd) == 0):
            self.released.set()


def test_watchdog_when_control_loop_stalls_releases_independently():
    # Given valid sensors and an active command with an injectable clock.
    robot, clock = FakeRobot(), [monotonic()]
    adapter = SdkAdapter(robot, SimpleNamespace, lambda: clock[0])
    try:
        robot.state_callback(SimpleNamespace(stamp=time_ns(), q=np.zeros(10), dq=np.zeros(10)))
        robot.imu_callback(SimpleNamespace(stamp=time_ns(), quat=[1., 0., 0., 0.], gyro=np.zeros(3)))
        adapter.publish(JointTargets(np.zeros(10), np.zeros(10), np.ones(10), np.ones(10)))
        # When time advances without main-loop sample() or publish().
        clock[0] += 1.
        # Then the independent thread releases and latches the fault.
        assert robot.released.wait(1.)
        with pytest.raises(PolicyInputError, match="watchdog"):
            adapter.sample(clock[0])
    finally:
        adapter.release()


@pytest.mark.parametrize("age_ns", [1_000_000_000, -1_000_000_000])
def test_source_timestamp_when_arrival_is_fresh_rejects_old_or_future_data(age_ns):
    # Given a newly delivered message with an invalid source timestamp.
    robot = FakeRobot()
    adapter = SdkAdapter(robot, SimpleNamespace)
    try:
        robot.state_callback(SimpleNamespace(stamp=time_ns() - age_ns, q=np.zeros(10), dq=np.zeros(10)))
        # When sampled, then callback arrival cannot disguise sensor age.
        with pytest.raises(PolicyInputError, match="source timestamp"):
            adapter.sample(monotonic())
    finally:
        adapter.release()


def test_source_timestamp_when_streams_skew_rejects_combined_state():
    # Given individually fresh streams separated by 100 ms.
    robot = FakeRobot()
    adapter = SdkAdapter(robot, SimpleNamespace)
    try:
        robot.state_callback(SimpleNamespace(stamp=time_ns(), q=np.zeros(10), dq=np.zeros(10)))
        robot.imu_callback(SimpleNamespace(stamp=time_ns() - 100_000_000, quat=[1., 0., 0., 0.], gyro=np.zeros(3)))
        # When sampled, then reject rather than fuse incompatible frames.
        with pytest.raises(PolicyInputError, match="skew"):
            adapter.sample(monotonic())
    finally:
        adapter.release()


def test_watchdog_when_startup_stalls_before_first_publish_releases():
    # Given a connected adapter that has never published a command.
    robot, clock = FakeRobot(), [monotonic()]
    adapter = SdkAdapter(robot, SimpleNamespace, lambda: clock[0])
    try:
        # When startup exceeds its five-second grace period without main-loop progress.
        clock[0] += 6.
        # Then the independent watchdog sends zero gains and stays latched.
        assert robot.released.wait(1.)
        assert adapter.tripped.is_set()
    finally:
        adapter.release()


def test_source_timestamp_when_sdk_uses_monotonic_nanoseconds_accepts_state():
    robot = FakeRobot()
    adapter = SdkAdapter(robot, SimpleNamespace)
    try:
        robot.state_callback(SimpleNamespace(stamp=monotonic_ns(), q=np.zeros(10), dq=np.zeros(10)))
        robot.imu_callback(SimpleNamespace(stamp=monotonic_ns(), quat=[1., 0., 0., 0.], gyro=np.zeros(3)))
        state, _ = adapter.sample(monotonic())
        np.testing.assert_array_equal(state.q, np.zeros(10))
    finally:
        adapter.release()


def test_duplicate_source_timestamp_is_ignored_without_refreshing_age():
    robot = FakeRobot()
    adapter = SdkAdapter(robot, SimpleNamespace)
    try:
        stamp = time_ns()
        message = SimpleNamespace(stamp=stamp, q=np.zeros(10), dq=np.zeros(10))
        robot.state_callback(message)
        accepted_time = adapter.state_time
        robot.state_callback(message)
        assert adapter.fault == ""
        assert adapter.state_time == accepted_time
    finally:
        adapter.release()


@pytest.mark.parametrize("stamp", [float("nan"), float("inf"), "invalid", None])
def test_malformed_source_timestamp_does_not_refresh_sample(stamp):
    robot = FakeRobot()
    adapter = SdkAdapter(robot, SimpleNamespace)
    try:
        robot.state_callback(SimpleNamespace(stamp=stamp, q=np.zeros(10), dq=np.zeros(10)))
        assert not adapter.ready_state.is_set()
        with pytest.raises(PolicyInputError, match="source timestamp"):
            adapter.sample(monotonic())
    finally:
        adapter.release()
