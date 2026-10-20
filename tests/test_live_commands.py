"""Live command file (run_policy.py --live-commands): round trips, staleness and mode transitions."""
from pathlib import Path

import pytest

from tron2_deploy.live_commands import (
    MODE_IDLE,
    MODE_RUN,
    STATE_LATCHED,
    Command,
    LiveChannel,
    LiveCommandSource,
    Status,
)


class Clock:
    def __init__(self) -> None:
        self.now = 100.

    def __call__(self) -> float:
        return self.now


def test_round_trip_and_reason_truncation(tmp_path: Path) -> None:
    writer = LiveChannel(tmp_path / "live.bin", create=True)
    reader = LiveChannel(tmp_path / "live.bin")
    writer.write_command(Command(1.5, MODE_RUN, 3, .5, 0., -.25, 1., 2.))
    assert reader.read_command() == Command(1.5, MODE_RUN, 3, .5, 0., -.25, 1., 2.)
    reader.write_status(Status(2., 5, STATE_LATCHED, 1.25, .1, 0., .2, "tilt 1.3 > 1.2 rad " + "x" * 400))
    status = writer.read_status()
    assert (status.phase, status.state, status.policy_t) == (5, STATE_LATCHED, 1.25)
    assert status.reason.startswith("tilt 1.3") and len(status.reason.encode()) <= 256
    writer.close()
    reader.close()


def test_rejects_foreign_file(tmp_path: Path) -> None:
    (tmp_path / "other.bin").write_bytes(b"\0" * 1024)
    with pytest.raises(ValueError):
        LiveChannel(tmp_path / "other.bin")


def test_transitions_and_stale_commands_read_zero(tmp_path: Path) -> None:
    gui = LiveChannel(tmp_path / "live.bin", create=True)
    clock = Clock()
    source = LiveCommandSource(tmp_path / "live.bin", clock=clock)
    gui.write_command(Command(clock.now, MODE_IDLE, 0, .5, 0., .25))
    assert source.poll() == (None, (.5, 0., .25))
    gui.write_command(Command(clock.now, MODE_RUN, 1, .5, 0., .25))
    assert source.poll() == ("run", (.5, 0., .25))
    assert source.poll() == (None, (.5, 0., .25))
    clock.now += .6  # the writer stopped refreshing: zero velocity, mode unchanged
    assert source.poll() == (None, (0., 0., 0.))
    gui.write_command(Command(clock.now, MODE_RUN, 2))  # a new RUN token re-prepares
    assert source.poll()[0] == "run"
    gui.write_command(Command(clock.now, MODE_IDLE, 2))
    assert source.poll()[0] == "idle" and not source.running
    assert source.cap is None  # no cap requested: the controller keeps its contract default
    gui.write_command(Command(clock.now, MODE_IDLE, 2, cap_vx=.8, cap_wz=.5))
    source.poll()
    assert source.cap == (.8, .5)
    source.close()
    gui.close()
