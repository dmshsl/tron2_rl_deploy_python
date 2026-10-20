"""Live velocity commands and mode control for run_policy.py through one small shared-memory file.

A GUI (or any local process) owns the file and writes the command block; run_policy.py (--live-commands
PATH) reads it every control tick and writes the status block back. Both blocks are seqlocked (the
sequence number is odd while a writer is inside, readers retry), so a reader never sees a torn record.
Nothing here imports the SDK.

Command block (GUI -> controller)
  stamp      monotonic seconds of the last write; commands older than COMMAND_TIMEOUT read as zero
  mode       MODE_IDLE: SDK connected, limp (zero gains, no stand-up); MODE_RUN: prepare -> stand-up ->
             hand-off -> Base; going back to MODE_IDLE stops the controller (limp)
  run_token  incremented by the writer for every RUN request; a new token while running re-prepares
  vx vy wz   requested body velocity (m/s, rad/s); the controller's own command path shapes them
  cap_vx cap_wz  software cap (|vx| m/s, |wz| rad/s; 0 = keep the controller's default): run_policy.py passes a
             changed cap to PolicyWheelfootController.set_command_cap, i.e. the deploy command shaper

Status block (controller -> GUI)
  stamp, phase (controller.standup_phase: -1 idle, 0-4 seated route, 5 stand-up policy, 6 handed off),
  state (STATE_*), policy_t, vx vy wz the policy saw (the controller's shaped command), latch reason (utf-8).
"""
from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Final

MAGIC: Final = 0x54324C43  # "T2LC"
VERSION: Final = 1
MODE_IDLE: Final = 0
MODE_RUN: Final = 1
STATE_STARTING: Final = 0
STATE_IDLE: Final = 1
STATE_RUNNING: Final = 2
STATE_LATCHED: Final = 3
STATE_EXITED: Final = 4
COMMAND_TIMEOUT: Final = .5
REASON_BYTES: Final = 256

_HEADER = struct.Struct("<II")
_COMMAND = struct.Struct("<Qdiiddddd")  # seq stamp mode run_token vx vy wz cap_vx cap_wz
_STATUS = struct.Struct(f"<Qdiidddd{REASON_BYTES}s")  # seq stamp phase state policy_t vx vy wz reason
_COMMAND_AT: Final = 64
_STATUS_AT: Final = 192
SIZE: Final = 512


@dataclass(frozen=True, slots=True)
class Command:
    stamp: float = 0.
    mode: int = MODE_IDLE
    run_token: int = 0
    vx: float = 0.
    vy: float = 0.
    wz: float = 0.
    cap_vx: float = 0.
    cap_wz: float = 0.

    def fresh(self, now: float, timeout: float = COMMAND_TIMEOUT) -> bool:
        return 0. <= now - self.stamp <= timeout


@dataclass(frozen=True, slots=True)
class Status:
    stamp: float = 0.
    phase: int = -1
    state: int = STATE_STARTING
    policy_t: float = 0.
    vx: float = 0.
    vy: float = 0.
    wz: float = 0.
    reason: str = ""


class LiveChannel:
    """Mapped view of the file. create=True (the GUI) makes a zeroed file; run_policy opens an existing one."""

    def __init__(self, path: Path, create: bool = False) -> None:
        self.path = Path(path)
        if create:
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            os.ftruncate(fd, SIZE)
        else:
            fd = os.open(self.path, os.O_RDWR | os.O_NOFOLLOW)
            if os.fstat(fd).st_size < SIZE:
                os.close(fd)
                raise ValueError(f"{self.path} is not a live command file")
        try:
            self.map = mmap.mmap(fd, SIZE)
        finally:
            os.close(fd)
        if create:
            _HEADER.pack_into(self.map, 0, MAGIC, VERSION)
            self.write_command(Command())
            self.write_status(Status())
        elif _HEADER.unpack_from(self.map, 0) != (MAGIC, VERSION):
            raise ValueError(f"{self.path}: bad live command header")

    def _write(self, layout: struct.Struct, offset: int, values: tuple) -> None:
        odd = (struct.unpack_from("<Q", self.map, offset)[0] + 1) | 1
        struct.pack_into("<Q", self.map, offset, odd)  # odd: a writer is inside
        layout.pack_into(self.map, offset, odd, *values)
        struct.pack_into("<Q", self.map, offset, odd + 1)  # even: record complete

    def _read(self, layout: struct.Struct, offset: int) -> tuple:
        """Consistent record; TimeoutError if a writer stays inside for ~2 ms (descheduled mid-write, or died
        there) - callers keep their previous record then."""
        deadline = monotonic() + .002
        while True:
            values = layout.unpack_from(self.map, offset)
            if values[0] % 2 == 0 and struct.unpack_from("<Q", self.map, offset)[0] == values[0]:
                return values[1:]
            if monotonic() > deadline:
                raise TimeoutError("live command block stayed locked")
            sleep(0)

    def write_command(self, command: Command) -> None:
        self._write(_COMMAND, _COMMAND_AT, (command.stamp, command.mode, command.run_token, command.vx,
                                            command.vy, command.wz, command.cap_vx, command.cap_wz))

    def read_command(self) -> Command:
        return Command(*self._read(_COMMAND, _COMMAND_AT))

    def write_status(self, status: Status) -> None:
        reason = status.reason.encode("utf-8")[:REASON_BYTES]
        self._write(_STATUS, _STATUS_AT, (status.stamp, status.phase, status.state, status.policy_t,
                                          status.vx, status.vy, status.wz, reason))

    def read_status(self) -> Status:
        stamp, phase, state, policy_t, vx, vy, wz, reason = self._read(_STATUS, _STATUS_AT)
        return Status(stamp, phase, state, policy_t, vx, vy, wz, reason.rstrip(b"\0").decode("utf-8", "replace"))

    def close(self) -> None:
        self.map.close()


class LiveCommandSource:
    """run_policy.py side: per-tick mode transitions and fresh-or-zero velocity commands."""

    def __init__(self, path: Path, clock=monotonic) -> None:
        self.channel = LiveChannel(path)
        self.clock = clock
        self.last = self.channel.read_command()
        self.token = self.last.run_token
        self.running = False
        self.cap: tuple[float, float] | None = None

    def poll(self) -> tuple[str | None, tuple[float, float, float]]:
        """(transition, (vx, vy, wz)); transition is "run" (prepare now), "idle" (stop) or None. The latest
        requested cap is in self.cap ((vx, wz) or None when the writer sets none)."""
        try:
            command = self.last = self.channel.read_command()
        except TimeoutError:  # the writer is mid-record: use the previous one (it still goes stale on time)
            command = self.last
        transition = None
        if command.mode == MODE_RUN and (not self.running or command.run_token != self.token):
            transition, self.running = "run", True
        elif command.mode != MODE_RUN and self.running:
            transition, self.running = "idle", False
        self.token = command.run_token
        self.cap = (command.cap_vx, command.cap_wz) if command.cap_vx > 0 and command.cap_wz > 0 else None
        velocity = (command.vx, command.vy, command.wz) if command.fresh(self.clock()) else (0., 0., 0.)
        return transition, velocity

    def report(self, phase: int, state: int, policy_t: float, velocity, reason: str) -> None:
        self.channel.write_status(Status(self.clock(), int(phase), int(state), float(policy_t),
                                         float(velocity[0]), float(velocity[1]), float(velocity[2]), reason))

    def close(self) -> None:
        self.channel.close()
