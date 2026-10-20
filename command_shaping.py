"""Velocity command shaping between every command source and the policy.

Order, applied to (vx, vy, wz) on every controller tick:

1. non-finite components -> 0 (a NaN from a GUI/joystick never trips the safety latch);
2. hard clamp to the trained range (contract ``command_ranges``: the hardware maxima);
3. software cap ``|vx| <= cap_vx``, ``|wz| <= cap_wz`` (contract ``commands.default_cap``,
   changeable at runtime with :meth:`CommandShaper.set_cap`, never above the trained range);
4. optional lateral-acceleration limit ``|vx * wz| <= lateral_accel_max`` (wz is reduced, speed kept);
5. slew-rate limiter: the output moves toward the target at ``accel`` while speeding up and at
   ``decel`` while slowing down or reversing through zero (piecewise-constant acceleration).

Pure numpy; no SDK, ONNX or contract-file I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

AXES = 3  # vx, vy, wz


@dataclass(frozen=True)
class CommandLimits:
    """Per-axis (vx, vy, wz) bounds; ``cap`` is the software cap below ``high``/``-low``."""

    low: tuple[float, float, float]
    high: tuple[float, float, float]
    cap: tuple[float, float, float]
    accel: tuple[float, float, float]
    decel: tuple[float, float, float]
    lateral_accel_max: float | None = None

    def __post_init__(self) -> None:
        for name in ("low", "high", "cap", "accel", "decel"):
            values = getattr(self, name)
            if len(values) != AXES or not all(math.isfinite(v) for v in values):
                raise ValueError(f"{name} must be {AXES} finite values")
        if any(lo > 0 or hi < 0 for lo, hi in zip(self.low, self.high, strict=True)):
            raise ValueError("command range must contain zero")
        if any(c < 0 for c in self.cap) or any(a <= 0 for a in self.accel + self.decel):
            raise ValueError("cap must be >= 0 and accelerations > 0")
        if self.lateral_accel_max is not None and not self.lateral_accel_max > 0:
            raise ValueError("lateral_accel_max must be positive")


class CommandShaper:
    """Stateful shaper; call once per control tick with the controller timestamp."""

    def __init__(self, limits: CommandLimits) -> None:
        self._limits = self._bounded(limits)
        self._output = np.zeros(AXES, dtype=np.float64)
        self._time: float | None = None
        self.target = np.zeros(AXES, dtype=np.float32)

    @staticmethod
    def _bounded(limits: CommandLimits) -> CommandLimits:
        reach = tuple(max(-lo, hi) for lo, hi in zip(limits.low, limits.high, strict=True))
        cap = tuple(min(c, r) for c, r in zip(limits.cap, reach, strict=True))
        return replace(limits, cap=cap)

    @property
    def limits(self) -> CommandLimits:
        return self._limits

    @property
    def output(self) -> np.ndarray:
        return self._output.astype(np.float32)

    def set_cap(self, vx: float | None = None, wz: float | None = None) -> tuple[float, float]:
        """Change the software cap at runtime (clipped to the trained range); returns (cap_vx, cap_wz)."""
        cap = list(self._limits.cap)
        for axis, value in ((0, vx), (2, wz)):
            if value is None:
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"command cap must be finite and >= 0, got {value}")
            cap[axis] = value
        self._limits = self._bounded(replace(self._limits, cap=tuple(cap)))
        return self._limits.cap[0], self._limits.cap[2]

    def reset(self, value: np.ndarray | None = None) -> None:
        """Restart from rest (or ``value``); the next call does not slew from the old output."""
        self._output = np.zeros(AXES) if value is None else np.asarray(value, dtype=np.float64).copy()
        self._time = None

    def clamp(self, raw: np.ndarray) -> np.ndarray:
        """Steps 1-4: the target the slew limiter moves toward."""
        lim = self._limits
        cmd = np.asarray(raw, dtype=np.float64).reshape(-1)
        if cmd.shape != (AXES,):
            raise ValueError(f"commands must have {AXES} components, got {cmd.shape}")
        cmd = np.where(np.isfinite(cmd), cmd, 0.0)
        cmd = np.clip(cmd, lim.low, lim.high)
        cap = np.asarray(lim.cap)
        cmd = np.clip(cmd, -cap, cap)
        if lim.lateral_accel_max is not None and abs(cmd[0]) > 1e-6:
            wz_max = lim.lateral_accel_max / abs(cmd[0])
            cmd[2] = float(np.clip(cmd[2], -wz_max, wz_max))
        return cmd

    def __call__(self, raw: np.ndarray, timestamp: float) -> np.ndarray:
        target = self.clamp(raw)
        self.target = target.astype(np.float32)
        dt = 0.0 if self._time is None else max(0.0, timestamp - self._time)
        self._time = timestamp
        lim = self._limits
        out = self._output
        for axis in range(AXES):
            delta = target[axis] - out[axis]
            # slowing down (toward / through zero) uses decel, speeding up uses accel
            braking = out[axis] * delta < 0
            rate = lim.decel[axis] if braking else lim.accel[axis]
            step = rate * dt
            if braking and abs(delta) > abs(out[axis]):
                # reversal: brake to zero at decel, then speed up at accel within the same tick
                t_zero = abs(out[axis]) / lim.decel[axis]
                if t_zero >= dt:
                    out[axis] -= math.copysign(step, out[axis])
                else:
                    rest = min(lim.accel[axis] * (dt - t_zero), abs(target[axis]))
                    out[axis] = math.copysign(rest, target[axis])
                continue
            out[axis] += float(np.clip(delta, -step, step))
        out[:] = np.clip(out, lim.low, lim.high)
        return out.astype(np.float32)


def limits_from_contract(contract) -> CommandLimits:
    """CommandLimits from a DeploymentContract (command_ranges + commands section)."""
    ranges = contract.command_ranges
    cfg = contract.commands
    return CommandLimits(
        low=tuple(float(r[0]) for r in ranges), high=tuple(float(r[1]) for r in ranges),
        cap=(cfg.default_cap[0], max(-ranges[1][0], ranges[1][1]), cfg.default_cap[1]),
        accel=(cfg.accel[0], cfg.accel[0], cfg.accel[1]), decel=(cfg.decel[0], cfg.decel[0], cfg.decel[1]),
        lateral_accel_max=cfg.lateral_accel_max)
