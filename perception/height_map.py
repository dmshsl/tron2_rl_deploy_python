"""NumPy CPU elevation memory, with no simulator, SDK, or middleware imports.

Depth is rectified optical-axis Z, uint16 millimetres or floating-point metres.
All timestamps are seconds on one monotonic clock. The supplied world<-base_Link
pose must correspond to the frames: synchronize cameras/odometry upstream, or
submit each camera separately with its capture-time pose. Empty updates only
recenter/expire memory and query a scan. FOV intrinsics are nominal, not a
substitute for device calibration. No distortion or robot self-filter is applied.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
DepthArray = NDArray[np.uint16] | NDArray[np.float32] | FloatArray
CameraName = Literal["front", "rear"]
DEFAULT_URDF: Final = (Path(__file__).resolve().parents[2] / "tron2_description"
                       / "tron2a/WF_TRON2A/urdf/robot_cam45.urdf")


@dataclass(frozen=True, slots=True)
class HeightMapInputError(ValueError):
    """An invalid frame, clock, calibration, or rigid transform at the boundary."""

    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Rectified 848x480 pinhole intrinsics; defaults derive from 87x58 degree FOV."""

    fx: float = float(848 / (2 * np.tan(np.deg2rad(87 / 2))))
    fy: float = float(480 / (2 * np.tan(np.deg2rad(58 / 2))))
    cx: float = 848 / 2
    cy: float = 480 / 2

    def __post_init__(self) -> None:
        if not np.isfinite((self.fx, self.fy, self.cx, self.cy)).all() or min(self.fx, self.fy) <= 0:
            raise HeightMapInputError("intrinsics must be finite with positive focal lengths")


@dataclass(frozen=True, slots=True)
class DepthFrame:
    """One camera capture; arrays are borrowed only for the duration of update."""

    camera: CameraName
    depth: DepthArray
    timestamp: float


@dataclass(frozen=True, slots=True)
class PolicyScan:
    """231 x-fastest values; reshape to (11,21), not (21,11).

    Unknown cells have value -1, age +inf and cell_z NaN. Validity, not the
    numeric value, distinguishes unknown from a valid clipped -1 reading.
    """

    values: FloatArray
    valid: NDArray[np.bool_]
    age: FloatArray
    cell_z: FloatArray


def policy_offsets() -> FloatArray:
    """Isaac GridPattern offsets: k -> (k % 21, k // 21), in base yaw axes."""
    x, y = np.meshgrid(np.arange(-10, 11) / 10, np.arange(-5, 6) / 10)
    return np.column_stack((x.ravel(), y.ravel()))


def load_camera_extrinsics(path: Path = DEFAULT_URDF) -> tuple[FloatArray, FloatArray]:
    """Compose fixed URDF chains to return base_Link<-front/rear optical frames."""
    edges: dict[str, tuple[str, FloatArray, str]] = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            raise HeightMapInputError("URDF joint is missing parent or child")
        transform = np.eye(4)
        origin = joint.find("origin")
        if origin is not None:
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite([xyz, rpy]).all():
                raise HeightMapInputError("URDF origin must contain finite xyz/rpy triples")
            cr, cp, cy = np.cos(rpy)
            sr, sp, sy = np.sin(rpy)
            transform[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                                  [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                                  [-sp, cp*sr, cp*cr]]
            transform[:3, 3] = xyz
        edges[child.attrib["link"]] = (parent.attrib["link"], transform, joint.get("type", ""))
    result = []
    for tag in ("front", "rear"):
        link, transform, visited = f"d455_{tag}_optical_frame", np.eye(4), set()
        while link != "base_Link":
            if link in visited or link not in edges:
                raise HeightMapInputError(f"broken or cyclic optical chain at {link}")
            visited.add(link)
            link, local, kind = edges[link]
            if kind != "fixed":
                raise HeightMapInputError("camera extrinsic chain must contain only fixed joints")
            transform = local @ transform
        result.append(transform)
    return result[0], result[1]


class HeightMap:
    """Mutable 80x80 world-lattice memory, recentered at 5cm increments.

    Each cell retains its maximum unexpired z and that height's capture timestamp.
    Lower returns do not refresh an old maximum; it expires at age >=5 seconds.
    Equal-height returns refresh it. No interpolation fabricates unknown cells.
    CPU is intentional: NumPy is the only dependency, including on GPU hosts.
    """

    def __init__(self, urdf_path: Path = DEFAULT_URDF, decimation: int = 4,
                 intrinsics: tuple[CameraIntrinsics, CameraIntrinsics] | None = None) -> None:
        if not isinstance(decimation, int) or decimation < 1:
            raise HeightMapInputError("decimation must be a positive integer pixel stride")
        calibration = intrinsics if intrinsics is not None else (CameraIntrinsics(), CameraIntrinsics())
        if len(calibration) != 2:
            raise HeightMapInputError("provide front and rear intrinsics")
        self._extrinsics = load_camera_extrinsics(urdf_path)
        self._stride = decimation
        u, v = np.meshgrid(np.arange(0, 848, decimation), np.arange(0, 480, decimation))
        self._rays = tuple(np.stack(((u-i.cx)/i.fx, (v-i.cy)/i.fy, np.ones_like(u)), -1)
                           .reshape(-1, 3) for i in calibration)
        self._z = np.full((80, 80), -np.inf)
        self._stamp = np.full((80, 80), -np.inf)
        self._origin = np.array([-40, -40], dtype=np.int64)
        self._last_update = -np.inf
        self._last_frame = {"front": -np.inf, "rear": -np.inf}

    def update(self, frames: tuple[DepthFrame, ...], world_from_base: FloatArray,
               timestamp: float) -> PolicyScan:
        """Fuse zero to two captures and return the yaw-aligned policy scan.

        Reject malformed inputs before mutating memory. Repeated camera timestamps
        are ignored (cannot refresh/reproject an old capture); regressions reject.
        timestamp is query time, >= frame times and >= previous query time.
        """
        transform = np.asarray(world_from_base, dtype=np.float64)
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-6)):
            raise HeightMapInputError("world_from_base must be an SE(3) 4x4 matrix")
        if not np.isfinite(timestamp) or timestamp < self._last_update:
            raise HeightMapInputError("query timestamp must be finite and monotonic")
        if len(frames) > 2 or len({f.camera for f in frames}) != len(frames):
            raise HeightMapInputError("at most one capture per camera per update")
        for frame in frames:
            if frame.camera not in self._last_frame or frame.depth.shape != (480, 848):
                raise HeightMapInputError("expected front/rear depth of shape (480,848)")
            if frame.depth.dtype != np.uint16 and not np.issubdtype(frame.depth.dtype, np.floating):
                raise HeightMapInputError("depth must be uint16 mm or floating-point metres")
            if (not np.isfinite(frame.timestamp) or frame.timestamp > timestamp
                    or frame.timestamp < self._last_frame[frame.camera]):
                raise HeightMapInputError("frame timestamps must be finite, monotonic, and <= query time")
        self._recenter(transform[:2, 3])
        expired = timestamp - self._stamp >= 5.0
        self._z[expired], self._stamp[expired] = -np.inf, -np.inf
        for frame in frames:
            if frame.timestamp > self._last_frame[frame.camera] and timestamp - frame.timestamp < 5.0:
                camera = ("front", "rear").index(frame.camera)
                depth = frame.depth[::self._stride, ::self._stride].astype(np.float64).ravel()
                depth *= 0.001 if frame.depth.dtype == np.uint16 else 1.0
                valid = np.isfinite(depth) & (depth >= 0.52) & (depth <= 6.0)
                optical = self._rays[camera][valid] * depth[valid, None]
                world = transform @ self._extrinsics[camera]
                points = optical @ world[:3, :3].T + world[:3, 3]
                self._fuse(points, frame.timestamp)
            self._last_frame[frame.camera] = frame.timestamp
        self._last_update = timestamp
        yaw = np.arctan2(transform[1, 0], transform[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        xy = policy_offsets() @ np.array([[c, s], [-s, c]]) + transform[:2, 3]
        cells = np.floor(xy / 0.05 + 0.5).astype(np.int64) - self._origin
        z, stamp = self._z[cells[:, 1], cells[:, 0]], self._stamp[cells[:, 1], cells[:, 0]]
        valid = np.isfinite(z)
        return PolicyScan(np.where(valid, np.clip(transform[2, 3] - z - 0.8, -1, 1), -1),
                          valid, np.where(valid, timestamp - stamp, np.inf), np.where(valid, z, np.nan))

    def _recenter(self, xy: FloatArray) -> None:
        origin = np.floor(xy / 0.05 + 0.5).astype(np.int64) - 40
        shift = self._origin - origin
        if np.any(shift):
            y, x = np.indices((80, 80))
            nx, ny = x + shift[0], y + shift[1]
            keep = (nx >= 0) & (nx < 80) & (ny >= 0) & (ny < 80)
            z, stamp = np.full((80, 80), -np.inf), np.full((80, 80), -np.inf)
            z[ny[keep], nx[keep]], stamp[ny[keep], nx[keep]] = self._z[keep], self._stamp[keep]
            self._z, self._stamp, self._origin = z, stamp, origin

    def _fuse(self, points: FloatArray, timestamp: float) -> None:
        cells = np.floor(points[:, :2] / 0.05 + 0.5).astype(np.int64) - self._origin
        keep = ((cells >= 0) & (cells < 80)).all(axis=1)
        cells, heights = cells[keep], points[keep, 2]
        indices = cells[:, 1] * 80 + cells[:, 0]
        maximum = np.full(6400, -np.inf)
        np.maximum.at(maximum, indices, heights)
        z, stamp = self._z.ravel(), self._stamp.ravel()
        replace = np.isfinite(maximum) & ((maximum > z) | ((maximum == z) & (timestamp > stamp)))
        z[replace], stamp[replace] = maximum[replace], timestamp
