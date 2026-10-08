"""Validated task-15 recording boundary, including clocks and ray ordering."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tron2_deploy.perception.height_map import policy_offsets

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RecordingError(ValueError):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class Recording:
    timestamp: FloatArray
    depth_timestamp: FloatArray
    q: FloatArray
    dq: FloatArray
    imu_wxyz: FloatArray
    gyro: FloatArray
    gt_pose: FloatArray
    raw_scan: FloatArray
    ray_hits: FloatArray
    reset: NDArray[np.bool_]
    depth_front: NDArray[np.float32]
    depth_rear: NDArray[np.float32]
    intrinsics: FloatArray
    imu_to_base: FloatArray


def load_recording(path: Path) -> Recording:
    with np.load(path, allow_pickle=False) as archive:
        required = set(Recording.__dataclass_fields__) | {"scan_offsets", "recording_version", "commands",
                                                        "camera_frame", "camera_pose", "capture_gt_pose"}
        if missing := required - set(archive.files):
            raise RecordingError(path, f"missing arrays: {sorted(missing)}")
        arrays = {key: archive[key] for key in required}
    times, captures = arrays["timestamp"], arrays["depth_timestamp"]
    if (times.ndim != 1 or captures.ndim != 1 or len(times) < 51 or not len(captures)
            or not np.isfinite(times).all() or not np.isfinite(captures).all()
            or times[0] < 0 or times[-1] <= 1.
            or not np.allclose(np.diff(times), .02, atol=1e-8)
            or np.any(np.diff(captures) <= 0) or captures[0] < times[0]
            or captures[-1] > times[-1] + 1e-9):
        raise RecordingError(path, "invalid policy/camera timestamps or empty scoring population")
    n, m = len(times), len(captures)
    scheduled = np.ceil(np.arange(1, m + 1) / 30 / .005 - 1e-8) * .005
    if m != round(times[-1] * 30) or not np.allclose(captures, scheduled, atol=1e-8):
        raise RecordingError(path, "depth stream must follow the 30 Hz deadlines on the 200 Hz clock")
    shapes = {"q": (n, 10), "dq": (n, 10), "imu_wxyz": (n, 4), "gyro": (n, 3),
              "gt_pose": (n, 4, 4), "raw_scan": (n, 231), "ray_hits": (n, 231, 3),
              "reset": (n,), "depth_front": (m, 480, 848), "depth_rear": (m, 480, 848),
              "intrinsics": (4,), "imu_to_base": (3, 3), "scan_offsets": (231, 2),
              "commands": (n, 3), "camera_frame": (m, 2), "camera_pose": (m, 2, 7),
              "capture_gt_pose": (m, 4, 4)}
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise RecordingError(path, f"{name}: expected shape {shape}, got {arrays[name].shape}")
    if arrays["reset"].dtype != np.bool_:
        raise RecordingError(path, "reset must be boolean; reset[i] occurs AFTER sensor sample i")
    finite = ("q", "dq", "imu_wxyz", "gyro", "gt_pose", "intrinsics", "imu_to_base", "commands",
              "camera_pose", "capture_gt_pose")
    if any(not np.isfinite(arrays[name]).all() for name in finite):
        raise RecordingError(path, "nonfinite state or calibration")
    frame_ids = arrays["camera_frame"]
    if not np.issubdtype(frame_ids.dtype, np.integer) or np.any(frame_ids < 0):
        raise RecordingError(path, "camera_frame must contain nonnegative integer frame IDs")
    frame_segments = np.searchsorted(times[arrays["reset"]], captures, side="left")
    repeated = (frame_ids[1:] <= frame_ids[:-1]).any(axis=1)
    if (arrays["recording_version"].shape != () or arrays["recording_version"].item() != 1
            or not np.allclose(arrays["scan_offsets"], policy_offsets(), atol=1e-6)
            or not np.allclose(arrays["commands"], [.5, 0., 0.], atol=1e-6)
            or np.any(repeated & (np.diff(frame_segments) == 0))):
        raise RecordingError(path, "version, x-fastest scan order, pinned command or unique frame violation")
    pose, hits = arrays["gt_pose"], arrays["ray_hits"]
    yaw = np.arctan2(pose[:, 1, 0], pose[:, 0, 0])
    valid = np.isfinite(hits).all(axis=2)
    delta = np.where(valid[:, :, None], hits[:, :, :2], 0.) - pose[:, None, :2, 3]
    local = np.stack((delta[:, :, 0] * np.cos(yaw[:, None]) + delta[:, :, 1] * np.sin(yaw[:, None]),
                      -delta[:, :, 0] * np.sin(yaw[:, None]) + delta[:, :, 1] * np.cos(yaw[:, None])), -1)
    if not np.allclose(local[valid], np.broadcast_to(policy_offsets(), local.shape)[valid], atol=2e-3):
        raise RecordingError(path, "ray-hit coordinates disagree with scan order/base pose")
    expected_scan = np.clip(pose[:, None, 2, 3] - hits[:, :, 2] - .8, -1, 1)
    if not np.allclose(arrays["raw_scan"][valid], expected_scan[valid], atol=1e-6):
        raise RecordingError(path, "raw scan disagrees with ray-hit heights")
    return Recording(*(arrays[name] for name in Recording.__dataclass_fields__))
