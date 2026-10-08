"""Recording boundary tests with small synthetic, correctly clocked streams."""
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from tron2_deploy.perception.height_map import CameraIntrinsics, policy_offsets
from tron2_deploy.tools.perception_recording import RecordingError, load_recording


def recording(path: Path, change: tuple[str, NDArray] | None = None) -> Path:
    n, m = 60, 36
    pose = np.repeat(np.eye(4)[None], n, axis=0)
    pose[:, 2, 3] = .5
    hits = np.zeros((n, 231, 3))
    hits[:, :, :2] = policy_offsets()
    intrinsics = CameraIntrinsics()
    arrays = {
        "timestamp": np.arange(1, n + 1) * .02,
        "depth_timestamp": np.ceil(np.arange(1, m + 1) / 30 / .005 - 1e-8) * .005,
        "q": np.zeros((n, 10)), "dq": np.zeros((n, 10)),
        "imu_wxyz": np.tile([1., 0., 0., 0.], (n, 1)), "gyro": np.zeros((n, 3)),
        "gt_pose": pose, "ray_hits": hits, "raw_scan": np.full((n, 231), -.3),
        "reset": np.zeros(n, dtype=bool), "depth_front": np.zeros((m, 480, 848), dtype=np.float32),
        "depth_rear": np.zeros((m, 480, 848), dtype=np.float32),
        "intrinsics": np.array([intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]),
        "imu_to_base": np.eye(3), "scan_offsets": policy_offsets(), "recording_version": np.array(1),
        "commands": np.tile([.5, 0., 0.], (n, 1)),
        "camera_frame": np.repeat(np.arange(m)[:, None], 2, axis=1),
        "camera_pose": np.zeros((m, 2, 7)), "capture_gt_pose": pose[:m],
    }
    if change is not None:
        arrays[change[0]] = change[1]
    np.savez_compressed(path, **arrays)
    return path


@pytest.mark.parametrize("field,value,reason", [
    ("reset", np.zeros(59, dtype=bool), "reset: expected shape"),
    ("reset", np.zeros(60), "reset must be boolean"),
    ("depth_timestamp", np.full(36, np.nan), "timestamps"),
    ("scan_offsets", policy_offsets()[::-1], "scan order"),
    ("camera_frame", np.zeros((36, 2), dtype=np.int64), "unique frame"),
    ("camera_frame", np.full((36, 2), np.nan), "integer frame IDs"),
    ("camera_frame", np.repeat((np.arange(36) + .5)[:, None], 2, axis=1), "integer frame IDs"),
    ("camera_frame", np.repeat((np.arange(36) - 1)[:, None], 2, axis=1), "integer frame IDs"),
    ("depth_rear", np.zeros((35, 480, 848)), "depth_rear: expected shape"),
])
def test_rejects_malformed_recording(tmp_path: Path, field: str, value: NDArray, reason: str) -> None:
    # Given: a correctly clocked recording with one corrupt array.
    path = recording(tmp_path / "bad.npz", (field, value))
    # When/Then: rejection identifies the contract violation at the boundary.
    with pytest.raises(RecordingError, match=reason):
        load_recording(path)


def test_loads_valid_recording(tmp_path: Path) -> None:
    # Given: a complete standalone synthetic recording.
    path = recording(tmp_path / "valid.npz")
    # When: crossing the file boundary.
    result = load_recording(path)
    # Then: validated sample dimensions are preserved.
    assert result.q.shape == (60, 10)
    assert result.depth_timestamp.shape == (36,)
