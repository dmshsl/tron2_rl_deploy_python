"""CPU geometry-to-policy scenarios; no simulator or hardware dependencies."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray
import pytest

from tron2_deploy.perception.height_map import (
    CameraIntrinsics, DepthFrame, HeightMap, HeightMapInputError, PolicyScan,
    DEFAULT_URDF, load_camera_extrinsics, policy_offsets,
)


def test_camera_optical_axes_when_using_payload_mount() -> None:
    # Given: the forward/rearward 45-degree downward D455 mounts.
    front, rear = load_camera_extrinsics()
    # When: transforming each optical forward axis into the base frame.
    directions = np.stack((front[:3, 2], rear[:3, 2]))
    # Then: neither camera looks sideways or upwards.
    expected = np.array([[1., 0., -1.], [-1., 0., -1.]]) / np.sqrt(2)
    np.testing.assert_allclose(directions, expected, atol=1e-5)


def pose(yaw: float = 0.0, xyz: tuple[float, float, float] = (0, 0, 0.6)) -> NDArray[np.float64]:
    c, s = np.cos(yaw), np.sin(yaw)
    result = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])
    result[:3, 3] = xyz
    return result


def reference_extrinsics(path: Path = DEFAULT_URDF) -> tuple[NDArray[np.float64], ...]:
    """Independent forward traversal with elementary axis rotations."""
    transforms = {"base_Link": np.eye(4)}
    joints = list(ET.parse(path).getroot().findall("joint"))
    for _ in range(len(joints)):
        for joint in joints:
            parent, child = joint.find("parent"), joint.find("child")
            assert parent is not None and child is not None
            if parent.attrib["link"] not in transforms:
                continue
            origin = joint.find("origin")
            local = np.eye(4)
            if origin is not None:
                local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                for axis, angle in enumerate(rpy):
                    vector = np.eye(3)[axis]
                    x, y, z = vector
                    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    rot = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew
                    local[:3, :3] = rot @ local[:3, :3]
            transforms[child.attrib["link"]] = transforms[parent.attrib["link"]] @ local
    return tuple(transforms[f"d455_{tag}_optical_frame"] for tag in ("front", "rear"))


def render_depths(world_from_base: NDArray[np.float64], height: float = 0.1,
                  step: bool = False) -> tuple[NDArray[np.float64], ...]:
    """Ray intersect horizontal surfaces (optical-axis depth, not ray length)."""
    intr = CameraIntrinsics()
    u, v = np.meshgrid(np.arange(848), np.arange(480))
    rays = np.stack(((u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones_like(u)), -1)
    depths = []
    for extrinsic in reference_extrinsics():
        transform = world_from_base @ extrinsic
        direction = rays @ transform[:3, :3].T
        origin = transform[:3, 3]
        candidates = []
        for level in ((height, height + 0.15) if step else (height,)):
            with np.errstate(divide="ignore", invalid="ignore"):
                depth = (level - origin[2]) / direction[..., 2]
            y = origin[1] + depth * direction[..., 1]
            region = (y >= 0.25) if level > height else (y < 0.25)
            good = (depth > 0) & (region if step else True)
            candidates.append(np.where(good, depth, np.inf))
        depths.append(np.minimum.reduce(candidates))
    return tuple(depths)


def frames(depths: tuple[NDArray[np.float64], ...], timestamp: float = 1.0) -> tuple[DepthFrame, ...]:
    return tuple(DepthFrame(camera, depth, timestamp)
                 for camera, depth in zip(("front", "rear"), depths, strict=True))


def drive(height: float = 0.1, step: bool = False, start_y: float = -1.2, end_y: float = 0.3,
         n: int = 60, dt: float = 0.02, x0: float = 0.12, y_off: float = -0.08, z0: float = 0.6,
         millimetres: bool = False,
         cameras: tuple[str, ...] = ("front", "rear")) -> tuple[PolicyScan, float]:
    """Advance the base along local y, the D455 boresight (roll 45, yaw 90, pitch 0 per the
    URDF): a static robot's closest valid return is >=0.52 m, outside the grid's 0.5 m
    half-width, so only a robot sweeping through the area - exactly how it is driven in
    practice - populates the near-field policy grid via the rolling memory under test.
    """
    mapper = HeightMap()
    scan = None
    for i in range(n):
        timestamp = i * dt
        y = y_off + start_y + (end_y - start_y) * (i / (n - 1))
        transform = pose(xyz=(x0, y, z0))
        depths = render_depths(transform, height=height, step=step)
        inputs = tuple(DepthFrame(camera, depth, timestamp)
                       for camera, depth in zip(("front", "rear"), depths, strict=True)
                       if camera in cameras)
        if millimetres:
            # Clamp grazing-ray depths (can reach ~1e5 m) before the uint16 cast: a real
            # sensor never reports such a range, and casting it would silently overflow.
            inputs = tuple(DepthFrame(f.camera, np.rint(np.where(np.isfinite(f.depth) & (f.depth < 65.0),
                                      f.depth, 0) * 1000).astype(np.uint16), f.timestamp) for f in inputs)
        scan = mapper.update(inputs, transform, timestamp)
    assert scan is not None
    return scan, y_off + end_y


def test_extrinsics_when_reading_full_urdf_chain() -> None:
    # Given the shipped multi-joint optical chains; When parsed; Then agree independently.
    np.testing.assert_allclose(load_camera_extrinsics(DEFAULT_URDF), reference_extrinsics(), atol=1e-12)


@pytest.mark.parametrize("millimetres", [False, True])
def test_flat_plane_when_both_cameras_observe(millimetres: bool) -> None:
    # Given a known world plane, approached by driving the base through the memory window.
    scan, _ = drive(millimetres=millimetres)
    # When the resulting policy grid is read back.
    # Then every observed cell agrees with the analytic value.
    assert scan.valid.sum() > 50
    np.testing.assert_allclose(scan.cell_z[scan.valid], 0.1, atol=0.01)
    np.testing.assert_allclose(scan.values[scan.valid], 0.6 - 0.1 - 0.8, atol=0.01)
    assert np.all(scan.values[~scan.valid] == -1)
    # And both cameras contribute independently when driven along their own boresight.
    front_only, _ = drive(start_y=-1.2, end_y=0.3, cameras=("front",))
    rear_only, _ = drive(start_y=1.2, end_y=-0.3, cameras=("rear",))
    assert front_only.valid.sum() > 20
    assert rear_only.valid.sum() > 20


def test_step_when_depths_contain_two_levels() -> None:
    # Given a 15 cm step at world y=0.25, approached by driving through the discontinuity.
    scan, base_y = drive(step=True, x0=0.0, y_off=0.0)
    # When visible policy cells away from the discontinuity are read back.
    boundary = 0.25 - base_y
    selected = scan.valid & (np.abs(policy_offsets()[:, 1] - boundary) > 0.05)
    expected = np.where(policy_offsets()[:, 1] >= boundary, 0.25, 0.1)
    # Then both analytic levels are represented and match.
    assert np.count_nonzero(selected & (expected == 0.25)) > 10
    assert np.count_nonzero(selected & (expected == 0.1)) > 10
    np.testing.assert_allclose(scan.cell_z[selected], expected[selected], atol=0.01)


@pytest.mark.parametrize("now,valid", [(5.999, True), (6.0, False), (6.1, False)])
def test_memory_when_robot_moves_past_observed_cells(now: float, valid: bool) -> None:
    # Given ground observed at world (1.2, 0) - within the forward FOV, outside the grid.
    mapper = HeightMap()
    mapper.update(frames(render_depths(pose())), pose(), 1.0)
    moved = pose(xyz=(1.2, 0, 0.6))
    blank = frames((np.zeros((480, 848)), np.zeros((480, 848))), now)
    # When the base moves onto that world cell with no fresh depth, including at expiry.
    scan = mapper.update(blank, moved, now)
    # Then the old world cell is preserved, never moved with the robot.
    k = 10 + 5 * 21  # local (0, 0), now coinciding with world (1.2, 0)
    assert bool(scan.valid[k]) == valid
    assert scan.values[k] == pytest.approx(-0.3 if valid else -1)
    assert scan.age[k] == pytest.approx(now - 1) if valid else np.isinf(scan.age[k])


def test_ordering_when_flattening_policy_grid() -> None:
    # Given the Isaac GridPattern contract; When offsets are built; Then x is fastest.
    offsets = policy_offsets()
    assert offsets.shape == (231, 2)
    np.testing.assert_allclose(offsets[:3], [[-1, -.5], [-.9, -.5], [-.8, -.5]])
    for k, xy in enumerate(offsets):
        np.testing.assert_allclose(xy, [-1 + (k % 21) * .1, -.5 + (k // 21) * .1], atol=1e-14)


def test_yaw_alignment_when_robot_turns_ninety_degrees() -> None:
    # Given a fixed world step remembered before rotation.
    mapper = HeightMap()
    mapper.update(frames(render_depths(pose(), step=True)), pose(), 1.0)
    # When querying after a 90 degree yaw without new frames.
    scan = mapper.update((), pose(np.pi / 2), 2.0)
    # Then local x now samples world y (not the unrotated local y).
    x = policy_offsets()[:, 0]
    selected = scan.valid & (np.abs(x - .25) > .05)
    assert selected.sum() > 30
    np.testing.assert_allclose(scan.cell_z[selected], np.where(x[selected] >= .25, .25, .1), atol=.01)


@pytest.mark.parametrize("invalid", [0., .519, 6.001, np.nan, np.inf, -1.])
def test_unknown_when_all_depth_is_outside_range(invalid: float) -> None:
    # Given invalid sensor values; When fused; Then they cannot fabricate terrain.
    depth = np.full((480, 848), invalid)
    scan = HeightMap().update(frames((depth, depth)), pose(), 1.0)
    assert not scan.valid.any()
    assert np.all(scan.values == -1) and np.all(np.isinf(scan.age))


def test_rollover_when_robot_teleports_outside_window() -> None:
    # Given a populated 4m map.
    mapper = HeightMap()
    mapper.update(frames(render_depths(pose())), pose(), 1.0)
    mapper.update((), pose(xyz=(10, 0, .6)), 2.0)
    # When returning, old cells outside the rolling window must not reappear.
    scan = mapper.update((), pose(), 3.0)
    assert not scan.valid.any()


def test_rejects_bad_pose_when_input_is_not_se3() -> None:
    # Given a scaled rotation; When crossing the API boundary; Then fail explicitly.
    transform = pose()
    transform[0, 0] = 2
    with pytest.raises(HeightMapInputError):
        HeightMap().update((), transform, 1.0)


def test_timing_when_updating_two_full_size_frames(capsys: pytest.CaptureFixture[str]) -> None:
    # Given two D455 frames per update at the default four-pixel stride, 30 Hz driving.
    mapper = HeightMap()
    timings, scan, n = [], None, 60
    for i in range(n):
        timestamp = i / 30
        y = -0.08 - 1.2 + 1.5 * (i / (n - 1))
        transform = pose(xyz=(0.12, y, 0.6))
        depths = render_depths(transform)
        # When running the complete CPU update, including scan production.
        start = perf_counter()
        scan = mapper.update(frames(depths, timestamp), transform, timestamp)
        timings.append((perf_counter() - start) * 1000)
    # Then report real measured time, not an environment-dependent speed gate.
    assert scan is not None and scan.valid.any()
    with capsys.disabled():
        print(f"\nCPU two 848x480 frames, stride=4: median={np.median(timings):.3f} ms "
              f"p95={np.percentile(timings, 95):.3f} ms, n={len(timings)}")
