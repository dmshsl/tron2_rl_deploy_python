"""Replay task-15 recordings; GT enters only scoring and a separate diagnostic.

python3 tron2_deploy/tools/eval_perception.py --rec DIR --json OUT
Depth is fused at its interpolated capture pose once the bracketing 50 Hz sensor
tick arrives (zero delay for coincident ticks). No future depth or GT is input.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tron2_deploy.perception.height_map import CameraIntrinsics, DepthFrame, HeightMap
from tron2_deploy.perception.odometry import Odometry, OdometryInput
from tron2_deploy.tools.perception_metrics import ScanSeries, TerrainMetrics, summarize
from tron2_deploy.tools.perception_recording import RecordingError, load_recording


def interpolate(poses: NDArray[np.float64], fraction: float) -> NDArray[np.float64]:
    """Interpolate between the two sensor ticks bracketing a camera timestamp."""
    pose = np.eye(4)
    pose[:3, :3] = Slerp([0., 1.], Rotation.from_matrix(poses[:, :3, :3]))(fraction).as_matrix()
    pose[:3, 3] = poses[0, :3, 3] * (1 - fraction) + poses[1, :3, 3] * fraction
    return pose


def evaluate(path: Path, output: Path) -> TerrainMetrics:
    """Run the unchanged CPU perception pipeline and save a diagnostic overlay."""
    rec = load_recording(path)
    times, frame_times = rec.timestamp, rec.depth_timestamp
    intrinsics = CameraIntrinsics(*rec.intrinsics)
    calibration = (intrinsics, intrinsics)
    mapper, oracle = HeightMap(intrinsics=calibration), HeightMap(intrinsics=calibration)
    fresh = HeightMap(intrinsics=calibration)
    odometry = Odometry(imu_to_base=rec.imu_to_base)
    poses, estimates, valid, truth_estimates, truth_valid = [], [], [], [], []
    distance, latency, cost = [], [], []
    fresh_estimates, fresh_valid = [], []
    frame = 0
    traveled = 0.
    segment_distance = 0.
    segment_start = 0
    drift_error = 0.
    for i, timestamp in enumerate(times):
        if i and rec.reset[i - 1]:
            segment_start = i
            segment_distance = 0.
            mapper, oracle = HeightMap(intrinsics=calibration), HeightMap(intrinsics=calibration)
            fresh = HeightMap(intrinsics=calibration)
            odometry = Odometry(imu_to_base=rec.imu_to_base)
        state = odometry.update(OdometryInput(float(timestamp), rec.imu_wxyz[i],
                                rec.gyro[i], rec.q[i], rec.dq[i]))
        poses.append(state.pose)
        if i > segment_start:
            increment = float(np.linalg.norm(rec.gt_pose[i, :2, 3] - rec.gt_pose[i - 1, :2, 3]))
            traveled += increment
            segment_distance += increment
        distance.append(segment_distance)
        while frame < len(frame_times) and frame_times[frame] <= timestamp + 1e-9:
            capture = float(frame_times[frame])
            if capture >= times[segment_start]:
                fraction = float(np.clip((capture - times[i - 1]) / .02, 0, 1))
                capture_pose = state.pose if i == segment_start else interpolate(np.stack(poses[-2:]), fraction)
                gt_capture_pose = rec.gt_pose[i] if i == segment_start else interpolate(rec.gt_pose[i - 1:i + 1], fraction)
                frames = (DepthFrame("front", rec.depth_front[frame], capture),
                          DepthFrame("rear", rec.depth_rear[frame], capture))
                started = perf_counter()
                mapper.update(frames, capture_pose, capture)
                cost.append(perf_counter() - started)
                oracle.update(frames, gt_capture_pose, capture)
                fresh = HeightMap(intrinsics=calibration)
                fresh.update(frames, gt_capture_pose, capture)
                if timestamp > 1.:
                    latency.append((timestamp - capture) / .02)
            frame += 1
        scan = mapper.update((), state.pose, float(timestamp))
        diagnostic = oracle.update((), rec.gt_pose[i], float(timestamp))
        instantaneous = fresh.update((), rec.gt_pose[i], float(timestamp))
        fresh_estimates.append(instantaneous.values)
        fresh_valid.append(instantaneous.valid)
        estimates.append(scan.values)
        valid.append(scan.valid)
        truth_estimates.append(diagnostic.values)
        truth_valid.append(diagnostic.valid)
        if rec.reset[i] or i == len(times) - 1:
            gt_delta = rec.gt_pose[i, :2, 3] - rec.gt_pose[segment_start, :2, 3]
            drift_error += float(np.linalg.norm(state.pose[:2, 3] - gt_delta))
    truth = np.where(np.isfinite(rec.ray_hits).all(axis=2), rec.raw_scan, np.nan)
    series = ScanSeries(times, np.array(estimates), truth, np.array(valid),
                        np.array(distance), np.array(latency), drift_error / max(traveled, 1e-12))
    metrics = summarize(series)
    diagnostic = summarize(ScanSeries(times, np.array(truth_estimates), truth,
                            np.array(truth_valid), np.array(distance), np.array(latency), 0.))
    fresh_diagnostic = summarize(ScanSeries(times, np.array(fresh_estimates), truth,
                                np.array(fresh_valid), np.array(distance), np.array(latency), 0.))
    print(f"{path.stem}: {json.dumps(asdict(metrics))}")
    print(f"  GT-pose diagnostic RMSE={diagnostic.rmse_m} unknown={diagnostic.unknown_frac_mean:.4f} "
          f"travel={traveled:.3f}m resets={int(rec.reset.sum())} "
          f"CPU fusion mean={1000 * np.mean(cost):.2f}ms p95={1000 * np.percentile(cost, 95):.2f}ms")
    print(f"  GT-pose WITHOUT memory diagnostic RMSE={fresh_diagnostic.rmse_m} "
          f"unknown={fresh_diagnostic.unknown_frac_mean:.4f}")
    np.savez_compressed(output / f"{path.stem}_evaluation.npz", timestamp=times,
                        estimate=series.estimate, valid=series.valid, truth=truth,
                        odom_pose=np.array(poses), distance=distance,
                        gt_map=truth_estimates, gt_valid=truth_valid)
    overlay(series, output / f"{path.stem}_overlay.png")
    return metrics


def overlay(series: ScanSeries, path: Path) -> None:
    """Show a fixed middle-time map and the complete coverage/error timelines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    index = len(series.timestamp) // 2
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, values, title in (
        (axes[0, 0], np.where(series.valid[index], series.estimate[index], np.nan), "Camera + odometry"),
        (axes[0, 1], series.truth[index], "Ray-cast truth"),
    ):
        image = axis.imshow(values.reshape(11, 21), origin="lower", extent=(-1.05, 1.05, -.55, .55),
                            vmin=-.8, vmax=.2, cmap="viridis")
        axis.set(title=f"{title} at {series.timestamp[index]:.2f}s", xlabel="base yaw x (m)", ylabel="y (m)")
        fig.colorbar(image, ax=axis, label="raw scan (m)")
    axes[1, 0].plot(series.timestamp, 1 - series.valid.mean(axis=1))
    axes[1, 0].set(xlabel="recording age (s)", ylabel="unknown fraction", ylim=(0, 1))
    errors = np.where(series.valid & np.isfinite(series.truth), series.estimate - series.truth, np.nan)
    counts = np.isfinite(errors).sum(axis=1)
    rms = np.sqrt(np.nansum(errors**2, axis=1) / np.maximum(counts, 1))
    axes[1, 1].plot(series.timestamp, np.where(counts > 0, rms, np.nan))
    axes[1, 1].set(xlabel="recording age (s)", ylabel="joint-valid RMSE (m)")
    fig.suptitle(path.stem)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rec", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    recordings, per_terrain = {}, {}
    passed = True
    for terrain in ("flat", "stairs", "rubble"):
        path = args.rec / f"{terrain}.npz"
        digest = hashlib.md5()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        recordings[terrain] = digest.hexdigest()
        metrics = evaluate(path, args.json.parent)
        per_terrain[terrain] = asdict(metrics)
        limit = {"flat": .02, "stairs": .04}.get(terrain)
        rmse_ok = limit is None or (metrics.rmse_m is not None and metrics.rmse_m < limit)
        fill_ok = metrics.blind_zone_fill is not None and metrics.blind_zone_fill > .8
        passed = passed and rmse_ok and fill_ok
        print(f"GATE {terrain}: rmse={'PASS' if rmse_ok else 'FAIL'} blind={'PASS' if fill_ok else 'FAIL'}")
    args.json.write_text(json.dumps({"version": 1, "recordings": recordings,
                                   "per_terrain": per_terrain}, indent=2, allow_nan=False) + "\n")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        status = main()
    except (OSError, RecordingError) as error:
        print(f"{error}\nRESULT: FAIL")
        status = 2
    raise SystemExit(status)
