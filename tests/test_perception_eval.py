"""Metric population regression tests independent of Isaac."""
import numpy as np
import pytest
from pathlib import Path
import subprocess
import sys


def test_population_when_warmup_and_unknown_cells_present() -> None:
    from tron2_deploy.tools.perception_metrics import summarize, ScanSeries
    # Given: the warmup has large errors; one of 231 later cells is unknown.
    times = np.array([.5, 1.02, 2.02])
    estimate = np.zeros((3, 231))
    truth = np.zeros_like(estimate)
    estimate[0] = 10
    estimate[1:] = .01
    valid = np.ones_like(estimate, dtype=bool)
    valid[1:, 0] = False
    estimate[1:, 0] = -1
    series = ScanSeries(times, estimate, truth, valid, np.array([0., 1.1, 2.]),
                        np.zeros(3), 0.)
    # When: summarizing the mandated post-one-second population.
    result = summarize(series)
    # Then: invalid cells do not affect RMSE, but count against coverage.
    assert result.rmse_m == pytest.approx(.01)
    assert result.unknown_frac_mean == pytest.approx(1 / 231)
    assert result.blind_zone_fill == 1.


def test_blind_fill_when_travel_never_reaches_one_metre() -> None:
    from tron2_deploy.tools.perception_metrics import summarize, ScanSeries
    # Given: a stationary robot with a completely valid map.
    values = np.zeros((2, 231))
    series = ScanSeries(np.array([1.02, 2.02]), values, values,
                        np.ones_like(values, dtype=bool), np.zeros(2), np.zeros(2), 0.)
    # When: computing blind-zone fill.
    result = summarize(series)
    # Then: absence of the required population cannot pass a gate.
    assert result.blind_zone_fill is None


def test_rmse_when_ray_cast_misses() -> None:
    from tron2_deploy.tools.perception_metrics import summarize, ScanSeries
    # Given: one valid truth hit and 230 ray misses.
    estimate = np.zeros((1, 231))
    truth = np.full_like(estimate, np.nan)
    truth[0, 0] = .03
    series = ScanSeries(np.array([2.]), estimate, truth,
                        np.ones_like(estimate, dtype=bool), np.ones(1) * 2, np.zeros(1), 0.)
    # When: reducing errors.
    result = summarize(series)
    # Then: only the joint-valid population contributes.
    assert result.rmse_m == pytest.approx(.03)


def test_cli_when_recordings_are_missing(tmp_path: Path) -> None:
    # Given: an empty recording directory.
    script = Path(__file__).resolve().parents[1] / "tools/eval_perception.py"
    # When: running the actual public command.
    result = subprocess.run([sys.executable, str(script), "--rec", str(tmp_path),
                             "--json", str(tmp_path / "out.json")],
                            capture_output=True, text=True, check=False)
    # Then: no misleading calibration is produced, and failure is explicit.
    assert result.returncode != 0
    assert "RESULT: FAIL" in result.stdout
    assert not (tmp_path / "out.json").exists()


def test_recording_when_required_fields_are_missing(tmp_path: Path) -> None:
    from tron2_deploy.tools.perception_recording import load_recording, RecordingError
    # Given: a valid NPZ container with no recording schema.
    path = tmp_path / "bad.npz"
    np.savez(path, timestamp=np.arange(60) * .02)
    # When/Then: schema rejection names missing arrays instead of failing mid-replay.
    with pytest.raises(RecordingError, match="missing arrays"):
        load_recording(path)


def test_blind_fill_when_new_segment_has_not_traveled_one_metre() -> None:
    from tron2_deploy.tools.perception_metrics import summarize, ScanSeries
    # Given: a valid old segment followed by an empty new segment's map.
    values = np.zeros((4, 231))
    valid = np.ones_like(values, dtype=bool)
    valid[2:] = False
    series = ScanSeries(np.array([2., 3., 4., 5.]), values, values, valid,
                        np.array([1.1, 1.2, 0., .1]), np.zeros(4), 0.)
    # When: computing fill with reset-local travel.
    result = summarize(series)
    # Then: the new segment cannot inherit the old segment's eligibility.
    assert result.blind_zone_fill == 1.


def test_pose_interpolation_when_camera_and_policy_ticks_coincide() -> None:
    from tron2_deploy.tools.eval_perception import interpolate
    # Given: adjacent sensor poses with a one-metre translation.
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 1.
    # When/Then: a coincident camera tick uses the current pose without extra delay.
    np.testing.assert_allclose(interpolate(poses, 1.), poses[1])
