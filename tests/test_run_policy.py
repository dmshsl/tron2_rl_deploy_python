import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "tron2_deploy/run_policy.py"


@pytest.mark.parametrize("policy,dimension", [("base", 273), ("base_blind", 42)])
def test_cli_when_dry_run_validates_models_without_sdk(policy, dimension):
    # Given deployed models, when invoking the real CLI, then validate without SDK.
    result = subprocess.run([sys.executable, str(ENTRY), "--policy", policy, "--dry-run"],
                            text=True, capture_output=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr
    assert f"DRY_RUN PASS actor={dimension} encoder=360; SDK not imported" in result.stdout
    assert "calibration: UNVERIFIED" in result.stdout


@pytest.mark.parametrize("arguments,reason", [
    (["--robot_ip", "192.0.2.1"], "forbids real-robot"),
    (["--timeout", "nan"], "positive and finite"),
    (["--imu_offset_rpy", "nan", "0", "0"], "finite"),
])
def test_cli_when_invalid_rejects_before_sdk(arguments, reason):
    # Given invalid inputs, when invoking the real CLI, then exit without connecting.
    result = subprocess.run([sys.executable, str(ENTRY), "--policy", "base_blind", *arguments],
                            text=True, capture_output=True, timeout=20, check=False)
    assert result.returncode != 0
    assert reason in result.stderr
    assert "No module named 'limxsdk'" not in result.stderr


def test_cli_when_schedule_outside_training_range_rejects(tmp_path):
    # Given finite but unsafe commanded velocity in a schedule.
    path = tmp_path / "schedule.yaml"
    path.write_text("- [0, 10000, 0, 0]\n")
    # When invoking the real CLI, then reject rather than silently scale or clip.
    result = subprocess.run([sys.executable, str(ENTRY), "--policy", "base_blind", "--schedule", str(path)],
                            text=True, capture_output=True, timeout=20, check=False)
    assert result.returncode == 2
    assert "outside the generated training command ranges" in result.stderr
