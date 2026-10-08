"""Task-15 population: all 231 cells, t>1 s; joint-valid RMSE, no imputation.

unknown_frac_by_age_s[k] is mean unknown fraction at recording ages [k,k+1),
intersected with t>1 s (empty bins are null). Blind fill uses t>1 s AND path>1 m.
Latency uses post-warmup capture-consumption events, in 20 ms steps, not CPU time.
"""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tron2_deploy.perception.height_map import policy_offsets


@dataclass(frozen=True, slots=True)
class ScanSeries:
    timestamp: NDArray[np.float64]
    estimate: NDArray[np.float64]
    truth: NDArray[np.float64]
    valid: NDArray[np.bool_]
    distance: NDArray[np.float64]
    latency: NDArray[np.float64]
    drift: float


@dataclass(frozen=True, slots=True)
class TerrainMetrics:
    rmse_m: float | None
    unknown_frac_mean: float
    unknown_frac_by_age_s: list[float | None]
    blind_zone_fill: float | None
    latency_steps_mean: float | None
    odom_drift_frac: float


def summarize(series: ScanSeries) -> TerrainMetrics:
    """Keep missing populations explicit so they cannot pass numerical gates."""
    population = series.timestamp > 1.
    overlap = series.valid & np.isfinite(series.truth) & population[:, None]
    error = (series.estimate - series.truth)[overlap]
    unknown = 1. - series.valid.mean(axis=1)
    blind = np.abs(policy_offsets()[:, 0]) < .4
    moving = population & (series.distance > 1.)
    bins = []
    for second in range(int(np.ceil(series.timestamp[-1]))):
        upper = series.timestamp <= second + 1 if second == int(np.ceil(series.timestamp[-1])) - 1 else series.timestamp < second + 1
        keep = population & (series.timestamp >= second) & upper
        bins.append(float(unknown[keep].mean()) if keep.any() else None)
    return TerrainMetrics(
        float(np.sqrt(np.mean(error**2))) if error.size else None,
        float(unknown[population].mean()), bins,
        float(series.valid[moving][:, blind].mean()) if moving.any() else None,
        float(series.latency.mean()) if series.latency.size else None,
        series.drift)
