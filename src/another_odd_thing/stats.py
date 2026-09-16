from __future__ import annotations

import numpy as np


def paired_bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    seed: int,
    resamples: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("differences must be a non-empty 1-D array")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha], method="linear")
    return float(low), float(high)
