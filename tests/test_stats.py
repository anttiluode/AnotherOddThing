import numpy as np

from another_odd_thing.stats import paired_bootstrap_mean_ci


def test_paired_bootstrap_is_deterministic_and_contains_observed_mean():
    differences = np.array([1.0, 1.0, 0.0, 0.0])
    first = paired_bootstrap_mean_ci(
        differences,
        seed=123,
        resamples=2000,
        confidence=0.95,
    )
    second = paired_bootstrap_mean_ci(
        differences,
        seed=123,
        resamples=2000,
        confidence=0.95,
    )
    assert first == second
    assert first[0] <= differences.mean() <= first[1]


def test_paired_bootstrap_rejects_empty_input():
    try:
        paired_bootstrap_mean_ci(np.array([]), seed=1, resamples=10, confidence=0.95)
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("expected ValueError")
