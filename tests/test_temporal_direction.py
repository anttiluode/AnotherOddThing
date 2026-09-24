import numpy as np

import another_odd_thing.trace as trace


def test_two_state_trace_separates_direction_when_level_is_matched():
    assert hasattr(trace, "matched_temporal_direction_pairs")

    pairs = trace.matched_temporal_direction_pairs()
    rising = pairs["rising"]
    falling = pairs["falling"]

    assert rising.shape == falling.shape
    assert rising.shape[0] >= 32
    assert np.max(np.abs(rising[:, 0] - falling[:, 0])) < 1e-12
    assert np.all(rising[:, 1] > 0.0)
    assert np.all(falling[:, 1] < 0.0)


def test_level_only_attacker_is_chance_on_exactly_matched_pairs():
    assert hasattr(trace, "temporal_direction_metrics")

    metrics = trace.temporal_direction_metrics()

    assert metrics["pair_count"] >= 32
    assert metrics["max_level_pair_gap"] < 1e-12
    assert metrics["two_state_accuracy"] >= 0.95
    assert np.isclose(metrics["level_only_accuracy"], 0.5)
    assert np.isclose(metrics["shuffled_contrast_accuracy"], 0.5)
