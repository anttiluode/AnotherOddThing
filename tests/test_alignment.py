import numpy as np

from another_odd_thing.alignment import (
    AlignmentConfig,
    heterosynaptic_normalize,
    train_alignment,
)


def test_heterosynaptic_normalization_moves_inactive_synapses_and_preserves_budget():
    weights = np.ones((4, 4), dtype=float)
    weights[0, 0] += 0.40
    normalized = heterosynaptic_normalize(weights, per_compartment_budget=4.0)

    assert np.isclose(normalized[:, 0].sum(), 4.0)
    assert np.all(normalized[1:, 0] < 1.0)
    assert np.allclose(normalized[:, 1:].sum(axis=0), 4.0)


def test_mirrored_independent_trace_matches_coupled_machine():
    cfg = AlignmentConfig(presentations_per_context=24)
    coupled = train_alignment(seed=7, mode="coupled", mismatch_probability=0.0, config=cfg)
    mirrored = train_alignment(seed=7, mode="decoupled_mirrored", mismatch_probability=0.0, config=cfg)

    assert np.allclose(coupled["weights"], mirrored["weights"])
    assert coupled["alignment_accuracy"] == mirrored["alignment_accuracy"]


def test_oracle_routed_decoupled_matches_when_mapping_is_aligned():
    cfg = AlignmentConfig(presentations_per_context=24)
    coupled = train_alignment(seed=11, mode="coupled", mismatch_probability=0.0, config=cfg)
    decoupled = train_alignment(seed=11, mode="decoupled_routed", mismatch_probability=0.0, config=cfg)

    assert np.allclose(coupled["weights"], decoupled["weights"])
    assert decoupled["mapping_mismatch_fraction"] == 0.0


def test_fully_misaligned_independent_routing_breaks_weight_alignment():
    cfg = AlignmentConfig(presentations_per_context=24)
    coupled_scores = []
    decoupled_scores = []
    for seed in range(24):
        coupled = train_alignment(seed=seed, mode="coupled", mismatch_probability=1.0, config=cfg)
        decoupled = train_alignment(
            seed=seed,
            mode="decoupled_routed",
            mismatch_probability=1.0,
            config=cfg,
        )
        coupled_scores.append(coupled["alignment_accuracy"])
        decoupled_scores.append(decoupled["alignment_accuracy"])

    assert np.mean(coupled_scores) > 0.95
    assert np.mean(decoupled_scores) < 0.25
