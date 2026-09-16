import numpy as np

from another_odd_thing.alignment import AlignmentConfig
from another_odd_thing.soft_setpoint import (
    continuous_plasticity_update,
    soft_inhibition,
    train_soft_misalignment,
)


def test_soft_inhibition_interpolates_relief_without_binary_routing():
    cfg = AlignmentConfig()

    q0 = soft_inhibition(correct=0, wrong=1, mismatch=0.0, config=cfg)
    qhalf = soft_inhibition(correct=0, wrong=1, mismatch=0.5, config=cfg)
    q1 = soft_inhibition(correct=0, wrong=1, mismatch=1.0, config=cfg)

    assert np.allclose(q0, [cfg.open_inhibition, cfg.closed_inhibition, cfg.closed_inhibition, cfg.closed_inhibition])
    assert np.isclose(qhalf[0], qhalf[1])
    assert cfg.open_inhibition < qhalf[0] < cfg.closed_inhibition
    assert np.allclose(q1, [cfg.closed_inhibition, cfg.open_inhibition, cfg.closed_inhibition, cfg.closed_inhibition])


def test_soft_misalignment_has_robust_region_then_breaks_alignment():
    cfg = AlignmentConfig(presentations_per_context=24)
    low = [train_soft_misalignment(seed=s, mismatch=0.2, config=cfg)["alignment_accuracy"] for s in range(24)]
    high = [train_soft_misalignment(seed=s, mismatch=0.7, config=cfg)["alignment_accuracy"] for s in range(24)]

    assert np.mean(low) > 0.95
    assert np.mean(high) < 0.10


def test_continuous_plasticity_update_preserves_no_update_floor_and_interpolates_between_thresholds():
    cfg = AlignmentConfig()
    td = cfg.trace.theta_d
    tp = cfg.trace.theta_p
    midpoint = 0.5 * (td + tp)

    updates = continuous_plasticity_update(
        np.array([0.0, 0.5 * td, td, midpoint, tp, 1.5 * tp]),
        config=cfg.trace,
    )

    assert np.allclose(updates, [0.0, 0.0, -1.0, 0.0, 1.0, 1.0])
    interior = continuous_plasticity_update(
        np.linspace(td, tp, 9),
        config=cfg.trace,
    )
    assert np.unique(np.round(interior, 8)).size > 3


def test_continuous_magnitude_attacker_softens_middle_without_changing_endpoints():
    cfg = AlignmentConfig(presentations_per_context=24)

    def mean_accuracy(mismatch: float, rule: str) -> float:
        return float(
            np.mean(
                [
                    train_soft_misalignment(
                        seed=s,
                        mismatch=mismatch,
                        plasticity_rule=rule,
                        config=cfg,
                    )["alignment_accuracy"]
                    for s in range(24)
                ]
            )
        )

    q0_quantized = mean_accuracy(0.0, "quantized")
    q0_continuous = mean_accuracy(0.0, "continuous")
    qhalf_quantized = mean_accuracy(0.5, "quantized")
    qhalf_continuous = mean_accuracy(0.5, "continuous")
    q1_quantized = mean_accuracy(1.0, "quantized")
    q1_continuous = mean_accuracy(1.0, "continuous")

    assert q0_quantized > 0.95
    assert q0_continuous > 0.95
    assert q1_quantized < 0.10
    assert q1_continuous < 0.10
    assert qhalf_continuous > qhalf_quantized + 0.30


def test_middle_difference_is_visible_before_argmax_in_weight_share():
    cfg = AlignmentConfig(presentations_per_context=24)
    quantized = [
        train_soft_misalignment(
            seed=s,
            mismatch=0.5,
            plasticity_rule="quantized",
            config=cfg,
        )["diagonal_weight_share"]
        for s in range(24)
    ]
    continuous = [
        train_soft_misalignment(
            seed=s,
            mismatch=0.5,
            plasticity_rule="continuous",
            config=cfg,
        )["diagonal_weight_share"]
        for s in range(24)
    ]

    assert np.mean(continuous) > np.mean(quantized) + 0.10
