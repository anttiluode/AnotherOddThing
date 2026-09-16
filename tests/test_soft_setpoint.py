import numpy as np

from another_odd_thing.alignment import AlignmentConfig
from another_odd_thing.soft_setpoint import soft_inhibition, train_soft_misalignment


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
