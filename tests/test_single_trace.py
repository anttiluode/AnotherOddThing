import numpy as np

from another_odd_thing.alignment import AlignmentConfig
from another_odd_thing.soft_setpoint import train_soft_misalignment
from another_odd_thing.trace import (
    CompartmentTrace,
    SingleCompartmentTrace,
    matched_single_tau_s,
)


def test_matched_single_trace_hits_same_level_at_plasticity_readout():
    cfg = AlignmentConfig()
    two = CompartmentTrace.zeros(1, config=cfg.trace)
    tau_s = matched_single_tau_s(config=cfg.trace, active_steps=cfg.active_steps)
    one = SingleCompartmentTrace.zeros(1, tau_s=tau_s, config=cfg.trace)

    drive = np.ones(1)
    for _ in range(cfg.active_steps):
        two.step(drive)
        one.step(drive)

    assert np.allclose(one.level, two.level, atol=1e-12, rtol=0.0)
    assert tau_s > cfg.trace.tau_slow_s


def test_matched_single_trace_is_a_real_attacker_not_an_exact_reparameterization():
    cfg = AlignmentConfig()
    two = CompartmentTrace.zeros(1, config=cfg.trace)
    tau_s = matched_single_tau_s(config=cfg.trace, active_steps=cfg.active_steps)
    one = SingleCompartmentTrace.zeros(1, tau_s=tau_s, config=cfg.trace)

    for _ in range(cfg.active_steps):
        two.step(np.ones(1))
        one.step(np.ones(1))
    matched_level = float(two.level[0])

    for _ in range(cfg.rest_steps):
        two.step(np.zeros(1))
        one.step(np.zeros(1))

    assert np.isclose(matched_level, 0.3715702527028714)
    assert not np.allclose(one.level, two.level, atol=1e-4, rtol=0.0)


def test_soft_setpoint_can_swap_only_the_plasticity_trace_dynamics():
    cfg = AlignmentConfig(presentations_per_context=12)
    two = train_soft_misalignment(
        seed=7,
        mismatch=0.5,
        plasticity_rule="quantized",
        plasticity_trace_mode="two_timescale",
        config=cfg,
    )
    one = train_soft_misalignment(
        seed=7,
        mismatch=0.5,
        plasticity_rule="quantized",
        plasticity_trace_mode="single_matched",
        config=cfg,
    )

    assert two["plasticity_trace_mode"] == "two_timescale"
    assert one["plasticity_trace_mode"] == "single_matched"
    assert one["matched_single_tau_s"] > cfg.trace.tau_slow_s
    assert two["wrong_compartments"].tolist() == one["wrong_compartments"].tolist()
    assert two["weight_budget_max_abs_error"] < 1e-12
    assert one["weight_budget_max_abs_error"] < 1e-12
