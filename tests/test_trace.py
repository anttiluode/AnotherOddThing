import numpy as np

from another_odd_thing.trace import CompartmentTrace, TraceConfig, plasticity_sign


def test_shared_trace_contrast_changes_sign_after_pulse():
    cfg = TraceConfig()
    trace = CompartmentTrace.zeros(1, config=cfg)

    for _ in range(6):
        trace.step(np.array([1.0]))
    assert trace.contrast[0] > 0.0

    for _ in range(6):
        trace.step(np.array([0.0]))
    assert trace.contrast[0] < 0.0


def test_plasticity_sign_has_three_regimes():
    cfg = TraceConfig(theta_d=0.10, theta_p=0.32)
    signs = plasticity_sign(np.array([0.05, 0.20, 0.50]), config=cfg)
    assert signs.tolist() == [0, -1, 1]


def test_trace_state_is_compartment_indexed():
    cfg = TraceConfig()
    trace = CompartmentTrace.zeros(3, config=cfg)
    trace.step(np.array([1.0, 0.0, 0.0]))
    assert trace.fast[0] > trace.fast[1] == trace.fast[2]
    assert trace.slow[0] > trace.slow[1] == trace.slow[2]
