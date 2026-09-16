import numpy as np

from another_odd_thing.core import (
    event_bit,
    passive_trace,
    probe_once,
    v0_operators,
)


def test_passive_observation_is_exactly_identical():
    op0, op1 = v0_operators()
    trace0 = passive_trace(op0, steps=4)
    trace1 = passive_trace(op1, steps=4)
    np.testing.assert_array_equal(trace0, trace1)


def test_visible_site_probe_does_not_reveal_operator():
    op0, op1 = v0_operators()
    y0 = probe_once(op0, passive_steps=4, site=0, amplitude=1.0)
    y1 = probe_once(op1, passive_steps=4, site=0, amplitude=1.0)
    assert y0 == y1


def test_hidden_site_probe_reveals_operator():
    op0, op1 = v0_operators()
    y0 = probe_once(op0, passive_steps=4, site=1, amplitude=1.0)
    y1 = probe_once(op1, passive_steps=4, site=1, amplitude=1.0)
    assert np.isclose(y1 - y0, 0.65)


def test_one_bit_publication_separates_hidden_probe_responses():
    op0, op1 = v0_operators()
    y0 = probe_once(op0, passive_steps=4, site=1, amplitude=1.0)
    y1 = probe_once(op1, passive_steps=4, site=1, amplitude=1.0)
    assert event_bit(y0, threshold=0.75) == 0
    assert event_bit(y1, threshold=0.75) == 1
