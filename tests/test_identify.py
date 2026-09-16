import numpy as np

from another_odd_thing.identify import (
    V1Config,
    choose_active_site,
    event_probability,
    operator_gain_table,
    posterior_update,
    run_paired_trials,
)


def test_gain_table_encodes_predeclared_operator_signatures():
    gains = operator_gain_table(low=0.2, high=1.4)
    expected_high = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(gains > 0.8, expected_high)


def test_event_likelihood_respects_low_and_high_gains():
    p_low = event_probability(mean=0.2, threshold=0.8, sigma=0.35)
    p_high = event_probability(mean=1.4, threshold=0.8, sigma=0.35)
    assert p_low < 0.05
    assert p_high > 0.95


def test_posterior_update_normalizes():
    config = V1Config()
    gains = operator_gain_table(low=config.low_gain, high=config.high_gain)
    prior = np.full(4, 0.25)
    posterior = posterior_update(prior, gains, site=1, event=1, config=config)
    assert np.isclose(posterior.sum(), 1.0)
    assert np.all(posterior >= 0.0)


def test_active_policy_chooses_balanced_probe_first():
    config = V1Config()
    gains = operator_gain_table(low=config.low_gain, high=config.high_gain)
    prior = np.full(4, 0.25)
    assert choose_active_site(prior, gains, available_sites=(0, 1, 2), config=config) == 1


def test_active_policy_adapts_second_probe_to_first_event():
    config = V1Config()
    gains = operator_gain_table(low=config.low_gain, high=config.high_gain)
    prior = np.full(4, 0.25)

    after_off = posterior_update(prior, gains, site=1, event=0, config=config)
    assert choose_active_site(after_off, gains, available_sites=(0, 2), config=config) == 0

    after_on = posterior_update(prior, gains, site=1, event=1, config=config)
    assert choose_active_site(after_on, gains, available_sites=(0, 2), config=config) == 2


def test_paired_trials_hold_probe_budget_fixed_and_active_wins_this_family():
    result = run_paired_trials(trials=1024, seed=20260916, config=V1Config())
    assert result["active_probe_count_mean"] == 2.0
    assert result["random_probe_count_mean"] == 2.0
    assert result["active_accuracy"] > result["random_accuracy"] + 0.10
    assert result["active_entropy_mean"] < result["random_entropy_mean"]
    assert len(result["paired_accuracy_differences"]) == 1024
