from __future__ import annotations

from dataclasses import dataclass
from math import erf, log2, sqrt
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class V1Config:
    low_gain: float = 0.20
    high_gain: float = 1.40
    threshold: float = 0.80
    sigma: float = 0.35
    probe_budget: int = 2


def operator_gain_table(*, low: float, high: float) -> np.ndarray:
    """Return the fixed four-operator, three-address v1 transfer codebook."""

    signatures = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    )
    return np.where(signatures, float(high), float(low))


def event_probability(*, mean: float, threshold: float, sigma: float) -> float:
    """P(mean + Normal(0, sigma) > threshold)."""

    if sigma <= 0:
        return float(mean > threshold)
    z = (float(mean) - float(threshold)) / (float(sigma) * sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


def entropy_bits(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    positive = probs[probs > 0.0]
    return float(-sum(p * log2(p) for p in positive))


def _site_event_probabilities(gains: np.ndarray, site: int, config: V1Config) -> np.ndarray:
    return np.asarray(
        [
            event_probability(mean=float(mean), threshold=config.threshold, sigma=config.sigma)
            for mean in gains[:, site]
        ],
        dtype=float,
    )


def posterior_update(
    prior: np.ndarray,
    gains: np.ndarray,
    *,
    site: int,
    event: int,
    config: V1Config,
) -> np.ndarray:
    prior = np.asarray(prior, dtype=float)
    if prior.ndim != 1 or prior.shape[0] != gains.shape[0]:
        raise ValueError("prior length must match operator count")
    if event not in (0, 1):
        raise ValueError("event must be 0 or 1")
    p_on = _site_event_probabilities(gains, site, config)
    likelihood = p_on if event == 1 else (1.0 - p_on)
    unnormalized = prior * likelihood
    mass = float(unnormalized.sum())
    if mass <= 0.0:
        raise ValueError("observation has zero probability under every hypothesis")
    return unnormalized / mass


def expected_information_gain(
    prior: np.ndarray,
    gains: np.ndarray,
    *,
    site: int,
    config: V1Config,
) -> float:
    prior = np.asarray(prior, dtype=float)
    p_on_by_operator = _site_event_probabilities(gains, site, config)
    p_on = float(prior @ p_on_by_operator)
    p_off = 1.0 - p_on
    expected_posterior_entropy = 0.0
    if p_on > 0.0:
        expected_posterior_entropy += p_on * entropy_bits(
            posterior_update(prior, gains, site=site, event=1, config=config)
        )
    if p_off > 0.0:
        expected_posterior_entropy += p_off * entropy_bits(
            posterior_update(prior, gains, site=site, event=0, config=config)
        )
    return entropy_bits(prior) - expected_posterior_entropy


def choose_active_site(
    prior: np.ndarray,
    gains: np.ndarray,
    *,
    available_sites: Iterable[int],
    config: V1Config,
) -> int:
    sites = tuple(int(site) for site in available_sites)
    if not sites:
        raise ValueError("at least one site must be available")
    scored = [
        (expected_information_gain(prior, gains, site=site, config=config), -site, site)
        for site in sites
    ]
    return max(scored)[2]


def _observe_event(
    *,
    true_operator: int,
    site: int,
    gains: np.ndarray,
    noise_z: float,
    config: V1Config,
) -> int:
    analog = float(gains[true_operator, site]) + config.sigma * float(noise_z)
    return int(analog > config.threshold)


def _run_policy(
    *,
    true_operator: int,
    gains: np.ndarray,
    noise_tape: np.ndarray,
    config: V1Config,
    random_sites: tuple[int, ...] | None,
) -> tuple[bool, float, tuple[int, ...], tuple[int, ...]]:
    prior = np.full(gains.shape[0], 1.0 / gains.shape[0], dtype=float)
    available = list(range(gains.shape[1]))
    sites_used: list[int] = []
    events: list[int] = []

    for round_index in range(config.probe_budget):
        if random_sites is None:
            site = choose_active_site(prior, gains, available_sites=available, config=config)
        else:
            site = int(random_sites[round_index])
        available.remove(site)
        event = _observe_event(
            true_operator=true_operator,
            site=site,
            gains=gains,
            noise_z=float(noise_tape[round_index]),
            config=config,
        )
        prior = posterior_update(prior, gains, site=site, event=event, config=config)
        sites_used.append(site)
        events.append(event)

    prediction = int(np.argmax(prior))
    return prediction == true_operator, entropy_bits(prior), tuple(sites_used), tuple(events)


def run_paired_trials(*, trials: int, seed: int, config: V1Config) -> dict[str, object]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 1 <= config.probe_budget <= 3:
        raise ValueError("probe_budget must be between 1 and 3")

    gains = operator_gain_table(low=config.low_gain, high=config.high_gain)
    rng = np.random.default_rng(seed)
    true_operators = rng.integers(0, gains.shape[0], size=trials)
    noise_tapes = rng.standard_normal((trials, config.probe_budget))
    random_site_orders = np.vstack([rng.permutation(gains.shape[1]) for _ in range(trials)])

    active_correct: list[int] = []
    random_correct: list[int] = []
    active_entropies: list[float] = []
    random_entropies: list[float] = []
    active_first_sites = np.zeros(gains.shape[1], dtype=int)
    random_first_sites = np.zeros(gains.shape[1], dtype=int)
    active_site_paths: dict[str, int] = {}
    random_site_paths: dict[str, int] = {}

    for trial in range(trials):
        true_operator = int(true_operators[trial])
        noise = noise_tapes[trial]
        active = _run_policy(
            true_operator=true_operator,
            gains=gains,
            noise_tape=noise,
            config=config,
            random_sites=None,
        )
        random_sites = tuple(int(x) for x in random_site_orders[trial, : config.probe_budget])
        random = _run_policy(
            true_operator=true_operator,
            gains=gains,
            noise_tape=noise,
            config=config,
            random_sites=random_sites,
        )

        active_correct.append(int(active[0]))
        random_correct.append(int(random[0]))
        active_entropies.append(float(active[1]))
        random_entropies.append(float(random[1]))
        active_first_sites[active[2][0]] += 1
        random_first_sites[random[2][0]] += 1
        active_key = "->".join(str(site) for site in active[2])
        random_key = "->".join(str(site) for site in random[2])
        active_site_paths[active_key] = active_site_paths.get(active_key, 0) + 1
        random_site_paths[random_key] = random_site_paths.get(random_key, 0) + 1

    active_arr = np.asarray(active_correct, dtype=int)
    random_arr = np.asarray(random_correct, dtype=int)
    paired = active_arr - random_arr

    return {
        "trials": int(trials),
        "seed": int(seed),
        "active_accuracy": float(active_arr.mean()),
        "random_accuracy": float(random_arr.mean()),
        "paired_accuracy_delta": float(paired.mean()),
        "active_entropy_mean": float(np.mean(active_entropies)),
        "random_entropy_mean": float(np.mean(random_entropies)),
        "active_probe_count_mean": float(config.probe_budget),
        "random_probe_count_mean": float(config.probe_budget),
        "active_first_site_counts": active_first_sites.tolist(),
        "random_first_site_counts": random_first_sites.tolist(),
        "active_site_paths": dict(sorted(active_site_paths.items())),
        "random_site_paths": dict(sorted(random_site_paths.items())),
        "active_only_correct": int(np.sum((active_arr == 1) & (random_arr == 0))),
        "random_only_correct": int(np.sum((active_arr == 0) & (random_arr == 1))),
        "paired_accuracy_differences": paired.tolist(),
    }
