from __future__ import annotations

import numpy as np

from .alignment import AlignmentConfig, alignment_metrics, heterosynaptic_normalize
from .trace import CompartmentTrace, plasticity_sign


def soft_inhibition(
    *,
    correct: int,
    wrong: int,
    mismatch: float,
    config: AlignmentConfig,
) -> np.ndarray:
    """Continuously transfer inhibitory relief from one compartment to another.

    ``mismatch=0`` fully disinhibits the correct compartment, ``mismatch=1``
    fully disinhibits the wrong compartment, and intermediate values split the
    same relief budget continuously between them. No binary route switch is
    used.
    """

    if not 0.0 <= mismatch <= 1.0:
        raise ValueError("mismatch must be between 0 and 1")
    n = config.n_compartments
    if not 0 <= correct < n or not 0 <= wrong < n:
        raise ValueError("compartment index out of range")
    if correct == wrong:
        raise ValueError("correct and wrong compartments must differ")

    inhibition = np.full(n, config.closed_inhibition, dtype=float)
    relief = config.closed_inhibition - config.open_inhibition
    inhibition[correct] = config.closed_inhibition - (1.0 - mismatch) * relief
    inhibition[wrong] = config.closed_inhibition - mismatch * relief
    return inhibition


def _choose_wrong_compartments(*, seed: int, n_compartments: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wrong = np.empty(n_compartments, dtype=int)
    all_compartments = np.arange(n_compartments)
    for context in range(n_compartments):
        wrong[context] = int(rng.choice(np.delete(all_compartments, context)))
    return wrong


def _drive(weights_for_pathway: np.ndarray, inhibition: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(weights_for_pathway, dtype=float) - inhibition, 0.0)


def train_soft_misalignment(
    *,
    seed: int,
    mismatch: float,
    config: AlignmentConfig | None = None,
) -> dict[str, object]:
    """Train with correct expression but continuously displaced plasticity relief."""

    cfg = config or AlignmentConfig()
    if not 0.0 <= mismatch <= 1.0:
        raise ValueError("mismatch must be between 0 and 1")

    n = cfg.n_compartments
    rng_weights = np.random.default_rng(seed)
    rng_schedule = np.random.default_rng(seed + 1_000_003)
    wrong = _choose_wrong_compartments(seed=seed + 2_000_003, n_compartments=n)

    weights = np.full((n, n), cfg.initial_weight, dtype=float)
    weights += rng_weights.normal(0.0, cfg.initial_weight_noise, size=(n, n))
    weights = np.maximum(weights, cfg.min_weight)
    weights = heterosynaptic_normalize(
        weights,
        per_compartment_budget=cfg.per_compartment_budget,
    )

    expression_trace = CompartmentTrace.zeros(n, config=cfg.trace)
    plasticity_trace = CompartmentTrace.zeros(n, config=cfg.trace)

    schedule = np.tile(np.arange(n, dtype=int), cfg.presentations_per_context)
    rng_schedule.shuffle(schedule)

    publication_target_hits = 0
    ltp_events = 0
    ltd_events = 0
    no_change_events = 0

    for context in schedule:
        pathway = int(context)
        wrong_compartment = int(wrong[pathway])

        expression_inhibition = soft_inhibition(
            correct=pathway,
            wrong=wrong_compartment,
            mismatch=0.0,
            config=cfg,
        )
        plasticity_inhibition = soft_inhibition(
            correct=pathway,
            wrong=wrong_compartment,
            mismatch=mismatch,
            config=cfg,
        )
        expression_drive = _drive(weights[pathway], expression_inhibition)
        plasticity_drive = _drive(weights[pathway], plasticity_inhibition)

        for _ in range(cfg.active_steps):
            expression_trace.step(expression_drive)
            plasticity_trace.step(plasticity_drive)

        if int(np.argmax(expression_trace.level)) == pathway:
            publication_target_hits += 1

        signs = plasticity_sign(plasticity_trace.level, config=cfg.trace)
        ltp_events += int(np.sum(signs > 0))
        ltd_events += int(np.sum(signs < 0))
        no_change_events += int(np.sum(signs == 0))

        weights[pathway] += cfg.learning_rate * signs
        weights = np.maximum(weights, cfg.min_weight)
        weights = heterosynaptic_normalize(
            weights,
            per_compartment_budget=cfg.per_compartment_budget,
        )

        zero_drive = np.zeros(n, dtype=float)
        for _ in range(cfg.rest_steps):
            expression_trace.step(zero_drive)
            plasticity_trace.step(zero_drive)

    metrics = alignment_metrics(weights)
    budget_error = float(
        np.max(np.abs(weights.sum(axis=0) - cfg.per_compartment_budget))
    )

    return {
        "seed": int(seed),
        "mismatch": float(mismatch),
        "alignment_accuracy": metrics["alignment_accuracy"],
        "diagonal_weight_share": metrics["diagonal_weight_share"],
        "publication_target_fraction": float(publication_target_hits / len(schedule)),
        "weight_budget_max_abs_error": budget_error,
        "ltp_events": int(ltp_events),
        "ltd_events": int(ltd_events),
        "no_change_events": int(no_change_events),
        "wrong_compartments": wrong,
        "weights": weights,
    }
