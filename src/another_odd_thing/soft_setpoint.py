from __future__ import annotations

import numpy as np

from .alignment import AlignmentConfig, alignment_metrics, heterosynaptic_normalize
from .trace import CompartmentTrace, TraceConfig, plasticity_sign


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


def continuous_plasticity_update(
    level: np.ndarray | float,
    *,
    config: TraceConfig,
) -> np.ndarray:
    """Continuous-magnitude attacker for the quantized 0/LTD/LTP readout.

    The control preserves the quantized rule's exact no-update floor below
    ``theta_d``. Between ``theta_d`` and ``theta_p`` it interpolates linearly
    from -1 through 0 to +1, then saturates at +1 above ``theta_p``. Thus the
    attacker smooths the LTD-to-LTP magnitude transition without gaining an
    extra subthreshold depression mechanism.
    """

    values = np.asarray(level, dtype=float)
    td = config.theta_d
    tp = config.theta_p

    between = -1.0 + 2.0 * (values - td) / (tp - td)
    return np.where(
        values < td,
        0.0,
        np.where(values < tp, between, 1.0),
    )


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
    plasticity_rule: str = "quantized",
    config: AlignmentConfig | None = None,
) -> dict[str, object]:
    """Train with correct expression but continuously displaced plasticity relief."""

    cfg = config or AlignmentConfig()
    if not 0.0 <= mismatch <= 1.0:
        raise ValueError("mismatch must be between 0 and 1")
    if plasticity_rule not in {"quantized", "continuous"}:
        raise ValueError("plasticity_rule must be 'quantized' or 'continuous'")

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
    positive_updates = 0
    negative_updates = 0
    zero_updates = 0

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

        if plasticity_rule == "quantized":
            updates = plasticity_sign(plasticity_trace.level, config=cfg.trace).astype(float)
        else:
            updates = continuous_plasticity_update(
                plasticity_trace.level,
                config=cfg.trace,
            )

        positive_updates += int(np.sum(updates > 0.0))
        negative_updates += int(np.sum(updates < 0.0))
        zero_updates += int(np.sum(np.isclose(updates, 0.0, atol=1e-12)))

        weights[pathway] += cfg.learning_rate * updates
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
        "plasticity_rule": plasticity_rule,
        "alignment_accuracy": metrics["alignment_accuracy"],
        "diagonal_weight_share": metrics["diagonal_weight_share"],
        "publication_target_fraction": float(publication_target_hits / len(schedule)),
        "weight_budget_max_abs_error": budget_error,
        "positive_update_events": int(positive_updates),
        "negative_update_events": int(negative_updates),
        "zero_update_events": int(zero_updates),
        "wrong_compartments": wrong,
        "weights": weights,
    }
