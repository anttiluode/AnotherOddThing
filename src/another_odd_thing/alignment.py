from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .trace import CompartmentTrace, TraceConfig, plasticity_sign


@dataclass(frozen=True)
class AlignmentConfig:
    """Configuration for the v2 expression/plasticity alignment world."""

    n_compartments: int = 4
    initial_weight: float = 0.90
    initial_weight_noise: float = 0.01
    open_inhibition: float = 0.10
    closed_inhibition: float = 0.70
    learning_rate: float = 0.04
    active_steps: int = 6
    rest_steps: int = 6
    presentations_per_context: int = 30
    min_weight: float = 0.05
    trace: TraceConfig = field(default_factory=TraceConfig)

    def __post_init__(self) -> None:
        if self.n_compartments < 2:
            raise ValueError("n_compartments must be at least 2")
        if self.initial_weight <= 0.0:
            raise ValueError("initial_weight must be positive")
        if self.initial_weight_noise < 0.0:
            raise ValueError("initial_weight_noise must be non-negative")
        if not self.open_inhibition < self.closed_inhibition:
            raise ValueError("open inhibition must be lower than closed inhibition")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.active_steps <= 0 or self.rest_steps < 0:
            raise ValueError("active_steps must be positive and rest_steps non-negative")
        if self.presentations_per_context <= 0:
            raise ValueError("presentations_per_context must be positive")

    @property
    def per_compartment_budget(self) -> float:
        return float(self.n_compartments * self.initial_weight)


def heterosynaptic_normalize(weights: np.ndarray, *, per_compartment_budget: float) -> np.ndarray:
    """Restore a fixed total excitatory budget independently on each compartment.

    Scaling the whole column means an update to one synapse necessarily changes
    inactive neighbours on the same compartment. That is the intentionally local,
    shared-bus part of v2.
    """

    values = np.asarray(weights, dtype=float).copy()
    if values.ndim != 2:
        raise ValueError("weights must be a 2D pathway-by-compartment matrix")
    if per_compartment_budget <= 0.0:
        raise ValueError("per_compartment_budget must be positive")
    sums = values.sum(axis=0)
    if np.any(sums <= 0.0):
        raise ValueError("each compartment must have positive incoming weight")
    values *= (per_compartment_budget / sums)[None, :]
    return values


def make_plasticity_map(
    *,
    seed: int,
    n_compartments: int,
    mismatch_probability: float,
) -> np.ndarray:
    """Create an independently routed context->plasticity-compartment map.

    A mismatched context is always redirected to a *wrong* compartment, so the
    realized mismatch fraction is directly measurable rather than hidden inside
    chance self-matches.
    """

    if not 0.0 <= mismatch_probability <= 1.0:
        raise ValueError("mismatch_probability must be between 0 and 1")
    rng = np.random.default_rng(seed)
    mapping = np.arange(n_compartments, dtype=int)
    for context in range(n_compartments):
        if rng.random() < mismatch_probability:
            choices = np.delete(np.arange(n_compartments), context)
            mapping[context] = int(rng.choice(choices))
    return mapping


def _inhibition_vector(*, target: int, config: AlignmentConfig) -> np.ndarray:
    inhibition = np.full(config.n_compartments, config.closed_inhibition, dtype=float)
    inhibition[target] = config.open_inhibition
    return inhibition


def _local_drive(weights_for_pathway: np.ndarray, *, target: int, config: AlignmentConfig) -> np.ndarray:
    """Soft offset gate: inhibition shifts local drive instead of multiplying by a mask."""

    inhibition = _inhibition_vector(target=target, config=config)
    return np.maximum(np.asarray(weights_for_pathway, dtype=float) - inhibition, 0.0)


def alignment_metrics(weights: np.ndarray) -> dict[str, float]:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("alignment metric expects a square pathway-by-compartment matrix")
    targets = np.arange(values.shape[0])
    strongest = np.argmax(values, axis=1)
    row_sums = values.sum(axis=1)
    return {
        "alignment_accuracy": float(np.mean(strongest == targets)),
        "diagonal_weight_share": float(np.mean(np.diag(values) / row_sums)),
    }


def train_alignment(
    *,
    seed: int,
    mode: str,
    mismatch_probability: float,
    config: AlignmentConfig | None = None,
) -> dict[str, object]:
    """Train the four-context toy world under one of three coordination models.

    Modes:
      - coupled: expression and plasticity literally read the same trace.
      - decoupled_mirrored: separate traces receive the same local signal.
      - decoupled_routed: plasticity has an independent context->compartment map.

    The mirrored attacker is important: if two separate state variables receive
    identical local information, they are expected to match the coupled system.
    The experiment tests the routing/coordination degree of freedom, not Python
    object identity.
    """

    cfg = config or AlignmentConfig()
    if mode not in {"coupled", "decoupled_mirrored", "decoupled_routed"}:
        raise ValueError("unknown mode")
    if not 0.0 <= mismatch_probability <= 1.0:
        raise ValueError("mismatch_probability must be between 0 and 1")

    n = cfg.n_compartments
    rng_weights = np.random.default_rng(seed)
    rng_schedule = np.random.default_rng(seed + 1_000_003)

    weights = np.full((n, n), cfg.initial_weight, dtype=float)
    weights += rng_weights.normal(0.0, cfg.initial_weight_noise, size=(n, n))
    weights = np.maximum(weights, cfg.min_weight)
    weights = heterosynaptic_normalize(weights, per_compartment_budget=cfg.per_compartment_budget)

    if mode == "decoupled_routed":
        plasticity_map = make_plasticity_map(
            seed=seed + 2_000_003,
            n_compartments=n,
            mismatch_probability=mismatch_probability,
        )
    else:
        plasticity_map = np.arange(n, dtype=int)

    expression_trace = CompartmentTrace.zeros(n, config=cfg.trace)
    plasticity_trace = expression_trace if mode == "coupled" else CompartmentTrace.zeros(n, config=cfg.trace)

    schedule = np.tile(np.arange(n, dtype=int), cfg.presentations_per_context)
    rng_schedule.shuffle(schedule)

    publication_target_hits = 0
    ltp_events = 0
    ltd_events = 0
    no_change_events = 0

    for context in schedule:
        pathway = int(context)
        expression_drive = _local_drive(weights[pathway], target=pathway, config=cfg)
        if mode == "decoupled_routed":
            plasticity_drive = _local_drive(
                weights[pathway], target=int(plasticity_map[pathway]), config=cfg
            )
        else:
            plasticity_drive = expression_drive

        for _ in range(cfg.active_steps):
            expression_trace.step(expression_drive)
            if plasticity_trace is not expression_trace:
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
            if plasticity_trace is not expression_trace:
                plasticity_trace.step(zero_drive)

    metrics = alignment_metrics(weights)
    budget_error = float(
        np.max(np.abs(weights.sum(axis=0) - cfg.per_compartment_budget))
    )
    mismatch_fraction = float(np.mean(plasticity_map != np.arange(n)))

    return {
        "mode": mode,
        "seed": int(seed),
        "mismatch_probability": float(mismatch_probability),
        "mapping_mismatch_fraction": mismatch_fraction,
        "alignment_accuracy": metrics["alignment_accuracy"],
        "diagonal_weight_share": metrics["diagonal_weight_share"],
        "publication_target_fraction": float(publication_target_hits / len(schedule)),
        "weight_budget_max_abs_error": budget_error,
        "ltp_events": int(ltp_events),
        "ltd_events": int(ltd_events),
        "no_change_events": int(no_change_events),
        "weights": weights,
        "plasticity_map": plasticity_map,
    }
