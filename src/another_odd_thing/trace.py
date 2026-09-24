from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TraceConfig:
    """Two-timescale local trace used by the v2 compartment experiment.

    The default 15 ms / 45 ms constants are inspired by the two dendritic
    voltage filters used by Onasch et al. They are not a reproduction of that
    plasticity rule. The two thresholds implement a deliberately simpler
    calcium-style three-regime readout.
    """

    dt_s: float = 0.005
    tau_fast_s: float = 0.015
    tau_slow_s: float = 0.045
    theta_d: float = 0.10
    theta_p: float = 0.32

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.tau_fast_s <= 0.0 or self.tau_slow_s <= 0.0:
            raise ValueError("time constants must be positive")
        if self.dt_s > self.tau_fast_s or self.dt_s > self.tau_slow_s:
            raise ValueError("dt_s must not exceed either time constant")
        if not self.theta_d < self.theta_p:
            raise ValueError("theta_d must be below theta_p")


@dataclass
class CompartmentTrace:
    """Shared analog state indexed by compartment, never by synapse."""

    fast: np.ndarray
    slow: np.ndarray
    config: TraceConfig

    def __post_init__(self) -> None:
        self.fast = np.asarray(self.fast, dtype=float).copy()
        self.slow = np.asarray(self.slow, dtype=float).copy()
        if self.fast.ndim != 1 or self.slow.shape != self.fast.shape:
            raise ValueError("fast and slow must be same-length 1D arrays")

    @classmethod
    def zeros(cls, compartments: int, *, config: TraceConfig | None = None) -> "CompartmentTrace":
        if compartments <= 0:
            raise ValueError("compartments must be positive")
        cfg = config or TraceConfig()
        return cls(np.zeros(compartments), np.zeros(compartments), cfg)

    @property
    def level(self) -> np.ndarray:
        """Slow local level used by the simple three-regime plasticity readout."""

        return self.slow.copy()

    @property
    def contrast(self) -> np.ndarray:
        """Fast-minus-slow temporal contrast: positive on rise, negative on decay."""

        return self.fast - self.slow

    def step(self, drive: np.ndarray) -> None:
        drive = np.asarray(drive, dtype=float)
        if drive.shape != self.fast.shape:
            raise ValueError("drive shape must match compartment count")

        af = self.config.dt_s / self.config.tau_fast_s
        ass = self.config.dt_s / self.config.tau_slow_s
        self.fast += af * (drive - self.fast)
        # Deliberately cascade slow from the newly updated fast state.
        self.slow += ass * (self.fast - self.slow)


@dataclass
class SingleCompartmentTrace:
    """Matched one-timescale attacker for the two-state compartment trace.

    The scalar state is calibrated separately by ``matched_single_tau_s`` so
    that a canonical active burst reaches the same plasticity decision level as
    ``CompartmentTrace``. Its subsequent history is intentionally different.
    """

    state: np.ndarray
    tau_s: float
    config: TraceConfig

    def __post_init__(self) -> None:
        self.state = np.asarray(self.state, dtype=float).copy()
        if self.state.ndim != 1:
            raise ValueError("state must be a 1D array")
        if self.tau_s <= 0.0:
            raise ValueError("tau_s must be positive")
        if self.config.dt_s > self.tau_s:
            raise ValueError("dt_s must not exceed tau_s")

    @classmethod
    def zeros(
        cls,
        compartments: int,
        *,
        tau_s: float,
        config: TraceConfig | None = None,
    ) -> "SingleCompartmentTrace":
        if compartments <= 0:
            raise ValueError("compartments must be positive")
        cfg = config or TraceConfig()
        return cls(np.zeros(compartments), float(tau_s), cfg)

    @property
    def level(self) -> np.ndarray:
        return self.state.copy()

    def step(self, drive: np.ndarray) -> None:
        drive = np.asarray(drive, dtype=float)
        if drive.shape != self.state.shape:
            raise ValueError("drive shape must match compartment count")
        alpha = self.config.dt_s / self.tau_s
        self.state += alpha * (drive - self.state)


def matched_single_tau_s(*, config: TraceConfig, active_steps: int) -> float:
    """Match a scalar leaky trace to the two-timescale level after one burst.

    Starting both traces from rest and driving them with unit input for
    ``active_steps``, solve the scalar time constant analytically so its final
    level equals the cascaded trace's slow level. This controls the decision
    coordinate at the plasticity readout while leaving temporal history free to
    differ.
    """

    if active_steps <= 0:
        raise ValueError("active_steps must be positive")

    reference = CompartmentTrace.zeros(1, config=config)
    unit_drive = np.ones(1, dtype=float)
    for _ in range(active_steps):
        reference.step(unit_drive)
    target = float(reference.level[0])
    if not 0.0 < target < 1.0:
        raise ValueError("matched target must lie strictly between zero and one")

    alpha = 1.0 - (1.0 - target) ** (1.0 / active_steps)
    return float(config.dt_s / alpha)


def plasticity_sign(level: np.ndarray | float, *, config: TraceConfig) -> np.ndarray:
    """Return the calcium-style 0 / LTD / LTP sign readout.

    Below theta_d there is no update, between theta_d and theta_p there is
    depression, and at/above theta_p there is potentiation.
    """

    values = np.asarray(level, dtype=float)
    return np.where(values >= config.theta_p, 1, np.where(values >= config.theta_d, -1, 0)).astype(int)


def _trace_endpoint(drive: np.ndarray, *, config: TraceConfig) -> tuple[float, float]:
    """Return (slow level, fast-minus-slow contrast) after one local history."""

    state = CompartmentTrace.zeros(1, config=config)
    for value in np.asarray(drive, dtype=float):
        state.step(np.asarray([value], dtype=float))
    return float(state.level[0]), float(state.contrast[0])


def matched_temporal_direction_pairs(
    *,
    config: TraceConfig | None = None,
) -> dict[str, np.ndarray]:
    """Build histories with the same present level but opposite temporal direction.

    Each rising history ends during a burst; each falling history ends after the
    burst has stopped. Because the trace is linear in drive amplitude, both
    histories in a pair are rescaled to the same slow level. The old level-only
    readout therefore receives exactly the same present decision coordinate,
    while ``fast - slow`` is free to report whether activity is entering or
    leaving the local window.

    The target levels are deliberately kept between the default LTD/LTP
    thresholds, so the existing categorical level rule also makes the same LTD
    decision for both histories.
    """

    cfg = config or TraceConfig()
    targets = np.linspace(0.14, 0.30, 8)
    rising_steps = (5, 6, 7, 8)
    falling_specs = ((5, 4), (6, 4), (7, 4), (8, 4))

    rising_rows: list[tuple[float, float]] = []
    falling_rows: list[tuple[float, float]] = []

    for target in targets:
        for rise_steps in rising_steps:
            rising_template = np.ones(rise_steps, dtype=float)
            rising_base_level, _ = _trace_endpoint(rising_template, config=cfg)
            rising_drive = rising_template * (float(target) / rising_base_level)
            rising_level, rising_contrast = _trace_endpoint(rising_drive, config=cfg)

            for active_steps, rest_steps in falling_specs:
                falling_template = np.concatenate(
                    [
                        np.ones(active_steps, dtype=float),
                        np.zeros(rest_steps, dtype=float),
                    ]
                )
                falling_base_level, _ = _trace_endpoint(falling_template, config=cfg)
                falling_drive = falling_template * (float(target) / falling_base_level)
                falling_level, falling_contrast = _trace_endpoint(falling_drive, config=cfg)

                rising_rows.append((rising_level, rising_contrast))
                falling_rows.append((falling_level, falling_contrast))

    return {
        "rising": np.asarray(rising_rows, dtype=float),
        "falling": np.asarray(falling_rows, dtype=float),
    }


def temporal_direction_metrics(
    *,
    config: TraceConfig | None = None,
) -> dict[str, float | int]:
    """Score whether the second trace state earns a temporal-direction role."""

    cfg = config or TraceConfig()
    pairs = matched_temporal_direction_pairs(config=cfg)
    rising = pairs["rising"]
    falling = pairs["falling"]
    if rising.shape != falling.shape or rising.ndim != 2 or rising.shape[1] != 2:
        raise RuntimeError("matched temporal pairs have an invalid shape")

    n = int(rising.shape[0])
    pair_levels = 0.5 * (rising[:, 0] + falling[:, 0])
    labels = np.column_stack([np.ones(n, dtype=int), -np.ones(n, dtype=int)])

    # Two-state rule: temporal direction is the sign of fast-minus-slow.
    two_predictions = np.column_stack(
        [
            np.where(rising[:, 1] >= 0.0, 1, -1),
            np.where(falling[:, 1] >= 0.0, 1, -1),
        ]
    )
    two_state_accuracy = float(np.mean(two_predictions == labels))

    # Matched level-only attacker: both members of a pair necessarily receive
    # the same prediction because their present slow levels are identical.
    median_level = float(np.median(pair_levels))
    level_prediction = np.where(pair_levels >= median_level, 1, -1)
    level_predictions = np.column_stack([level_prediction, level_prediction])
    level_only_accuracy = float(np.mean(level_predictions == labels))

    # Negative control: destroy the contrast/label relation in exactly half of
    # the matched pairs by swapping the two contrasts. This keeps every value,
    # level and class count unchanged while forcing chance accuracy overall.
    shuffled_predictions = two_predictions.copy()
    swap = np.arange(n) % 2 == 1
    shuffled_predictions[swap] = shuffled_predictions[swap, ::-1]
    shuffled_contrast_accuracy = float(np.mean(shuffled_predictions == labels))

    rising_sign = plasticity_sign(rising[:, 0], config=cfg)
    falling_sign = plasticity_sign(falling[:, 0], config=cfg)

    return {
        "pair_count": n,
        "max_level_pair_gap": float(np.max(np.abs(rising[:, 0] - falling[:, 0]))),
        "two_state_accuracy": two_state_accuracy,
        "level_only_accuracy": level_only_accuracy,
        "shuffled_contrast_accuracy": shuffled_contrast_accuracy,
        "min_abs_contrast": float(
            min(np.min(np.abs(rising[:, 1])), np.min(np.abs(falling[:, 1])))
        ),
        "mean_rising_contrast": float(np.mean(rising[:, 1])),
        "mean_falling_contrast": float(np.mean(falling[:, 1])),
        "target_level_min": float(np.min(pair_levels)),
        "target_level_max": float(np.max(pair_levels)),
        "same_level_rule_fraction": float(np.mean(rising_sign == falling_sign)),
        "level_rule_ltd_fraction": float(
            np.mean((rising_sign == -1) & (falling_sign == -1))
        ),
    }
