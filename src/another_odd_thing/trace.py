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
