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


def plasticity_sign(level: np.ndarray | float, *, config: TraceConfig) -> np.ndarray:
    """Return the calcium-style 0 / LTD / LTP sign readout.

    Below theta_d there is no update, between theta_d and theta_p there is
    depression, and at/above theta_p there is potentiation.
    """

    values = np.asarray(level, dtype=float)
    return np.where(values >= config.theta_p, 1, np.where(values >= config.theta_d, -1, 0)).astype(int)
