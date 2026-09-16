from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidentOperator:
    """Stable linear resident dynamics with one scalar visible readout."""

    matrix: np.ndarray
    initial_state: np.ndarray
    readout: np.ndarray
    name: str = "operator"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        state = np.asarray(self.initial_state, dtype=float)
        readout = np.asarray(self.readout, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        if state.shape != (matrix.shape[0],):
            raise ValueError("initial_state dimension must match matrix")
        if readout.shape != (matrix.shape[0],):
            raise ValueError("readout dimension must match matrix")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "initial_state", state)
        object.__setattr__(self, "readout", readout)

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0])

    def observe(self, state: np.ndarray) -> float:
        return float(self.readout @ np.asarray(state, dtype=float))

    def step(
        self,
        state: np.ndarray,
        *,
        site: int | None = None,
        amplitude: float = 0.0,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=float).copy()
        if site is not None:
            if not 0 <= site < self.dimension:
                raise IndexError("probe site outside state dimension")
            state[site] += float(amplitude)
        return self.matrix @ state


def v0_operators() -> tuple[ResidentOperator, ResidentOperator]:
    initial = np.array([1.0, 0.0])
    readout = np.array([1.0, 0.0])
    op0 = ResidentOperator(
        matrix=np.array([[0.85, 0.00], [0.00, 0.55]]),
        initial_state=initial,
        readout=readout,
        name="O0_uncoupled",
    )
    op1 = ResidentOperator(
        matrix=np.array([[0.85, 0.65], [0.00, 0.55]]),
        initial_state=initial,
        readout=readout,
        name="O1_hidden_coupling",
    )
    return op0, op1


def passive_trace(operator: ResidentOperator, *, steps: int) -> np.ndarray:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    state = operator.initial_state.copy()
    observations = [operator.observe(state)]
    for _ in range(steps):
        state = operator.step(state)
        observations.append(operator.observe(state))
    return np.asarray(observations)


def probe_once(
    operator: ResidentOperator,
    *,
    passive_steps: int,
    site: int,
    amplitude: float,
) -> float:
    state = operator.initial_state.copy()
    for _ in range(passive_steps):
        state = operator.step(state)
    state = operator.step(state, site=site, amplitude=amplitude)
    return operator.observe(state)


def event_bit(value: float, *, threshold: float) -> int:
    return int(float(value) > float(threshold))
