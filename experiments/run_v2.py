from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from another_odd_thing.alignment import (
    AlignmentConfig,
    heterosynaptic_normalize,
    train_alignment,
)
from another_odd_thing.stats import paired_bootstrap_mean_ci
from another_odd_thing.trace import CompartmentTrace


CANONICAL_SEEDS = 128
MISMATCH_PROBABILITIES = (0.0, 0.25, 0.50, 0.75, 1.0)
BOOTSTRAP_SEED = 9162027
CANONICAL_BOOTSTRAP_RESAMPLES = 4000


def _summarize(raw: list[dict[str, object]]) -> dict[str, float]:
    return {
        "alignment_accuracy_mean": float(np.mean([float(r["alignment_accuracy"]) for r in raw])),
        "diagonal_weight_share_mean": float(
            np.mean([float(r["diagonal_weight_share"]) for r in raw])
        ),
        "publication_target_fraction_mean": float(
            np.mean([float(r["publication_target_fraction"]) for r in raw])
        ),
        "mapping_mismatch_fraction_mean": float(
            np.mean([float(r["mapping_mismatch_fraction"]) for r in raw])
        ),
        "weight_budget_max_abs_error_max": float(
            np.max([float(r["weight_budget_max_abs_error"]) for r in raw])
        ),
    }


def build_receipt(
    *,
    seeds: int = CANONICAL_SEEDS,
    bootstrap_resamples: int = CANONICAL_BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    if seeds <= 0:
        raise ValueError("seeds must be positive")

    config = AlignmentConfig()

    trace = CompartmentTrace.zeros(1, config=config.trace)
    for _ in range(config.active_steps):
        trace.step(np.array([1.0]))
    rising_contrast = float(trace.contrast[0])
    slow_after_pulse = float(trace.level[0])
    for _ in range(config.rest_steps):
        trace.step(np.array([0.0]))
    falling_contrast = float(trace.contrast[0])
    slow_after_decay = float(trace.level[0])

    normalization_before = np.ones((4, 4), dtype=float)
    normalization_before[0, 0] += 0.40
    normalization_after = heterosynaptic_normalize(
        normalization_before,
        per_compartment_budget=4.0,
    )

    sweep: list[dict[str, object]] = []
    midpoint_differences: np.ndarray | None = None

    for mismatch_probability in MISMATCH_PROBABILITIES:
        raw_by_mode: dict[str, list[dict[str, object]]] = {}
        for mode in ("coupled", "decoupled_mirrored", "decoupled_routed"):
            raw_by_mode[mode] = [
                train_alignment(
                    seed=seed,
                    mode=mode,
                    mismatch_probability=mismatch_probability,
                    config=config,
                )
                for seed in range(seeds)
            ]

        coupled = raw_by_mode["coupled"]
        routed = raw_by_mode["decoupled_routed"]
        if mismatch_probability == 0.50:
            midpoint_differences = np.asarray(
                [
                    float(c["alignment_accuracy"]) - float(r["alignment_accuracy"])
                    for c, r in zip(coupled, routed, strict=True)
                ],
                dtype=float,
            )

        sweep.append(
            {
                "mismatch_probability": float(mismatch_probability),
                "coupled": _summarize(raw_by_mode["coupled"]),
                "decoupled_mirrored": _summarize(raw_by_mode["decoupled_mirrored"]),
                "decoupled_routed": _summarize(raw_by_mode["decoupled_routed"]),
            }
        )

    assert midpoint_differences is not None
    midpoint_ci = paired_bootstrap_mean_ci(
        midpoint_differences,
        seed=BOOTSTRAP_SEED,
        resamples=bootstrap_resamples,
        confidence=0.95,
    )

    routed_alignment = [
        float(row["decoupled_routed"]["alignment_accuracy_mean"]) for row in sweep
    ]
    q0 = sweep[0]
    q1 = sweep[-1]
    passed = (
        rising_contrast > 0.0
        and falling_contrast < 0.0
        and float(q0["coupled"]["alignment_accuracy_mean"]) >= 0.95
        and float(q0["decoupled_mirrored"]["alignment_accuracy_mean"])
        == float(q0["coupled"]["alignment_accuracy_mean"])
        and float(q0["decoupled_routed"]["alignment_accuracy_mean"])
        == float(q0["coupled"]["alignment_accuracy_mean"])
        and float(q1["decoupled_routed"]["alignment_accuracy_mean"]) <= 0.25
        and all(a >= b for a, b in zip(routed_alignment, routed_alignment[1:]))
        and midpoint_ci[0] > 0.0
        and float(np.mean(normalization_after[1:, 0] - 1.0)) < 0.0
    )

    return {
        "gate": "v2_shared_compartment_trace_alignment",
        "classification": "PASS" if passed else "FAIL",
        "config": {
            "n_compartments": config.n_compartments,
            "initial_weight": config.initial_weight,
            "initial_weight_noise": config.initial_weight_noise,
            "open_inhibition": config.open_inhibition,
            "closed_inhibition": config.closed_inhibition,
            "learning_rate": config.learning_rate,
            "active_steps": config.active_steps,
            "rest_steps": config.rest_steps,
            "presentations_per_context": config.presentations_per_context,
            "min_weight": config.min_weight,
            "per_compartment_budget": config.per_compartment_budget,
            "trace": {
                "dt_s": config.trace.dt_s,
                "tau_fast_s": config.trace.tau_fast_s,
                "tau_slow_s": config.trace.tau_slow_s,
                "theta_d": config.trace.theta_d,
                "theta_p": config.trace.theta_p,
            },
        },
        "seeds": int(seeds),
        "mismatch_probabilities": [float(x) for x in MISMATCH_PROBABILITIES],
        "trace_witness": {
            "rising_contrast_after_pulse": rising_contrast,
            "slow_level_after_pulse": slow_after_pulse,
            "falling_contrast_after_six_decay_steps": falling_contrast,
            "slow_level_after_six_decay_steps": slow_after_decay,
        },
        "heterosynaptic_normalization_witness": {
            "active_weight_before": float(normalization_before[0, 0]),
            "active_weight_after": float(normalization_after[0, 0]),
            "inactive_weight_before": 1.0,
            "inactive_weight_after_mean": float(np.mean(normalization_after[1:, 0])),
            "inactive_weight_mean_change": float(
                np.mean(normalization_after[1:, 0] - 1.0)
            ),
            "compartment_budget_after": float(normalization_after[:, 0].sum()),
        },
        "sweep": sweep,
        "midpoint_mismatch_probability": 0.50,
        "midpoint_coupled_minus_routed_alignment_delta_mean": float(
            midpoint_differences.mean()
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": int(bootstrap_resamples),
        "midpoint_paired_alignment_delta_bootstrap_95": [midpoint_ci[0], midpoint_ci[1]],
        "claim": "In this toy local-learning system, sharing the compartment signal used by expression and plasticity removes a context-to-compartment coordination requirement. A separately routed plasticity signal matches when aligned and degrades when that routing is corrupted.",
        "boundary": "The mirrored independent-trace control matches the coupled machine when it receives the same local signal. The result is about coordination/routing, not evidence that sharing a state variable is intrinsically superior or that this toy reproduces calcium biology.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=CANONICAL_SEEDS)
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=CANONICAL_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    receipt = build_receipt(
        seeds=args.seeds,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(text, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
