from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from another_odd_thing.alignment import AlignmentConfig
from another_odd_thing.soft_setpoint import train_soft_misalignment
from another_odd_thing.stats import paired_bootstrap_mean_ci


CANONICAL_SEEDS = 128
SETPOINT_MISMATCHES = tuple(i / 10.0 for i in range(11))
CANONICAL_BOOTSTRAP_RESAMPLES = 4000
ACCURACY_BOOTSTRAP_SEED = 9162028
WEIGHT_SHARE_BOOTSTRAP_SEED = 9162029


def _summarize(raw: list[dict[str, object]]) -> dict[str, float]:
    return {
        "alignment_accuracy_mean": float(
            np.mean([float(row["alignment_accuracy"]) for row in raw])
        ),
        "diagonal_weight_share_mean": float(
            np.mean([float(row["diagonal_weight_share"]) for row in raw])
        ),
        "publication_target_fraction_mean": float(
            np.mean([float(row["publication_target_fraction"]) for row in raw])
        ),
        "weight_budget_max_abs_error_max": float(
            np.max([float(row["weight_budget_max_abs_error"]) for row in raw])
        ),
    }


def _steepest_drop(
    sweep: list[dict[str, object]],
    *,
    rule: str,
) -> dict[str, object]:
    shares = [float(row[rule]["diagonal_weight_share_mean"]) for row in sweep]
    drops = [a - b for a, b in zip(shares, shares[1:])]
    index = int(np.argmax(drops))
    return {
        "drop": float(drops[index]),
        "from_mismatch": float(sweep[index]["mismatch"]),
        "to_mismatch": float(sweep[index + 1]["mismatch"]),
    }


def build_receipt(
    *,
    seeds: int = CANONICAL_SEEDS,
    bootstrap_resamples: int = CANONICAL_BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")

    config = AlignmentConfig()
    sweep: list[dict[str, object]] = []
    midpoint_accuracy_differences: np.ndarray | None = None
    midpoint_weight_share_differences: np.ndarray | None = None

    for mismatch in SETPOINT_MISMATCHES:
        raw_by_rule: dict[str, list[dict[str, object]]] = {}
        for rule in ("quantized", "continuous"):
            raw_by_rule[rule] = [
                train_soft_misalignment(
                    seed=seed,
                    mismatch=mismatch,
                    plasticity_rule=rule,
                    config=config,
                )
                for seed in range(seeds)
            ]

        if np.isclose(mismatch, 0.5):
            quantized = raw_by_rule["quantized"]
            continuous = raw_by_rule["continuous"]
            midpoint_accuracy_differences = np.asarray(
                [
                    float(c["alignment_accuracy"]) - float(q["alignment_accuracy"])
                    for q, c in zip(quantized, continuous, strict=True)
                ],
                dtype=float,
            )
            midpoint_weight_share_differences = np.asarray(
                [
                    float(c["diagonal_weight_share"])
                    - float(q["diagonal_weight_share"])
                    for q, c in zip(quantized, continuous, strict=True)
                ],
                dtype=float,
            )

        sweep.append(
            {
                "mismatch": float(mismatch),
                "quantized": _summarize(raw_by_rule["quantized"]),
                "continuous": _summarize(raw_by_rule["continuous"]),
            }
        )

    assert midpoint_accuracy_differences is not None
    assert midpoint_weight_share_differences is not None

    accuracy_ci = paired_bootstrap_mean_ci(
        midpoint_accuracy_differences,
        seed=ACCURACY_BOOTSTRAP_SEED,
        resamples=bootstrap_resamples,
        confidence=0.95,
    )
    weight_share_ci = paired_bootstrap_mean_ci(
        midpoint_weight_share_differences,
        seed=WEIGHT_SHARE_BOOTSTRAP_SEED,
        resamples=bootstrap_resamples,
        confidence=0.95,
    )

    q0 = sweep[0]
    q02 = sweep[2]
    q05 = sweep[5]
    q07 = sweep[7]
    q1 = sweep[-1]
    max_budget_error = max(
        float(row[rule]["weight_budget_max_abs_error_max"])
        for row in sweep
        for rule in ("quantized", "continuous")
    )

    passed = (
        float(q0["quantized"]["alignment_accuracy_mean"]) >= 0.95
        and float(q0["continuous"]["alignment_accuracy_mean"]) >= 0.95
        and float(q02["quantized"]["alignment_accuracy_mean"]) >= 0.95
        and float(q07["quantized"]["alignment_accuracy_mean"]) <= 0.10
        and float(q1["quantized"]["alignment_accuracy_mean"]) <= 0.10
        and float(q1["continuous"]["alignment_accuracy_mean"]) <= 0.10
        and float(q05["continuous"]["alignment_accuracy_mean"])
        > float(q05["quantized"]["alignment_accuracy_mean"])
        and float(q05["continuous"]["diagonal_weight_share_mean"])
        > float(q05["quantized"]["diagonal_weight_share_mean"])
        and accuracy_ci[0] > 0.0
        and weight_share_ci[0] > 0.0
        and max_budget_error < 1e-12
    )

    return {
        "gate": "v3_continuous_inhibitory_setpoint",
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
        "setpoint_mismatches": [float(x) for x in SETPOINT_MISMATCHES],
        "continuous_control": {
            "description": "The continuous control preserves the exact no-update region below theta_d. At theta_d it is -1, it interpolates linearly through 0 at (theta_d+theta_p)/2 to +1 at theta_p, and then saturates at +1.",
            "purpose": "Attack the possibility that the apparent knee is created only by winner-take-all measurement or recurrent normalization, without granting the control an extra subthreshold LTD mechanism.",
        },
        "sweep": sweep,
        "midpoint_mismatch": 0.5,
        "midpoint_continuous_minus_quantized_accuracy_delta_mean": float(
            midpoint_accuracy_differences.mean()
        ),
        "midpoint_accuracy_delta_bootstrap_95": [accuracy_ci[0], accuracy_ci[1]],
        "midpoint_continuous_minus_quantized_weight_share_delta_mean": float(
            midpoint_weight_share_differences.mean()
        ),
        "midpoint_weight_share_delta_bootstrap_95": [
            weight_share_ci[0],
            weight_share_ci[1],
        ],
        "shape_summary": {
            "quantized_steepest_adjacent_weight_share_drop": _steepest_drop(
                sweep,
                rule="quantized",
            ),
            "continuous_steepest_adjacent_weight_share_drop": _steepest_drop(
                sweep,
                rule="continuous",
            ),
        },
        "max_weight_budget_abs_error": max_budget_error,
        "bootstrap": {
            "accuracy_seed": ACCURACY_BOOTSTRAP_SEED,
            "weight_share_seed": WEIGHT_SHARE_BOOTSTRAP_SEED,
            "resamples": int(bootstrap_resamples),
        },
        "claim": "In this fixed toy world, continuously moving inhibitory relief away from the expression-aligned compartment produces a robust region followed by failure under the quantized three-regime plasticity readout. A continuous-magnitude control with the same subthreshold no-update floor and threshold anchors preserves substantially more alignment at the middle of the sweep, including in continuous weight-share measurements.",
        "boundary": "This isolates a contribution of categorical local plasticity in this construction; it does not establish a biological calcium mechanism, universal criticality, or that the chosen thresholds are optimal. The continuous control is an attacker, not a biological alternative model.",
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
