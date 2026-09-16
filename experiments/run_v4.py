from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from another_odd_thing.alignment import AlignmentConfig
from another_odd_thing.soft_setpoint import train_soft_misalignment
from another_odd_thing.stats import paired_bootstrap_mean_ci
from another_odd_thing.trace import CompartmentTrace, SingleCompartmentTrace, matched_single_tau_s


CANONICAL_SEEDS = 128
SETPOINT_MISMATCHES = tuple(i / 10.0 for i in range(11))
CANONICAL_BOOTSTRAP_RESAMPLES = 4000
ACCURACY_BOOTSTRAP_SEED = 9162030
WEIGHT_SHARE_BOOTSTRAP_SEED = 9162031


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
    mode: str,
) -> dict[str, object]:
    shares = [float(row[mode]["diagonal_weight_share_mean"]) for row in sweep]
    drops = [a - b for a, b in zip(shares, shares[1:])]
    index = int(np.argmax(drops))
    return {
        "drop": float(drops[index]),
        "from_mismatch": float(sweep[index]["mismatch"]),
        "to_mismatch": float(sweep[index + 1]["mismatch"]),
    }


def _first_below_half_accuracy(
    sweep: list[dict[str, object]],
    *,
    mode: str,
) -> float | None:
    for row in sweep:
        if float(row[mode]["alignment_accuracy_mean"]) < 0.5:
            return float(row["mismatch"])
    return None


def _calibration(config: AlignmentConfig, tau_s: float) -> dict[str, float]:
    two = CompartmentTrace.zeros(1, config=config.trace)
    one = SingleCompartmentTrace.zeros(1, tau_s=tau_s, config=config.trace)
    unit = np.ones(1, dtype=float)
    zero = np.zeros(1, dtype=float)

    for _ in range(config.active_steps):
        two.step(unit)
        one.step(unit)
    two_active = float(two.level[0])
    one_active = float(one.level[0])

    for _ in range(config.rest_steps):
        two.step(zero)
        one.step(zero)
    two_rest = float(two.level[0])
    one_rest = float(one.level[0])

    return {
        "two_timescale_active_readout_level": two_active,
        "single_matched_active_readout_level": one_active,
        "active_readout_level_difference": one_active - two_active,
        "two_timescale_post_rest_level": two_rest,
        "single_matched_post_rest_level": one_rest,
        "post_rest_level_difference": one_rest - two_rest,
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
    matched_tau = matched_single_tau_s(
        config=config.trace,
        active_steps=config.active_steps,
    )
    calibration = _calibration(config, matched_tau)

    sweep: list[dict[str, object]] = []
    midpoint_accuracy_differences: np.ndarray | None = None
    midpoint_weight_share_differences: np.ndarray | None = None

    for mismatch in SETPOINT_MISMATCHES:
        raw_by_mode: dict[str, list[dict[str, object]]] = {}
        for mode in ("two_timescale", "single_matched"):
            raw_by_mode[mode] = [
                train_soft_misalignment(
                    seed=seed,
                    mismatch=mismatch,
                    plasticity_rule="quantized",
                    plasticity_trace_mode=mode,
                    config=config,
                )
                for seed in range(seeds)
            ]

        if np.isclose(mismatch, 0.5):
            two = raw_by_mode["two_timescale"]
            one = raw_by_mode["single_matched"]
            midpoint_accuracy_differences = np.asarray(
                [
                    float(s["alignment_accuracy"])
                    - float(t["alignment_accuracy"])
                    for t, s in zip(two, one, strict=True)
                ],
                dtype=float,
            )
            midpoint_weight_share_differences = np.asarray(
                [
                    float(s["diagonal_weight_share"])
                    - float(t["diagonal_weight_share"])
                    for t, s in zip(two, one, strict=True)
                ],
                dtype=float,
            )

        sweep.append(
            {
                "mismatch": float(mismatch),
                "two_timescale": _summarize(raw_by_mode["two_timescale"]),
                "single_matched": _summarize(raw_by_mode["single_matched"]),
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
    q03 = sweep[3]
    q05 = sweep[5]
    q07 = sweep[7]
    q1 = sweep[-1]
    two_crossing = _first_below_half_accuracy(sweep, mode="two_timescale")
    one_crossing = _first_below_half_accuracy(sweep, mode="single_matched")
    two_steepest = _steepest_drop(sweep, mode="two_timescale")
    one_steepest = _steepest_drop(sweep, mode="single_matched")

    same_knee_bin = (
        two_crossing is not None
        and one_crossing is not None
        and np.isclose(two_crossing, one_crossing)
    )
    same_steepest_bin = (
        np.isclose(two_steepest["from_mismatch"], one_steepest["from_mismatch"])
        and np.isclose(two_steepest["to_mismatch"], one_steepest["to_mismatch"])
    )
    same_qualitative_regime = (
        float(q0["two_timescale"]["alignment_accuracy_mean"]) >= 0.95
        and float(q0["single_matched"]["alignment_accuracy_mean"]) >= 0.95
        and float(q03["two_timescale"]["alignment_accuracy_mean"]) >= 0.90
        and float(q03["single_matched"]["alignment_accuracy_mean"]) >= 0.90
        and float(q05["two_timescale"]["alignment_accuracy_mean"]) < 0.50
        and float(q05["single_matched"]["alignment_accuracy_mean"]) < 0.50
        and float(q07["two_timescale"]["alignment_accuracy_mean"]) < 0.10
        and float(q07["single_matched"]["alignment_accuracy_mean"]) < 0.10
        and float(q1["two_timescale"]["alignment_accuracy_mean"]) < 0.10
        and float(q1["single_matched"]["alignment_accuracy_mean"]) < 0.10
    )

    if same_qualitative_regime and same_knee_bin and same_steepest_bin:
        classification = "SINGLE_TRACE_SUFFICIENT"
    elif (
        float(q05["two_timescale"]["alignment_accuracy_mean"]) >= 0.50
        > float(q05["single_matched"]["alignment_accuracy_mean"])
    ):
        classification = "TWO_TIMESCALE_NEEDED"
    else:
        classification = "MIXED"

    max_accuracy_difference = max(
        abs(
            float(row["single_matched"]["alignment_accuracy_mean"])
            - float(row["two_timescale"]["alignment_accuracy_mean"])
        )
        for row in sweep
    )
    max_weight_share_difference = max(
        abs(
            float(row["single_matched"]["diagonal_weight_share_mean"])
            - float(row["two_timescale"]["diagonal_weight_share_mean"])
        )
        for row in sweep
    )
    max_budget_error = max(
        float(row[mode]["weight_budget_max_abs_error_max"])
        for row in sweep
        for mode in ("two_timescale", "single_matched")
    )

    return {
        "gate": "v4_matched_single_trace_attacker",
        "classification": classification,
        "plasticity_rule": "quantized",
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
        "matched_single_tau_s": matched_tau,
        "calibration": calibration,
        "control_definition": "Replace only the plasticity trace with one leaky scalar. Choose its time constant analytically so a six-step unit burst from rest reaches exactly the same decision level as the original 15/45 ms cascade. Keep expression, inhibitory set-point displacement, thresholds, update rule, schedule and branch normalization unchanged.",
        "sweep": sweep,
        "midpoint_mismatch": 0.5,
        "midpoint_single_minus_two_accuracy_delta_mean": float(
            midpoint_accuracy_differences.mean()
        ),
        "midpoint_accuracy_delta_bootstrap_95": [accuracy_ci[0], accuracy_ci[1]],
        "midpoint_single_minus_two_weight_share_delta_mean": float(
            midpoint_weight_share_differences.mean()
        ),
        "midpoint_weight_share_delta_bootstrap_95": [
            weight_share_ci[0],
            weight_share_ci[1],
        ],
        "shape_summary": {
            "two_timescale_first_below_half_accuracy": two_crossing,
            "single_matched_first_below_half_accuracy": one_crossing,
            "same_half_accuracy_crossing_bin": bool(same_knee_bin),
            "two_timescale_steepest_adjacent_weight_share_drop": two_steepest,
            "single_matched_steepest_adjacent_weight_share_drop": one_steepest,
            "same_steepest_drop_bin": bool(same_steepest_bin),
            "max_abs_alignment_accuracy_difference": float(max_accuracy_difference),
            "max_abs_diagonal_weight_share_difference": float(max_weight_share_difference),
        },
        "max_weight_budget_abs_error": max_budget_error,
        "bootstrap": {
            "accuracy_seed": ACCURACY_BOOTSTRAP_SEED,
            "weight_share_seed": WEIGHT_SHARE_BOOTSTRAP_SEED,
            "resamples": int(bootstrap_resamples),
        },
        "claim": "v4 asks whether the 15/45 ms two-state plasticity trace is required for the v3 set-point knee after matching the scalar attacker's decision level at the canonical six-step burst. Classification is based on whether the two traces occupy the same qualitative accuracy regimes and the same 0.1-grid knee/drop bins, not on forcing their curves to be numerically identical.",
        "boundary": "A matched scalar can reproduce a level-threshold phenomenon without reproducing fast-minus-slow temporal contrast. This test therefore cannot rule out a two-timescale advantage on tasks that explicitly use rise/fall history; it only attacks necessity for the current level-only set-point learning effect.",
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
