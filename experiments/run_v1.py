from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from another_odd_thing.identify import V1Config, operator_gain_table, run_paired_trials
from another_odd_thing.stats import paired_bootstrap_mean_ci


CANONICAL_TRIALS = 4096
CANONICAL_SEED = 20260916
BOOTSTRAP_SEED = 9162026
CANONICAL_BOOTSTRAP_RESAMPLES = 4000


def build_receipt(
    *,
    trials: int = CANONICAL_TRIALS,
    seed: int = CANONICAL_SEED,
    bootstrap_resamples: int = CANONICAL_BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    config = V1Config()
    raw = run_paired_trials(trials=trials, seed=seed, config=config)
    differences = np.asarray(raw.pop("paired_accuracy_differences"), dtype=float)
    ci = paired_bootstrap_mean_ci(
        differences,
        seed=BOOTSTRAP_SEED,
        resamples=bootstrap_resamples,
        confidence=0.95,
    )
    passed = raw["active_accuracy"] > raw["random_accuracy"] and ci[0] > 0.0

    return {
        "gate": "v1_active_one_bit_operator_identification",
        "classification": "PASS" if passed else "FAIL",
        "config": {
            "low_gain": config.low_gain,
            "high_gain": config.high_gain,
            "threshold": config.threshold,
            "sigma": config.sigma,
            "probe_budget": config.probe_budget,
            "operator_gain_table": operator_gain_table(
                low=config.low_gain, high=config.high_gain
            ).tolist(),
        },
        **raw,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": int(bootstrap_resamples),
        "paired_accuracy_delta_bootstrap_95": [ci[0], ci[1]],
        "claim": "In this fixed noisy operator family, information-gain probe selection identifies the resident operator more accurately than equal-budget random distinct probes.",
        "boundary": "This is a family-specific experiment-design result, not evidence that active probing is universally superior.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=CANONICAL_TRIALS)
    parser.add_argument("--seed", type=int, default=CANONICAL_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=CANONICAL_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    receipt = build_receipt(
        trials=args.trials,
        seed=args.seed,
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
