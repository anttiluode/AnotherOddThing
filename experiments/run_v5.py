from __future__ import annotations

import argparse
import json
from pathlib import Path

from another_odd_thing.trace import TraceConfig, temporal_direction_metrics


def build_receipt() -> dict[str, object]:
    config = TraceConfig()
    metrics = temporal_direction_metrics(config=config)

    passes = (
        int(metrics["pair_count"]) >= 32
        and float(metrics["max_level_pair_gap"]) < 1e-12
        and float(metrics["two_state_accuracy"]) >= 0.95
        and float(metrics["level_only_accuracy"]) <= 0.55
        and float(metrics["shuffled_contrast_accuracy"]) <= 0.55
        and float(metrics["same_level_rule_fraction"]) == 1.0
        and float(metrics["level_rule_ltd_fraction"]) == 1.0
        and float(metrics["min_abs_contrast"]) > 0.01
    )

    return {
        "gate": "v5_contrast_earns_state",
        "classification": "PASS_CONTRAST_EARNS_STATE" if passes else "FAIL_CONTRAST_EARNS_STATE",
        "config": {
            "dt_s": config.dt_s,
            "tau_fast_s": config.tau_fast_s,
            "tau_slow_s": config.tau_slow_s,
            "theta_d": config.theta_d,
            "theta_p": config.theta_p,
        },
        "metrics": metrics,
        "gate_rule": {
            "minimum_pair_count": 32,
            "maximum_level_pair_gap": 1e-12,
            "minimum_two_state_accuracy": 0.95,
            "maximum_level_only_accuracy": 0.55,
            "maximum_shuffled_contrast_accuracy": 0.55,
            "minimum_abs_contrast": 0.01,
            "require_same_level_rule_fraction": 1.0,
            "require_level_rule_ltd_fraction": 1.0,
        },
        "construction": "Rising histories end during a local burst; falling histories end after the burst. Each history is independently amplitude-rescaled so the slow decision level is exactly matched within each pair. All target levels lie between theta_d and theta_p, so the old level-only categorical rule assigns LTD to both members. Only fast-minus-slow retains rise/fall direction.",
        "attacker": "The level-only attacker sees only the exactly matched present slow level. It must emit the same prediction for both members of a pair and is therefore limited to one correct label per balanced pair. A second control swaps the two contrasts within exactly half of the matched pairs while preserving every level and contrast value.",
        "claim": "For this deliberately constructed temporal-direction task, the existing two-timescale local trace contains a second coordinate that distinguishes entering from leaving a write window when the present slow plasticity level is held fixed. The extra state therefore earns a narrow role that v4's level-only set-point task could not test.",
        "boundary": "This is a known-answer mechanism witness. It does not show that biology reads fast-minus-slow as phase, that the sign is a learned rule, or that this trace generates theta/BTSP windows. It only establishes that a local two-state trace can carry temporal direction that an exactly matched one-state level cannot carry.",
        "lineage": "v4 found one matched scalar sufficient for the level-only set-point knee and explicitly left temporal contrast unemployed. TATWATASW then made temporal direction inside a write window computationally relevant. v5 supplies that missing task without importing TATWATASW's imposed global phase variable.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    receipt = build_receipt()
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(text, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
