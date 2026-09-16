from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from another_odd_thing.core import event_bit, passive_trace, probe_once, v0_operators


PASSIVE_STEPS = 4
PROBE_AMPLITUDE = 1.0
THRESHOLD = 0.75


def build_receipt() -> dict[str, object]:
    op0, op1 = v0_operators()
    trace0 = passive_trace(op0, steps=PASSIVE_STEPS)
    trace1 = passive_trace(op1, steps=PASSIVE_STEPS)
    visible0 = probe_once(op0, passive_steps=PASSIVE_STEPS, site=0, amplitude=PROBE_AMPLITUDE)
    visible1 = probe_once(op1, passive_steps=PASSIVE_STEPS, site=0, amplitude=PROBE_AMPLITUDE)
    hidden0 = probe_once(op0, passive_steps=PASSIVE_STEPS, site=1, amplitude=PROBE_AMPLITUDE)
    hidden1 = probe_once(op1, passive_steps=PASSIVE_STEPS, site=1, amplitude=PROBE_AMPLITUDE)
    bits = [event_bit(hidden0, threshold=THRESHOLD), event_bit(hidden1, threshold=THRESHOLD)]

    passive_delta = float(np.max(np.abs(trace0 - trace1)))
    visible_delta = float(abs(visible1 - visible0))
    hidden_delta = float(abs(hidden1 - hidden0))
    passed = passive_delta == 0.0 and visible_delta == 0.0 and hidden_delta > 0.0 and bits == [0, 1]

    return {
        "gate": "v0_passive_equivalence_interventional_separation",
        "classification": "PASS" if passed else "FAIL",
        "passive_steps": PASSIVE_STEPS,
        "probe_amplitude": PROBE_AMPLITUDE,
        "publication_threshold": THRESHOLD,
        "operators": {
            op0.name: op0.matrix.tolist(),
            op1.name: op1.matrix.tolist(),
        },
        "passive_trace": trace0.tolist(),
        "passive_max_abs_difference": passive_delta,
        "visible_probe_responses": [visible0, visible1],
        "visible_probe_abs_difference": visible_delta,
        "hidden_probe_responses": [hidden0, hidden1],
        "hidden_probe_abs_difference": hidden_delta,
        "hidden_probe_event_bits": bits,
        "claim": "Passive equality does not imply equal resident operators; an addressed hidden-state perturbation can expose the difference.",
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
