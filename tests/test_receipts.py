from experiments.run_v0 import build_receipt as build_v0_receipt
from experiments.run_v1 import build_receipt as build_v1_receipt


def test_v0_receipt_contains_destructive_controls():
    receipt = build_v0_receipt()
    assert receipt["classification"] == "PASS"
    assert receipt["passive_max_abs_difference"] == 0.0
    assert receipt["visible_probe_abs_difference"] == 0.0
    assert receipt["hidden_probe_abs_difference"] > 0.0
    assert receipt["hidden_probe_event_bits"] == [0, 1]


def test_v1_receipt_records_paired_budget_and_interval():
    receipt = build_v1_receipt(trials=512, seed=20260916, bootstrap_resamples=500)
    assert receipt["active_probe_count_mean"] == receipt["random_probe_count_mean"] == 2.0
    assert receipt["active_accuracy"] > receipt["random_accuracy"]
    lo, hi = receipt["paired_accuracy_delta_bootstrap_95"]
    assert lo <= receipt["paired_accuracy_delta"] <= hi


def test_canonical_receipts_match_frozen_files():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    frozen_v0 = json.loads((root / "results" / "v0.json").read_text())
    frozen_v1 = json.loads((root / "results" / "v1.json").read_text())
    assert build_v0_receipt() == frozen_v0
    assert build_v1_receipt() == frozen_v1
