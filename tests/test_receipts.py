from experiments.run_v0 import build_receipt as build_v0_receipt
from experiments.run_v1 import build_receipt as build_v1_receipt
from experiments.run_v2 import build_receipt as build_v2_receipt
from experiments.run_v3 import build_receipt as build_v3_receipt


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


def test_v2_receipt_preserves_oracle_attacker_and_breaks_under_mismatch():
    receipt = build_v2_receipt(seeds=32, bootstrap_resamples=200)
    assert receipt["trace_witness"]["rising_contrast_after_pulse"] > 0.0
    assert receipt["trace_witness"]["falling_contrast_after_six_decay_steps"] < 0.0
    assert receipt["heterosynaptic_normalization_witness"]["inactive_weight_mean_change"] < 0.0

    q0 = receipt["sweep"][0]
    q1 = receipt["sweep"][-1]
    assert q0["coupled"]["alignment_accuracy_mean"] == q0["decoupled_mirrored"]["alignment_accuracy_mean"]
    assert q0["coupled"]["alignment_accuracy_mean"] == q0["decoupled_routed"]["alignment_accuracy_mean"]
    assert q1["coupled"]["alignment_accuracy_mean"] > q1["decoupled_routed"]["alignment_accuracy_mean"]


def test_v3_receipt_separates_quantized_knee_from_continuous_magnitude_control():
    receipt = build_v3_receipt(seeds=24, bootstrap_resamples=200)
    assert receipt["classification"] == "PASS"

    q0 = receipt["sweep"][0]
    qhalf = receipt["sweep"][5]
    q1 = receipt["sweep"][-1]

    assert q0["quantized"]["alignment_accuracy_mean"] > 0.95
    assert q0["continuous"]["alignment_accuracy_mean"] > 0.95
    assert q1["quantized"]["alignment_accuracy_mean"] < 0.10
    assert q1["continuous"]["alignment_accuracy_mean"] < 0.10
    assert qhalf["continuous"]["alignment_accuracy_mean"] > qhalf["quantized"]["alignment_accuracy_mean"]
    assert qhalf["continuous"]["diagonal_weight_share_mean"] > qhalf["quantized"]["diagonal_weight_share_mean"]
    assert receipt["midpoint_accuracy_delta_bootstrap_95"][0] > 0.0
    assert receipt["midpoint_weight_share_delta_bootstrap_95"][0] > 0.0


def test_canonical_receipts_match_frozen_files():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    frozen_v0 = json.loads((root / "results" / "v0.json").read_text())
    frozen_v1 = json.loads((root / "results" / "v1.json").read_text())
    frozen_v2 = json.loads((root / "results" / "v2.json").read_text())
    frozen_v3 = json.loads((root / "results" / "v3.json").read_text())
    assert build_v0_receipt() == frozen_v0
    assert build_v1_receipt() == frozen_v1
    assert build_v2_receipt() == frozen_v2
    assert build_v3_receipt() == frozen_v3
