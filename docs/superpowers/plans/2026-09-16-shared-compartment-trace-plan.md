# Shared Compartment Trace v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compartment-shared two-timescale trace and test whether coupling expression and plasticity removes an otherwise explicit context-to-compartment coordination requirement.

**Architecture:** Add a focused `trace.py` for the local two-timescale state and threshold readouts, plus `alignment.py` for the four-context learning world and coupled/decoupled controls. `run_v2.py` freezes the scientific receipt. Tests cover trace dynamics, three-regime signs, heterosynaptic normalization, oracle decoupled parity, and mismatch degradation.

**Tech Stack:** Python 3.11+, NumPy, pytest, dependency-free HTML lab.

**Spec:** `docs/superpowers/specs/2026-09-16-shared-compartment-trace-design.md`

## Global Constraints

- Synthetic mechanism test only; do not claim biological reproduction.
- Shared trace is compartment-indexed, never per-weight.
- Decoupled `q=0` is an oracle attacker and is allowed to match coupled.
- Same trial schedule, initial weights, thresholds, learning rate, and normalization are used across coupled/decoupled conditions.
- Frozen results must reproduce deterministically in CI.

---

### Task 1: Shared trace primitive

**Files:** create `src/another_odd_thing/trace.py`; create `tests/test_trace.py`.

**Interfaces:** `TraceConfig`, `CompartmentTrace`, `plasticity_sign(level, config)`.

- [ ] Write failing tests for rising/falling contrast and the 0/-1/+1 threshold regimes.
- [ ] Implement the two-stage leaky cascade and readouts.
- [ ] Run focused tests.

### Task 2: Alignment learning world

**Files:** create `src/another_odd_thing/alignment.py`; create `tests/test_alignment.py`.

**Interfaces:** `AlignmentConfig`, `make_plasticity_map`, `train_alignment`, `alignment_metrics`, `heterosynaptic_normalize`.

- [ ] Write failing tests showing normalization changes inactive synapses while preserving a per-compartment budget.
- [ ] Write the oracle-control test where coupled and aligned-decoupled share the same mapping.
- [ ] Write the corrupted-mapping test where decoupled alignment degrades.
- [ ] Implement minimal learning world and controls.
- [ ] Run focused tests.

### Task 3: Frozen experiment

**Files:** create `experiments/run_v2.py`; create `results/v2.json`; modify `tests/test_receipts.py`.

- [ ] Add receipt regression test first.
- [ ] Run a deterministic mismatch sweep across multiple seeds.
- [ ] Freeze only metrics needed to support/attack the claim.
- [ ] Regenerate and compare the receipt.

### Task 4: Documentation and live microscope

**Files:** modify `README.md`, `docs/RELATED_WORK.md`, `index.html`, `.github/workflows/ci.yml` if needed.

- [ ] Document the distinction between Yang-style calcium thresholds and Onasch-style voltage timescales.
- [ ] Add v2 visualization for shared trace and alignment-vs-mismatch.
- [ ] Ensure CI regenerates `v2.json` and checks browser JS syntax.

### Task 5: Verification

- [ ] Run full pytest suite.
- [ ] Reproduce all frozen receipts byte-for-byte.
- [ ] Run browser JavaScript syntax check.
- [ ] Open PR, wait for Python 3.11/3.12 CI, and merge only if green.
