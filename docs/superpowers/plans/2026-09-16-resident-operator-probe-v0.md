# Resident Operator Probe v0/v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic research repo showing passive operator ambiguity, addressed interventional separation, and an equal-budget active-probe identification advantage through a one-bit response channel.

**Architecture:** A tiny NumPy resident linear system provides the causal witness. A separate Bayesian identification module computes Bernoulli likelihoods and expected information gain over a four-operator codebook, while a statistics helper computes a paired bootstrap interval. Deterministic CLIs freeze JSON receipts and a dependency-free HTML page visualizes the same mechanisms.

**Tech Stack:** Python 3.11+, NumPy, pytest, standard-library JSON/argparse/math, vanilla HTML/CSS/JavaScript.

**Spec:** `docs/superpowers/specs/2026-09-16-resident-operator-probe-design.md`

## Global Constraints

- Biological vocabulary is motivational; no claim that dendrites are arbitrary programs or that interneurons perform the exact algorithm.
- v0 must include no-probe and wrong-site controls.
- v1 active and random policies get exactly two distinct probe sites.
- v1 trials are paired on true operator and Gaussian noise tape.
- Canonical v1 uses 4096 trials, seed 20260916, bootstrap seed 9162026, and 4000 bootstrap resamples.
- CI must not require a preferred scientific sign except in frozen receipt regression tests.

---

### Task 1: Resident operator causal witness

**Files:**
- Create: `tests/test_core.py`
- Create: `src/another_odd_thing/core.py`
- Create: `src/another_odd_thing/__init__.py`

**Interfaces:**
- Produces: `ResidentOperator`, `v0_operators()`, `passive_trace()`, `probe_once()`, `event_bit()`.

- [ ] **Step 1: Write failing tests** asserting exact passive-trace equality, wrong-site equality, hidden-site separation, and one-bit operator separation.
- [ ] **Step 2: Run `PYTHONPATH=src pytest tests/test_core.py -q` and verify import/behavior failure.**
- [ ] **Step 3: Implement the minimal resident operator dynamics.**
- [ ] **Step 4: Re-run the tests and verify they pass.**

### Task 2: Bayesian active probe selection

**Files:**
- Create: `tests/test_identify.py`
- Create: `src/another_odd_thing/identify.py`

**Interfaces:**
- Consumes: probe semantics from `core.py` conceptually, but owns its fixed v1 operator family.
- Produces: `operator_gain_table()`, `event_probability()`, `posterior_update()`, `expected_information_gain()`, `choose_active_site()`, `run_paired_trials()`.

- [ ] **Step 1: Write failing tests** for normalized posteriors, site-1 first choice from the uniform prior, adaptive second-site selection, and equal probe budgets.
- [ ] **Step 2: Run `PYTHONPATH=src pytest tests/test_identify.py -q` and verify failure.**
- [ ] **Step 3: Implement Bernoulli likelihoods, entropy/information gain, active/random policies, and paired common-noise trials.**
- [ ] **Step 4: Re-run tests and verify they pass.**

### Task 3: Paired uncertainty interval and deterministic receipts

**Files:**
- Create: `tests/test_stats.py`
- Create: `src/another_odd_thing/stats.py`
- Create: `experiments/run_v0.py`
- Create: `experiments/run_v1.py`
- Create: `tests/test_receipts.py`
- Create: `results/v0.json`
- Create: `results/v1.json`

**Interfaces:**
- Produces: `paired_bootstrap_mean_ci(differences, seed, resamples, confidence)` and JSON receipt schemas consumed by README/CI.

- [ ] **Step 1: Write failing tests** for deterministic bootstrap output and required receipt fields.
- [ ] **Step 2: Run the focused tests and verify failure.**
- [ ] **Step 3: Implement the statistics helper and CLIs.**
- [ ] **Step 4: Run canonical experiments and write frozen JSON receipts.**
- [ ] **Step 5: Add frozen receipt regression tests and verify the full suite.**

### Task 4: Research fence and browser lab

**Files:**
- Create: `docs/RELATED_WORK.md`
- Replace: `README.md`
- Create: `index.html`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Human-facing documentation and visualization only; no Python API changes.

- [ ] **Step 1: Document exact claims, destructive controls, and related work.**
- [ ] **Step 2: Add README result spine from the frozen receipts.**
- [ ] **Step 3: Add an interactive static lab showing passive overlap, addressed probe separation, one-bit publication, and v1 probe signatures.**
- [ ] **Step 4: Add packaging and CI that runs tests plus deterministic experiment smoke/canonical checks.**
- [ ] **Step 5: Run `python -m pip install -e '.[test]'`, `pytest -q`, both experiment CLIs, and a JavaScript syntax check extracted from `index.html`.**
