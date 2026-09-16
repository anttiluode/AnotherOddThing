# Resident Operator Probe Design

## Purpose

AnotherOddThing tests one narrow idea: a small message can be useful because it perturbs rich computation already resident at its destination. The first experiments deliberately avoid claiming that biological dendrites are arbitrary programs. They ask a systems question instead:

> Can two systems with the same visible state and the same passive observations be distinguished by a known addressed perturbation because their resident operators differ?

The repo starts from the collision of three Genealogy motifs: persistent state, structure-as-computation, and active intervention.

## Claim boundary

The v0/v1 program may establish synthetic identifiability and active experiment-design results. It does not establish that real dendrites implement the exact operators, that spikes are software instructions, or that interneuron circuits perform Bayesian system identification.

The biological papers motivate local nonlinear processing and branch/pathway-specific gating only. Yang, Murray & Wang (2016, Nature Communications, DOI 10.1038/ncomms12815) model branch-specific disinhibition as a mechanism for pathway-specific gating. Onasch et al. (2026, Neuron, DOI 10.1016/j.neuron.2026.07.028) model nonlinear dendrites plus context-dependent inhibitory gating of branch-specific plasticity, allowing overlapping assemblies to be learned and combined. Neither paper tests the operator-probe hypothesis here.

## Core model

A resident node owns an operator matrix `A`, a state vector `x`, and a scalar readout `y = Cx`. A probe is an addressed unit vector `b_k` with amplitude `u` that is written locally before the resident dynamics act:

```text
x+      = x + b_k u
x_next  = A x+
y_next  = C x_next
```

A publication boundary can reduce the analog response to one bit:

```text
e = 1[y_next > threshold]
```

The message therefore does not encode `A`. Its effect depends on the operator already present at the destination and on the addressed site.

## Gate v0: passive equivalence, interventional separation

Use two stable 2x2 operators with a shared visible subspace:

```text
A0 = [[0.85, 0.00],
      [0.00, 0.55]]

A1 = [[0.85, 0.65],
      [0.00, 0.55]]

x0 = [1, 0]
C  = [1, 0]
```

With no hidden-coordinate excitation, both systems produce exactly the same passive visible trajectory because the second state remains zero. A probe written into hidden coordinate 1 creates a response difference on the next step because only `A1` couples that hidden coordinate into the visible coordinate.

Required controls:

- no probe: traces remain exactly equal;
- probe the visible coordinate instead: responses remain equal because the first columns match;
- probe the hidden coordinate: responses separate;
- one-bit threshold: choose a predeclared threshold between the two hidden-probe responses so the event bit identifies the operator in this noiseless witness.

The v0 result is an existence proof, not a broad performance claim.

## Gate v1: active probe choice under a one-bit channel

Generalize to four 4x4 stable resident operators. Coordinate 0 is visible; coordinates 1-3 are addressable hidden probe sites. The operators share all dynamics except hidden-to-visible couplings. Their three probe-response signatures are:

```text
operator  site0 site1 site2
O0          0     0     0
O1          1     0     0
O2          0     1     0
O3          0     1     1
```

`0` and `1` are implemented as low/high hidden-to-visible gains, not as direct labels. The analog response is corrupted by Gaussian noise and thresholded into one event bit. A Bayesian observer maintains a posterior over the four resident operators.

Compare two policies at the same budget of two distinct probes:

- `active`: choose the unused probe site with maximum expected information gain;
- `random`: choose two sites uniformly without replacement.

Use paired trials: both policies see the same true operator and the same per-round Gaussian noise tape. Random policy site order comes from a separately seeded tape. Report identification accuracy, final posterior entropy, the paired accuracy delta, and a deterministic paired bootstrap 95% interval for that delta.

The active policy must not be hard-coded to the answer. It computes expected posterior entropy from the likelihood model at each round.

## Frozen parameters

- v0 passive steps: 4
- v0 hidden probe amplitude: 1.0
- v0 publication threshold: 0.75
- v1 low gain: 0.20
- v1 high gain: 1.40
- v1 publication threshold: 0.80
- v1 observation noise sigma: 0.35
- v1 probe budget: 2
- v1 canonical trials: 4096
- v1 trial seed: 20260916
- v1 bootstrap seed: 9162026
- v1 bootstrap resamples: 4000

## Repository shape

- `src/another_odd_thing/core.py`: resident operator dynamics and v0 witness helpers.
- `src/another_odd_thing/identify.py`: v1 operator family, Bayesian update, information gain, policies, and paired trial runner.
- `src/another_odd_thing/stats.py`: deterministic paired bootstrap interval.
- `experiments/run_v0.py`, `run_v1.py`: deterministic receipts.
- `results/v0.json`, `results/v1.json`: frozen canonical outputs.
- `tests/`: mechanism, controls, policy behavior, and frozen receipt checks.
- `index.html`: dependency-free visual lab reproducing the qualitative v0/v1 mechanisms in the browser.
- `docs/RELATED_WORK.md`: novelty fence and links to the biological and system-identification neighborhoods.

## Success and failure interpretation

A v0 pass means only that operator identity can be hidden from passive observation yet exposed by a targeted perturbation in this construction.

A v1 pass means only that information-gain-chosen addresses beat an equal-budget random address policy in the predeclared noisy operator family. It does not imply active probing is always superior. A failure is retained and narrows the active-intervention motif.

The project should prefer destructive controls and exact claim boundaries over biological storytelling.
