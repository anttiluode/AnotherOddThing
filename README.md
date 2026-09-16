# AnotherOddThing

> **What if a message does not carry the computation, but only perturbs the computation already living at its destination?**

AnotherOddThing is a small systems-identification and local-learning research repo spun out of `AnttisNeuron`, `GrowingAnttisNeuron`, `ActiveVectorNN`, `FusionMachine`, and the cross-repo `Genealogy` audit.

The biological vocabulary is motivational. The experiments are synthetic mechanism tests with explicit controls.

**Live lab:** `https://anttiluode.github.io/AnotherOddThing/`

## The object

A node owns a resident operator `A` and state `x`. A tiny addressed event does not describe `A`; it perturbs one local coordinate and lets `A` determine what happens next:

```text
x+      = x + b_k u          # addressed write / poke
x_next  = A x+               # resident computation acts
y_next  = C x_next           # narrow observation
bit     = 1[y_next > theta]  # optional publication boundary
```

The first question is deliberately brutal:

> **Can two resident operators be perfectly indistinguishable under passive observation, yet become distinguishable after the same known local perturbation?**

## v0 — same visible state, different operator

Two stable operators share the same visible dynamics:

```text
A0 = [[0.85, 0.00],
      [0.00, 0.55]]

A1 = [[0.85, 0.65],
      [0.00, 0.55]]

x0 = [1, 0]
C  = [1, 0]
```

Because hidden state starts at zero, four passive steps are **exactly identical** at the readout. Probing the visible coordinate also fails to distinguish them because their first columns match. But an equal pulse into hidden coordinate 1 exposes the coupling that passive observation never excited.

Frozen receipt:

| condition | O0 | O1 | absolute difference |
|---|---:|---:|---:|
| passive trace max difference | — | — | **0.000000** |
| visible-site probe | 1.2937053125 | 1.2937053125 | **0.000000** |
| hidden-site probe | 0.4437053125 | 1.0937053125 | **0.650000** |
| one-bit publication at `theta=0.75` | 0 | 1 | separated |

**Classification: PASS.** This is an existence proof only: passive equality does not imply equal resident operators when the passive trajectory never excites the direction on which they differ.

See [`results/v0.json`](results/v0.json).

## v1 — choose the poke, keep the answer one bit wide

v1 turns the witness into a tiny active-identification problem. Four resident operators have three addressable hidden probe sites. Low/high hidden-to-visible gains encode this response family:

```text
operator  site0 site1 site2
O0          0     0     0
O1          1     0     0
O2          0     1     0
O3          0     1     1
```

A probe produces a noisy analog response, but the observer receives only:

```text
bit = 1[response > 0.8]
```

The active policy maintains a posterior over operators and chooses the unused site with maximum expected information gain. The random control chooses two distinct sites uniformly without replacement. Both receive exactly **two probes**, the same true operator, and the same per-round Gaussian noise tape.

### Frozen 4,096-trial result

| metric | active information-gain probes | random distinct probes |
|---|---:|---:|
| identification accuracy | **0.915283** | 0.695312 |
| mean final posterior entropy (bits) | **0.517595** | 0.855408 |
| mean probe count | 2.000 | 2.000 |

Paired accuracy delta: **+0.219971**.

Deterministic paired-bootstrap 95% interval: **[+0.205560, +0.234375]**.

Discordant paired trials: active-only correct **986**, random-only correct **85**.

The active policy chose site 1 first on all 4,096 trials, then chose site 0 or site 2 according to the first event. That behavior is not hard-coded: it falls out of expected posterior entropy.

**Classification: PASS for this fixed operator family.** It does **not** mean active probing is universally better; it means the predeclared family contains a real matched-budget experiment-design advantage.

See [`results/v1.json`](results/v1.json).

## v2 — one compartment trace, two jobs

v2 asks whether we misplaced a computational layer between the synapse and the branch.

Each compartment owns a shared two-timescale analog state:

```text
fast <- fast + dt/tau_fast * (local_drive - fast)
slow <- slow + dt/tau_slow * (fast - slow)

level    = slow
contrast = fast - slow
```

The default constants are `tau_fast = 15 ms` and `tau_slow = 45 ms`. They are inspired by the two dendritic voltage filters in Onasch et al.; this toy is **not** their plasticity rule. The slow level is read with two thresholds:

```text
level < theta_d                 ->  0   no update
theta_d <= level < theta_p      -> -1   depression
level >= theta_p                -> +1   potentiation
```

The same trace is indexed by **compartment**, not by weight. After a homosynaptic update, a fixed per-compartment incoming-weight budget is restored, so inactive synapses on the same compartment move too.

The main attacker is not a deliberately weak baseline. We compare three machines:

1. `coupled` — expression and plasticity literally use the same compartment trace;
2. `decoupled_mirrored` — a separate plasticity trace receives the same local signal;
3. `decoupled_routed` — plasticity has its own context→compartment routing, which can be independently corrupted.

The mirrored control matters: **a separate state variable fed the same local information matches the coupled machine exactly.** The effect is therefore not “shared Python object good.” It is the removal of a second routing/alignment degree of freedom.

### Frozen 128-seed coordination sweep

| independent plasticity-routing mismatch | coupled alignment | mirrored independent trace | separately routed plasticity |
|---:|---:|---:|---:|
| 0.00 | **1.000000** | 1.000000 | 1.000000 |
| 0.25 | **1.000000** | 1.000000 | 0.734375 |
| 0.50 | **1.000000** | 1.000000 | 0.472656 |
| 0.75 | **1.000000** | 1.000000 | 0.220703 |
| 1.00 | **1.000000** | 1.000000 | 0.000000 |

At 50% routing corruption, the paired coupled-minus-routed alignment advantage is **+0.527344**, with deterministic paired-bootstrap 95% interval **[+0.484375, +0.572266]**.

Two small mechanism witnesses are frozen too:

- after a pulse, `fast - slow = +0.540638`; after six decay steps it is `-0.226494`, so the same local state carries a rise/fall contrast;
- after one synapse is raised from `1.0` to `1.4`, branch-budget normalization moves the three inactive neighbours from `1.0` to `0.909091` while restoring the total branch budget to `4.0`.

**Classification: PASS for a coordination claim only.** A perfectly co-aligned decoupled machine matches. v2 says that using one local signal for expression and plasticity can remove an alignment requirement that an independently routed plasticity channel has to satisfy; it does not establish that shared traces universally improve learning or reproduce calcium biology.

See [`results/v2.json`](results/v2.json).

## What this changes in the old neuron story

The useful abstraction is no longer just “resident operator + addressed event.” There may be a local transducer/state between the connection and the branch:

```text
small event + local context
          ↓
shared compartment state
   ↙               ↘
expression         plasticity
   ↓                  ↓
resident dynamics   future transfer
```

That gives a cleaner interpretation of several old intuitions:

- the rich thing can stay resident;
- the pulse can remain tiny;
- **where** it lands matters;
- local state can change what the event means;
- the same physical state can constrain both what is expressed now and what is allowed to change for later;
- if a response guides the next perturbation, the push has become a probe.

## Biology fence

Three distinct biological ideas motivate different parts of this repo:

- Yang, Murray & Wang (2016), *A dendritic disinhibitory circuit mechanism for pathway-specific gating*, DOI `10.1038/ncomms12815` — NMDA dendritic nonlinearity plus local inhibition supports pathway-specific expression gating; their calcium-based plasticity model also uses depression and potentiation thresholds.
- Onasch et al. (2026), *Assembly-based computations through contextual dendritic gating of plasticity*, DOI `10.1016/j.neuron.2026.07.028` — dendritic depolarization with distinct 15 ms and 45 ms filters controls plasticity; dendrite-specific heterosynaptic normalization couples synapses sharing a branch; the authors explicitly report that randomized context→inhibition→dendrite mapping loses cross-context protection.
- Vollan et al. (2025), *Left–right-alternating theta sweeps in entorhinal–hippocampal maps of space*, DOI `10.1038/s41586-024-08527-1` — small internal-direction signals are associated with trajectories through an already resident spatial representation, motivating the separate “control vector moves resident state” thread.

These papers do **not** establish the synthetic operator-probing or v2 coordination claims. See [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).

## Next attacks

The next useful gates are destructive rather than decorative:

1. **State back-action.** Join v1 to actual resident dynamics so probe 1 changes the state encountered by probe 2.
2. **Trace attacker.** Replace the two-timescale trace with a single leaky scalar and quantify what, if anything, the fast/slow pair buys beyond the coordination effect.
3. **Soft routing noise.** Replace hard context-map corruption with continuous inhibitory offsets and ask whether alignment fails gradually rather than categorically.
4. **Hidden state versus hidden operator.** Make `same A / different x` and `different A / same visible x` compete under the same one-bit probe budget.
5. **Grow the operator.** Replace hand-coded structure with operators produced by local temporal statistics, reconnecting to `GrowingAnttisNeuron`.

## Run

```bash
python -m pip install -e ".[test]"
pytest -q
python -m experiments.run_v0 --out /tmp/v0.json
python -m experiments.run_v1 --out /tmp/v1.json
python -m experiments.run_v2 --out /tmp/v2.json
```

## Repository map

- `src/another_odd_thing/core.py` — resident dynamics and v0 causal witness.
- `src/another_odd_thing/identify.py` — one-bit likelihood model, Bayesian update, information gain, paired policies.
- `src/another_odd_thing/trace.py` — compartment-shared two-timescale analog state and threshold sign readout.
- `src/another_odd_thing/alignment.py` — v2 local-learning world, mirrored attacker, independently routed attacker, and heterosynaptic normalization.
- `src/another_odd_thing/stats.py` — deterministic paired bootstrap interval.
- `experiments/run_v0.py`, `run_v1.py`, `run_v2.py` — deterministic scientific receipts.
- `results/v0.json`, `results/v1.json`, `results/v2.json` — frozen canonical outputs.
- `tests/` — mechanism, destructive controls, policy behavior, and receipt regressions.
- `index.html` — dependency-free browser microscope.
- `docs/RELATED_WORK.md` — prior-art and biology fence.
- `docs/superpowers/` — frozen design and implementation plan.
