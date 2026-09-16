# AnotherOddThing

> **What if a message does not carry the computation, but only perturbs the computation already living at its destination?**

AnotherOddThing is a small systems-identification research repo spun out of `AnttisNeuron`, `GrowingAnttisNeuron`, `ActiveVectorNN`, `FusionMachine`, and the cross-repo `Genealogy` audit.

The biological vocabulary is motivational. The experiments are synthetic linear dynamical systems with explicit controls.

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

## What this changes in the old neuron story

The useful abstraction is not “a dendrite contains an arbitrary algorithm.” It is:

```text
structure/operator + resident state + addressed perturbation
                         ↓
                 state-dependent response
                         ↓
                    small event
```

That gives a cleaner interpretation of the FusionMachine intuition:

- the rich thing stays resident;
- the pulse can be tiny;
- **where** it lands matters;
- the response can reveal something about the operator that passive watching could not;
- if the response guides the next perturbation, the push has become a probe.

## Biology fence

Two papers are especially relevant as motivation, not validation:

- Yang, Murray & Wang (2016), *A dendritic disinhibitory circuit mechanism for pathway-specific gating*, DOI `10.1038/ncomms12815` — branch-specific disinhibition can gate pathway-specific dendritic input in a network model.
- Onasch et al. (2026), *Assembly-based computations through contextual dendritic gating of plasticity*, DOI `10.1016/j.neuron.2026.07.028` — nonlinear dendrites plus branch-specific inhibitory context can gate plasticity and support overlapping assemblies in a spiking model.

Neither paper says dendrites are arbitrary programs or that spikes implement the Bayesian probing scheme here. See [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).

## Next attacks

The next useful gates are destructive rather than decorative:

1. **Hidden state versus hidden operator.** Make `same A / different x` and `different A / same visible x` compete under the same one-bit probe budget.
2. **Route destruction.** Remove or corrupt probe-site identity and measure how much identifiability was actually carried by address.
3. **Probe cost / dense causes.** Charge for interventions and add multiple simultaneous hidden differences; find where active probing stops paying.
4. **Back-action.** Let probing slowly modify the operator, turning identification into measurement-plus-write.
5. **Grow the operator.** Replace the hand-coded gain table with operators produced by local temporal statistics, reconnecting to `GrowingAnttisNeuron`.

## Run

```bash
python -m pip install -e ".[test]"
pytest -q
python -m experiments.run_v0 --out /tmp/v0.json
python -m experiments.run_v1 --out /tmp/v1.json
```

## Repository map

- `src/another_odd_thing/core.py` — resident dynamics and v0 causal witness.
- `src/another_odd_thing/identify.py` — one-bit likelihood model, Bayesian update, information gain, paired policies.
- `src/another_odd_thing/stats.py` — deterministic paired bootstrap interval.
- `experiments/run_v0.py`, `run_v1.py` — deterministic scientific receipts.
- `results/v0.json`, `results/v1.json` — frozen canonical outputs.
- `tests/` — mechanism, destructive controls, policy behavior, and receipt regressions.
- `index.html` — dependency-free browser microscope.
- `docs/RELATED_WORK.md` — prior-art and biology fence.
- `docs/superpowers/` — frozen design and implementation plan.
