# Shared Compartment Trace v2 — Design

## Question

Does using one shared compartment state for both expression and plasticity remove a coordination burden that appears when those two functions are represented by independent variables?

This is a synthetic mechanism test, not a reproduction of Yang et al. or Onasch et al.

## Core object

Each compartment owns a two-timescale analog trace:

- `fast` follows local post-inhibition drive with a short time constant.
- `slow` follows `fast` with a longer time constant.
- `level = slow` is read through two thresholds, producing a three-regime plasticity sign: no change / depression / potentiation.
- `contrast = fast - slow` is recorded as a temporal derivative-like diagnostic; it is positive on rising activity and negative after drive removal.

The trace is indexed by compartment, not by synapse. All synapses on the same compartment therefore see the same local state.

## Expression/plasticity coupling

There are four contexts, four pathways and four compartments. Context `c` disinhibits compartment `c` for expression. During training, pathway `c` is active in context `c`.

### Coupled condition

The local analog drive that controls expression is also the drive that updates the compartment trace used for plasticity. There is only one context→compartment alignment to specify.

### Decoupled condition

Expression uses the same context→compartment mapping as the coupled condition, but plasticity has an independent context→compartment mapping. At mismatch probability `q`, each context's plasticity target is independently redirected to a wrong compartment. `q=0` is an oracle-coordinated decoupled control and is expected to match the coupled condition. Increasing `q` measures the coordination burden of keeping two mappings aligned.

## Plasticity

For the active pathway, every compartment reads its shared trace:

- `level < theta_d`: sign `0`
- `theta_d <= level < theta_p`: sign `-1`
- `level >= theta_p`: sign `+1`

The active pathway weight is updated by `eta * sign`.

After homosynaptic updates, a dendrite-specific heterosynaptic normalization restores a fixed total incoming excitatory-weight budget per compartment. This necessarily changes inactive synapses on a compartment when another synapse changes, explicitly testing the shared-local-bus idea.

## Metrics

Primary:

- **weight alignment accuracy**: fraction of pathways whose strongest learned weight lands on the compartment disinhibited for that pathway/context.
- **diagonal weight share**: fraction of total pathway weight that lies on the matched compartment.

Diagnostics:

- expression/plasticity mismatch rate,
- trace rise/fall contrast witness,
- inactive-synapse change under heterosynaptic normalization,
- weight budget error.

## Gates

1. **Trace witness** — a pulse makes `fast - slow > 0`; after removal it becomes `< 0`.
2. **Three-regime sign** — low / middle / high trace levels return 0 / -1 / +1.
3. **Shared bus** — potentiating one active synapse causes at least one inactive synapse on the same compartment to decrease through normalization while preserving the branch budget.
4. **Oracle decoupled attacker** — at `q=0`, decoupled must be allowed to match coupled. We do not count a coupled advantage here as necessary.
5. **Coordination sweep** — as mismatch rises, decoupled alignment should degrade while coupled is invariant because it has no second mapping to misalign.

## Classification

A PASS supports only this claim:

> In this toy local-learning system, sharing the compartment state used by expression and plasticity removes a context-to-compartment coordination requirement. An independently represented plasticity gate can match it when externally aligned, but becomes vulnerable when that alignment is corrupted.

It does not establish that calcium is the biological mechanism responsible for this effect, nor that shared traces universally improve learning.
