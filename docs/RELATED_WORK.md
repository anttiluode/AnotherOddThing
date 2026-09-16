# Related work and novelty fence

AnotherOddThing sits at an intersection that is already well populated: dendritic nonlinear computation, pathway-specific inhibitory gating, compartment-local plasticity, event-triggered communication, and active experiment design/system identification. The repo is not claiming invention of any of those components.

## Dendritic pathway gating — expression

Yang, Murray & Wang (2016), **“A dendritic disinhibitory circuit mechanism for pathway-specific gating”**, *Nature Communications* 7:12815, DOI `10.1038/ncomms12815`, propose a model in which inputs from different pathways cluster on different dendritic territories and branch-specific disinhibition selectively opens one pathway. Their reduced pyramidal model treats dendritic compartments as quasi-independent nonlinear processors capable of NMDA plateaus, while SOM/VIP circuitry supplies context-dependent control.

The relevant nonlinearity is not merely an abstract gate. Their thin-branch model produces a sigmoidal input-rate/voltage relation through regenerative NMDA plateaus. Moderate GABAergic inhibition can prevent the plateau and linearize the branch response. In their supplementary targeting rule they also impose a conductance cutoff so excitation is assigned only to sufficiently disinhibited dendrites; that cutoff is a **model construction**, not an experimentally measured synaptic transfer law.

**What this motivates here:** an addressed event can matter because it lands on a local computational territory whose operating regime depends continuously on inhibition rather than on a clean Boolean mask.

**What it does not show:** that a dendritic branch stores an arbitrary algorithm, that spikes carry software addresses, or that the brain performs the Bayesian identification algorithm in this repo.

## Contextual dendritic gating — plasticity

Onasch, Miehl, Miekus & Gjorgjieva (2026), **“Assembly-based computations through contextual dendritic gating of plasticity”**, *Neuron*, DOI `10.1016/j.neuron.2026.07.028`, build a multi-area spiking model with nonlinear dendritic branches and context-specific inhibitory gating of branch-level plasticity. In their voltage rule, dendritic depolarization strongly determines plasticity direction; the reported dendritic voltage filters include `tau_- = 15 ms` and `tau_+ = 45 ms` together with distinct voltage thresholds. Their model also applies a dendrite-specific heterosynaptic normalization: when some synapses on a branch potentiate, other synapses sharing that branch can be depressed to keep total branch input near a target budget.

That shared-compartment organization is important for v2. The local variable is not naturally indexed only by the individual weight; a branch-level state can couple synapses that never directly communicate. Onasch et al. also identify a strong architectural assumption: their one-to-one context→inhibitory-population→dendrite mapping protects learned structures, whereas randomizing that mapping removes the protection across contexts.

**What this motivates here:** expression and plasticity may share a physically aligned local coordinate system, and branch-local normalization can couple weights through shared compartment state.

**What it does not show:** that literally sharing one scalar state variable is always superior. v2 therefore contains a `decoupled_mirrored` attacker in which a separate trace receives exactly the same local signal and is expected to match the coupled machine.

## Yang-style calcium thresholds versus Onasch-style voltage timescales

The two papers should not be collapsed into one biological rule.

Yang et al. use a calcium-based plasticity formalism in which time above a lower depression threshold and a higher potentiation threshold contributes to LTD and LTP; synapses are modeled as probabilistically bistable DOWN/UP states. Onasch et al. use a Clopath-style voltage rule with distinct filtered dendritic-voltage variables and different time constants.

AnotherOddThing v2 intentionally builds a **minimal hybrid abstraction** rather than claiming either paper's exact rule:

```text
shared local drive
      ↓
fast leaky trace (15 ms)
      ↓
slow leaky trace (45 ms)

slow level -> two thresholds -> 0 / LTD / LTP
fast - slow -> rise/fall diagnostic
```

The 15/45 ms pair is used as a computational probe of a two-timescale local state. The three-regime sign readout is calcium-like. This is a synthetic test of what the shape can buy, not a biological fit.

## v2 coordination claim

v2 compares three systems under the same stimuli, weights, learning rate, trace constants, thresholds, branch budget and training schedule:

- `coupled`: expression and plasticity share one compartment trace;
- `decoupled_mirrored`: separate expression/plasticity traces receive the same local signal;
- `decoupled_routed`: plasticity has an independent context→compartment route that can be corrupted.

The crucial negative control is `decoupled_mirrored`: it matches the coupled system when local information is identical. Therefore the frozen result does **not** support “one state variable is inherently more powerful than two.” It supports the narrower structural statement that independent routing adds a coordination degree of freedom. Corrupting that routing degrades excitation/disinhibition alignment; sharing the local coordinate removes that particular failure mode by construction.

The next attacker is to replace hard route corruption with continuous inhibitory offsets and ask whether the effect survives when misalignment is soft rather than categorical.

## Entorhinal sweeps — moving resident state

Vollan, Gardner, Moser & Moser (2025), **“Left–right-alternating theta sweeps in entorhinal–hippocampal maps of space”**, *Nature* 639:995–1004, DOI `10.1038/s41586-024-08527-1`, report theta-paced trajectories through grid/place representations that extend into never-visited locations. They identify an alternating internal-direction signal aligned with the sweeps and discuss a circuit in which direction-related input shifts activity across the resident grid-cell manifold. Their artificial agent reproduces alternation by choosing sweep directions that minimize overlap with a decaying coverage trace.

**What this motivates here:** a small control signal may move activity through an already resident representation rather than carrying the rich representation itself.

**What it does not show:** Bayesian information-gain probing or the shared-trace plasticity mechanism in v2.

## Experiment design and system identification

The central mathematical neighbor is optimal experiment design for dynamical-system identification. Gevers, Bombois, Hildebrand & Solari (2011), **“Optimal experiment design for open and closed-loop system identification”**, *Communications in Information and Systems* 11(3):197–224, DOI `10.4310/CIS.2011.v11.n3.a1`, review the long-standing idea that the excitation applied to a system determines how informative the resulting data can be.

Rajendran, Reizinger, Brendel & Ravikumar (2024), **“An Interventional Perspective on Identifiability in Gaussian LTI Systems with Independent Component Analysis”**, PMLR 236:41–70, explicitly connect intervention design and identifiability in Gaussian LTI systems under stated assumptions.

AnotherOddThing v1 is therefore not novel because it chooses informative probes. Its contribution, if useful, is the very small synthetic composition being tested:

```text
resident operator + local state
        ↓
small addressed perturbation
        ↓
operator-dependent response
        ↓
one-bit publication
        ↓
next probe chosen from uncertainty
```

## Event-triggered communication

ActiveVectorNN and FusionMachine already fenced this part internally: maintained state need not be continuously retransmitted. Delta/event-triggered methods in control and neural computation are mature prior art. Here the event is intentionally impoverished because the experimental question is whether information can live in the **response of the resident operator**, not in a rich transmitted payload.

## Internal lineage

- `AnttisNeuron`: physical/branched operators, modal filtering, nonlinear subunits, and negative generalization results.
- `GrowingAnttisNeuron`: development chooses input addresses and can associate addresses with genuinely different physical operators.
- `ActiveVectorNN`: separates resident state from communication events.
- `FusionMachine`: separates resident state, publication, route identity, and downstream local writes.
- `Genealogy`: exposed `persistent-state`, `structure-as-computation`, `active-intervention`, and now statistical evidence as independently recurring motifs.
- `AnotherOddThing v2`: inserts a compartment-shared analog state between connection and branch, then attacks whether using it for both expression and plasticity removes an alignment requirement.

## Novelty standard

A defensible eventual claim is **not**:

- dendrites compute nonlinearly;
- inhibition can gate dendritic pathways;
- context can gate plasticity;
- calcium or voltage traces can threshold plasticity;
- heterosynaptic normalization exists;
- informative excitation improves system identification;
- sparse events can communicate maintained state;
- modular dynamical systems can be probed.

The potentially distinctive composition is narrower:

> **A message can be computationally useful without carrying the computation: it can act as an addressed intervention whose local meaning is supplied by shared compartment state and resident dynamics; the same local state can also constrain how that territory is rewritten.**

v0 is only an existence proof for the resident-operator sentence. v1 tests active address choice through a noisy one-bit response. v2 tests the additional coordination consequence of letting expression and plasticity share a local coordinate. Later gates must attack these constructions rather than broaden the prose.
