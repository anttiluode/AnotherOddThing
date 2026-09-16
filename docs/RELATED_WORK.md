# Related work and novelty fence

AnotherOddThing sits at an intersection that is already well populated: dendritic nonlinear computation, pathway-specific inhibitory gating, event-triggered communication, and active experiment design/system identification. The repo is not claiming invention of any of those components.

## Dendritic pathway gating

Yang, Murray & Wang (2016), **“A dendritic disinhibitory circuit mechanism for pathway-specific gating”**, *Nature Communications* 7:12815, DOI `10.1038/ncomms12815`, propose a model in which inputs from different pathways cluster on different dendritic territories and branch-specific disinhibition selectively opens one pathway. Their reduced pyramidal model treats dendritic compartments as quasi-independent nonlinear processors capable of NMDA plateaus, while SOM/VIP circuitry supplies context-dependent control.

**What this motivates here:** an addressed perturbation can matter because it lands on a particular local computational territory rather than on a featureless scalar neuron.

**What it does not show:** that a dendritic branch stores an arbitrary algorithm, that spikes carry software addresses, or that the brain performs the Bayesian identification algorithm in this repo.

## Contextual dendritic gating of plasticity

Onasch, Miehl, Miekus & Gjorgjieva (2026), **“Assembly-based computations through contextual dendritic gating of plasticity”**, *Neuron*, DOI `10.1016/j.neuron.2026.07.028`, build a multi-area spiking model with nonlinear dendritic branches and context-specific inhibitory gating of branch-level plasticity. In their model, selected branches can be opened for plasticity while others remain protected, allowing overlapping assemblies to coexist and participate in projections/associations across areas.

**What this motivates here:** local branch state and local branch plasticity can be separately gated by context; rich computation need not be globally rewritten whenever current relevance changes.

**What it does not show:** the resident-operator probe hypothesis. Their contextual signal gates learning/expression; AnotherOddThing asks whether a deliberately chosen perturbation can identify a hidden local operator.

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
- `Genealogy`: exposed `persistent-state`, `structure-as-computation`, and `active-intervention` as independently recurring motifs.

AnotherOddThing deliberately strips those biological stories down to an identifiability test before recombining them.

## Novelty standard

A defensible eventual claim is **not**:

- dendrites compute nonlinearly;
- inhibition can gate dendritic pathways;
- context can gate plasticity;
- informative excitation improves system identification;
- sparse events can communicate maintained state;
- modular dynamical systems can be probed.

The potentially distinctive object is narrower:

> **A message can be computationally useful without carrying the computation: it can instead act as an addressed intervention whose meaning is supplied by the resident operator it perturbs.**

v0 is only an existence proof for that sentence. v1 asks whether active address choice remains useful through a noisy one-bit response channel. Later gates must attack the construction rather than broaden the prose.
