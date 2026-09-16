# v4 — matched single-trace attacker

v3 found a sharp structural knee when a continuously displaced inhibitory set-point was read through the categorical `0 / LTD / LTP` plasticity rule. The obvious next attack is whether that result actually needs the two-state `15 ms / 45 ms` compartment trace used since v2.

v4 replaces **only the plasticity trace** with one leaky scalar. Expression keeps the original two-timescale trace. The routing, continuous inhibitory displacement, thresholds, quantized update rule, presentation schedule, learning rate and heterosynaptic branch budget all stay fixed.

## Matched control

A weak scalar attacker would be easy to beat simply because it reaches a different plasticity level after a burst. v4 therefore solves its time constant analytically so that, starting from rest under the canonical six-step unit drive, it reaches exactly the same decision coordinate as the original cascaded trace.

```text
two-state level after six active steps = 0.3715702527028714
single level after six active steps    = 0.37157025270287153
absolute difference                   ≈ 1.1e-16
matched single tau                    = 67.1135 ms
```

The histories are not equivalent. After the same six decay steps:

```text
two-state level  = 0.30657822991715195
single level     = 0.23350580000919569
single - two     = -0.07307242990795626
```

So this is a genuine temporal-history attacker, not an exact state reparameterization.

## Frozen 128-seed sweep

Both models use the quantized plasticity rule. Alignment accuracy:

| set-point displacement q | two-timescale | matched single |
|---:|---:|---:|
| 0.0 | 1.000000 | 1.000000 |
| 0.1 | 1.000000 | 1.000000 |
| 0.2 | 1.000000 | 1.000000 |
| 0.3 | 0.947266 | 0.984375 |
| 0.4 | 0.648438 | 0.726562 |
| 0.5 | 0.228516 | 0.316406 |
| 0.6 | 0.064453 | 0.058594 |
| 0.7 | 0.013672 | 0.015625 |
| 0.8 | 0.001953 | 0.000000 |
| 0.9 | 0.000000 | 0.000000 |
| 1.0 | 0.000000 | 0.000000 |

The matched single trace does **not** erase the set-point knee. Both traces first fall below 50% alignment in the same frozen grid bin, `q = 0.5`. Their largest adjacent diagonal-weight-share drop also lands in the same `q = 0.3 -> 0.4` bin.

At `q = 0.5`, the single trace is actually modestly better:

- single-minus-two alignment delta: **+0.087890625**;
- paired-bootstrap 95% interval: **[+0.048828125, +0.125]**;
- single-minus-two diagonal-weight-share delta: **+0.02251477**;
- paired-bootstrap 95% interval: **[+0.01382173, +0.03124608]**.

The largest absolute accuracy difference anywhere on the sweep is **0.087890625**. The largest absolute diagonal-weight-share difference is **0.05673332**.

## Classification

**`SINGLE_TRACE_SUFFICIENT` for this level-only set-point task.**

That is a correction to the richer interpretation of v2. The current v3 knee does not independently require a 15/45 ms fast/slow cascade. Once the scalar control is calibrated to the same six-step decision level, a one-state leaky trace retains the same qualitative robust-region-to-failure transition.

What survives is smaller and cleaner:

```text
continuous local drive / inhibition
             ↓
      local scalar state
             ↓
   categorical thresholds
             ↓
 branch-wide normalized update
             ↓
    future transfer geometry
```

The important object for **this** experiment is the thresholded local level and its coupling to branch-wide change, not the existence of two temporal filters.

## What is not killed

The original two-timescale trace also exposes `fast - slow`, which changes sign between rise and decay. v4 never gives the learning rule a task that needs that contrast. A one-state scalar can therefore win this attack while still being incapable of representing two histories that have the same current level but opposite temporal direction.

The next honest test is consequently different: construct matched histories with the same plasticity level but opposite rise/fall contrast, require different updates, and ask whether the second state then earns its cost.

This is a synthetic mechanism result, not evidence that biological dendrites can be reduced to one scalar state. It only says that the richer trace is unnecessary for the present level-threshold phenomenon.

Machine-readable source: [`../results/v4.json`](../results/v4.json). Reproduce with:

```bash
python -m experiments.run_v4 --out /tmp/v4.json
cmp /tmp/v4.json results/v4.json
```
