# v3 — continuous inhibitory set-point attack

v2 showed a coordination effect: expression and plasticity can be represented by separate state variables with no penalty **if they receive the same local signal**, but an independently routed plasticity channel becomes vulnerable when its context→compartment mapping is corrupted.

v3 removes the easy part of that attack. There is no longer a binary switch from a correct compartment to a wrong compartment. Instead, inhibitory relief is transferred continuously between the two:

```text
relief = I_closed - I_open

I_correct(q) = I_closed - (1-q) * relief
I_wrong(q)   = I_closed - q * relief
```

At `q=0`, only the expression-aligned compartment is fully disinhibited. At `q=0.5`, correct and wrong compartments receive the same intermediate inhibition. At `q=1`, the wrong compartment receives the full relief.

Expression keeps the correct `q=0` context signal. Only the independently represented plasticity set-point is displaced.

## The destructive control

The quantized trace readout is the v2 rule:

```text
level < theta_d             ->  0
theta_d <= level < theta_p  -> -1
level >= theta_p            -> +1
```

The stricter continuous-magnitude attacker now preserves the **same exact no-update floor** below `theta_d`, so it does not gain an extra weak-activity LTD mechanism. It differs only inside the active plasticity range:

```text
level < theta_d             ->  0
update(theta_d)             -> -1
update((theta_d+theta_p)/2) ->  0
update(theta_p)             -> +1
level > theta_p             -> +1
```

Between `theta_d` and `theta_p`, magnitude is linearly interpolated. This is intentionally **not** presented as a biological alternative rule. Its job is to attack a simpler explanation: perhaps the apparent set-point knee is produced only by argmax scoring, heterosynaptic normalization, recurrent weight feedback, or an unfair subthreshold depression advantage, and the categorical local readout contributes nothing.

## Frozen 128-seed sweep

Alignment accuracy:

| q | quantized | continuous magnitude |
|---:|---:|---:|
| 0.0 | 1.000000 | 1.000000 |
| 0.1 | 1.000000 | 1.000000 |
| 0.2 | 1.000000 | 1.000000 |
| 0.3 | 0.947266 | 1.000000 |
| 0.4 | 0.648438 | 0.980469 |
| 0.5 | 0.228516 | 0.666016 |
| 0.6 | 0.064453 | 0.394531 |
| 0.7 | 0.013672 | 0.222656 |
| 0.8 | 0.001953 | 0.169922 |
| 0.9 | 0.000000 | 0.046875 |
| 1.0 | 0.000000 | 0.001953 |

The same separation is visible before the winner-take-all alignment score in the continuous diagonal-weight-share statistic:

| q | quantized weight share | continuous weight share |
|---:|---:|---:|
| 0.0 | 0.548795 | 0.456368 |
| 0.1 | 0.556228 | 0.463647 |
| 0.2 | 0.552588 | 0.471072 |
| 0.3 | 0.507615 | 0.471798 |
| 0.4 | 0.374224 | 0.440173 |
| 0.5 | 0.256044 | 0.398858 |
| 0.6 | 0.175454 | 0.326785 |
| 0.7 | 0.140373 | 0.244784 |
| 0.8 | 0.134531 | 0.213327 |
| 0.9 | 0.141450 | 0.190784 |
| 1.0 | 0.151139 | 0.184199 |

At the predeclared midpoint `q=0.5`:

- continuous-minus-quantized alignment delta: **+0.437500**;
- deterministic paired-bootstrap 95% interval: **[+0.386719, +0.488281]**;
- continuous-minus-quantized diagonal-weight-share delta: **+0.14281462**;
- deterministic paired-bootstrap 95% interval: **[+0.13406188, +0.15120240]**.

On the frozen 0.1-spaced grid, the largest adjacent drop in mean diagonal weight share occurs from `q=0.3 -> 0.4` for the quantized rule (`0.13339`) and from `q=0.6 -> 0.7` for the stricter continuous control (`0.08200`). These are descriptive grid summaries, not estimates of a biological critical point.

## Interpretation

The strong version of the old intuition does **not** survive:

> continuous inhibition by itself does not guarantee a categorical learning transition.

The narrower construction survives a more demanding attacker:

> with the same continuously moved inhibitory set-point and the same subthreshold no-update region, making the local plasticity readout categorical causes the learned structure to lose alignment substantially earlier and more sharply than a continuously varying LTD-to-LTP magnitude in this toy world.

That makes the emerging computational object more precise:

```text
continuous local chemistry / voltage-like state
                    ↓
         compartment thresholds
                    ↓
     categorical update regime
                    ↓
      branch-wide weight change
                    ↓
        future expression changes
```

The thresholded local state is therefore not merely a gate in this construction. It is a **regime selector** for how a whole compartment changes.

## What v3 does not establish

v3 does not show that real calcium follows this toy rule, that the thresholds used here are biologically correct or optimal, that neural tissue sits at a critical point, or that quantized plasticity is universally preferable. The continuous control is deliberately synthetic, and the test was designed as a destructive follow-up after v2 rather than as a preregistered biological model comparison.

The frozen machine-readable receipt is [`../results/v3.json`](../results/v3.json), generated by [`../experiments/run_v3.py`](../experiments/run_v3.py).
