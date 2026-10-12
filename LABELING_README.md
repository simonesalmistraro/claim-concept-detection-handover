# Labeling guide — claim concept presence

Fill `labeling_sheet.csv`. For each claim (row) and each concept (column) enter:

- **1** = the concept is genuinely present / true for this claim
- **0** = absent, OR only mentioned to say it did NOT happen (negation)

The negation rule is the whole point of this exercise. Label the *meaning*, not the words.

## The rule that matters most

Label **0** when the word appears but the thing didn't happen:

- "warning light came on but brakes worked fine" → `brake_problem = 0`, `warning_light = 1`
- "checked for leaks, found none" → `fluid_leak = 0`
- "no transmission issues" → `transmission_problem = 0`
- "engine ran fine, never stalled" → `engine_stall = 0`

Label **1** for paraphrase even if the exact keyword is absent:

- "liquid was pooling under the car" → `fluid_leak = 1`
- "pedal went to the floor" → `brake_problem = 1`
- "car shut off while driving" → `engine_stall = 1`
- "dashboard lit up like a christmas tree" → `warning_light = 1`

## Concept definitions

| column | present (=1) when the claim describes… |
|---|---|
| `fluid_leak` | any fluid leaking/pooling/dripping: oil, coolant, brake fluid, fuel, transmission fluid |
| `brake_problem` | brakes failed, soft/spongy pedal, not stopping, grinding, brake defect |
| `warning_light` | a dashboard warning light / indicator illuminated (check engine, ABS, airbag light, etc.) |
| `transmission_problem` | slipping, won't shift, jerking gears, stuck in gear, gearbox/transmission failure |
| `electrical_problem` | electrical fault: wiring, battery, fuses, electronics/infotainment/lights malfunction |
| `engine_stall` | engine stalled, shut off, cut out, or lost power **while driving** |

Edge calls: if genuinely ambiguous, pick the label a claims adjuster would defend, and stay consistent across rows. Do not leave cells blank — blank rows are dropped from the eval.

## When done

Save the file (keep it CSV, keep the `id` column) and tell me. I re-run A (embeddings)
vs B1 (Laya-noul) on these six concepts and report AUROC + best-F1 + negation-subset accuracy,
same format as the fire/crash/injured run.
