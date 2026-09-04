# Canonical Report Experiment Suite

This file defines which experiments must be rerun before report writing and which outputs are report-authoritative.

## Global report rules

- Use seed `20260903` unless a canonical experiment states otherwise.
- Use `100` trials per reported canonical configuration unless a later report decision explicitly changes that contract.
- Smoke/trend runs using fewer trials or a sampled voter cap are validation/preview only and must not be mixed into formal report tables.
- Within a comparison, all algorithms must use paired scenarios: the same robot/task realization and the same packet-loss realization.
- Robust Optimization and multi-objective success/time/energy cost models remain excluded from this report cycle.
- The scalar task cost remains the existing spatial cost.
- Partial output from a run that terminates before `save_outputs` is not report data.

## Experiment 1 - Single-task P2P majority robustness

Owner:

```text
run_peer_cost_majority_experiment.py
```

Canonical run:

```bash
python run_peer_cost_majority_experiment.py
```

Purpose:

Measure optimal-robot execution success as robot count and directed peer-cost packet loss vary for one task.

Authoritative output root:

```text
results/peer_cost_majority/
```

Primary result:

```text
packet-loss x robot-count optimal execution success
```

The current 100-trial Experiment 1 dataset is already complete and can be retained for the one-page paper.

## Experiment 2 - Matched-scale lossy P2P Voting through 1000 tasks

Owner:

```text
run_multitask_peer_cost_all_optimizers.py
```

The filename is retained for compatibility, but the main scaling experiment now excludes the two optimizer families that made the receiver-local sweep prohibitively slow.

Canonical matched scales:

```text
robots = tasks
50, 100, 200, 400, 600, 800, 1000
```

Controlled condition:

```text
30% independent directed scalar task-cost packet loss
one simultaneous task per robot
100 paired trials per formal scale point
all robots participate as voting receivers in canonical mode
```

Compared report methods:

```text
Hungarian Oracle
Voting Greedy
Voting Hungarian
Voting Auction
```

`Voting MILP` and `Voting ACO + Local Search` are no longer part of the main Experiment 2 scale sweep. Their previous complete-information screening remains supporting optimizer characterization only.

### Primary report metric

The one-page paper should primarily plot:

```text
Trials within 5% of optimum (%)
```

against matched robot/task scale.

Higher is better.

Supporting CSV metrics remain:

```text
average optimality gap
optimal-cost match
exact optimal assignment
valid local proposal rate
```

### Memory/scaling contract

Experiment 2 no longer materializes the full receiver x sender x task tensor at once. Receiver-local views are streamed in bounded batches and proposal support is accumulated before final consensus.

The default voter batch size is `8`; changing it changes memory/runtime only, not the experimental result for a fixed seed/configuration.

Authoritative output root:

```text
results/multitask_peer_cost_scaling/
```

### Trend preview before a formal run

To inspect the 50-to-1000 trend without waiting for every robot to vote in every trial:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --tasks 50 100 200 400 600 800 1000 \
  --trials 10 \
  --max-voters 100
```

This uses all voters when the fleet has at most 100 robots and a deterministic random sample of 100 voting receivers for larger fleets.

This command is **preview only**. Its CSVs must not be presented as canonical full-voter 100-trial report data.

A smaller smoke is:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --tasks 50 100 200 \
  --trials 2 \
  --max-voters 50
```

### Formal run

The current formal contract remains:

```bash
python run_multitask_peer_cost_all_optimizers.py
```

This means all robots vote and every listed scale point uses 100 trials. Because the 1000-scale full-voter sweep is still computationally expensive, do not begin it until the trend preview is inspected and the final paper scale points are confirmed.

## Experiment 3 - Direct vs Voting ablation

Owner:

```text
run_multitask_voting_ablation.py
```

This experiment remains available as supporting causal evidence, but the user has explicitly decided not to include the ablation in the current one-page CACS paper. It is therefore not required for the current one-page rerun cycle.

Its existing Hungarian/Auction results remain historical supporting evidence and must not be mixed into the Experiment 2 scaling curve.

## Required rerun order for the current one-page paper

Experiment 1 is already complete.

The immediate next step is the Experiment 2 trend preview:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --tasks 50 100 200 400 600 800 1000 \
  --trials 10 \
  --max-voters 100
```

Inspect the resulting `Trials within 5% of optimum (%)` trend before deciding whether every scale point needs the canonical 100-trial/full-voter run.

## Report data acceptance checklist

Before any dataset is called formal report data, confirm:

1. Experiment 1 uses the completed canonical 100-trial dataset;
2. Experiment 2 clearly records `robots`, `tasks`, and `voters` in raw/summary output;
3. canonical Experiment 2 uses `robots == tasks`;
4. formal Experiment 2 uses all robots as voters unless the canonical specification is explicitly revised later;
5. preview runs with `--max-voters` are labeled preview and are not silently mixed with formal data;
6. the zero-loss gate reports Hungarian/Auction consistency with the Oracle;
7. all final report CSVs were generated by the final code version;
8. report plots use the <=5% near-optimal rate as the primary Experiment 2 curve;
9. calculations use raw CSV values even when presentation values are rounded.

## Current one-page report structure

The CACS one-page paper should tell two main stories:

1. **Single-task communication robustness** - how packet loss and fleet size affect majority execution success.
2. **Large-scale multi-task Voting** - how the <=5% near-optimal success rate changes as the matched robot/task system grows toward 1000.

The Direct-vs-Voting ablation and complete-information optimizer screening remain supporting material rather than main one-page figures.
