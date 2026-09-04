# Canonical Report Experiment Suite

This file defines which experiments must be rerun before report writing and which outputs are report-authoritative.

## Global report rules

- Use seed `20260903` unless a canonical experiment states otherwise.
- Use `100` trials per reported canonical configuration unless that contract is explicitly revised.
- Smoke/trend runs using fewer trials or a sampled voter cap are preview data only and must not be mixed into formal report tables.
- Within one comparison, all enabled algorithms must use paired scenarios: the same physical robot/task realization, voter identities, packet-loss realization, task order, tie priority, and capacity contract.
- Robust Optimization and multi-objective success/time/energy cost models remain excluded from this report cycle.
- Scalar task cost remains the existing spatial cost.
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

Purpose: measure optimal-robot execution success as robot count and directed peer-cost packet loss vary for one task.

Authoritative output root:

```text
results/peer_cost_majority/
```

The current 100-trial Experiment 1 dataset is already complete and can be retained for the one-page paper.

## Experiment 2 - Fixed 100 robots, task workload 100 to 1000

Owner:

```text
run_multitask_peer_cost_all_optimizers.py
```

The report-facing Experiment 2 is a fixed-fleet workload experiment.

Canonical workload contract:

```text
physical robots = 100 fixed
task batches = 100, 200, 300, ..., 1000
capacity_per_robot = ceil(tasks / 100)
30% independent directed scalar task-cost packet loss
```

Task counts above 100 represent a batch of tasks allocated across the fixed fleet. They must not be described as one robot physically executing multiple tasks simultaneously.

Fast default report methods:

```text
Hungarian Oracle
Voting Greedy
Voting Hungarian
Voting Auction
```

Additional optimizer families:

```text
Voting MILP               -> --include-milp
Voting ACO + Local Search -> --include-aco
```

The intended complete multi-optimizer comparison uses both flags.

The optimizer implementations remain owned by their existing modules. Experiment 2 uses uniform capacity-slot expansion and maps slot assignments back to physical robots rather than duplicating or rewriting optimizer state machines.

### Primary report metric

The one-page paper should primarily plot direct cost error relative to the full-information capacitated minimum:

```text
Cost error (%) = 100 * (method_cost - oracle_cost) / oracle_cost
```

Lower is better.

The code/CSV field is:

```text
average_optimality_gap_percent
```

The `<=5%` rate remains a supporting metric only.

Exactly identical plotted method series are merged into one legend entry; numerically different series remain separate.

### Canonical x-axis

```text
Task batch size (100 robots fixed)
```

The 10 points are:

```text
100 200 300 400 500 600 700 800 900 1000
```

### Capacity contract

For task count `T`:

```text
K = ceil(T / 100)
```

Each physical robot may receive at most `K` tasks from the batch.

The full-information Oracle, every receiver-local optimizer, and final support consensus all use this same uniform capacity through capacity-slot expansion.

Raw/summary files must record:

```text
robots
voters
tasks
capacity_per_robot
assignment_slots
```

### Communication and pairing contract

All enabled methods share the same:

- physical `100 x T` true cost matrix;
- physical voters;
- packet-loss realization;
- task order;
- tie-priority matrix;
- capacity value.

ACO internal randomness uses a separate deterministic per-receiver stream and must not consume or perturb communication RNG state.

### Runtime / batching contract

Physical receiver views are streamed in bounded batches before capacity-slot expansion.

Default voter batch size:

```text
8
```

Changing batch size changes memory/runtime only, not experiment semantics for a fixed configuration.

`--max-voters` is preview-only. Canonical formal data uses all `100` physical robots as voters.

### Output root

The workload experiment is intentionally separated from the superseded matched-scale outputs:

```text
results/multitask_peer_cost_fixed100_workload/
```

Primary raw/summary files:

```text
workload_comparison_raw.csv
workload_comparison_summary.csv
```

Primary figure:

```text
average_optimality_gap_percent.png
```

### Measured macOS runtime boundary

A real all-family runtime probe at:

```text
100 robots fixed
1000 tasks
capacity 10
1 trial
1 voter
Greedy + Hungarian + Auction + MILP + ACO + Local Search
```

completed in about `85.44 s` wall time on the user's MacBook Air. This is runtime validation only, not report data.

### New-machine runtime check

Before starting the formal run on another machine, reproduce the same largest-point timing boundary:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --tasks 1000 \
  --trials 1 \
  --max-voters 1 \
  --voter-batch-size 1 \
  --include-milp \
  --include-aco
```

Then run the full canonical x-axis at low cost:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --trials 1 \
  --max-voters 5 \
  --voter-batch-size 1 \
  --include-milp \
  --include-aco
```

These are preview/runtime-validation runs only.

### Intended complete Experiment 2 rerun

Once the new-machine timing is acceptable:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --include-milp \
  --include-aco
```

This means:

```text
100 fixed physical robots
10 task-batch points, 100 through 1000 by 100
100 trials per point
all 100 robots vote
30% directed cost-message loss
Greedy, Hungarian, Auction, MILP, ACO + Local Search
Hungarian Oracle as the full-information minimum-cost reference
```

Because MILP and especially ACO run receiver-locally, the complete command may be expensive. If the full run is not practical, any reduced trial/voter contract must be explicitly revised in the canonical documents before those data are called formal report results.

## Experiment 3 - Direct vs Voting ablation

Owner:

```text
run_multitask_voting_ablation.py
```

This experiment remains supporting causal evidence. The user has decided not to include it in the current one-page CACS paper, so it is not required for the current one-page rerun cycle.

Its historical fixed-100 capacity-one results must not be mixed into the new Experiment 2 workload curve.

## Report data acceptance checklist

Before any dataset is called formal report data, confirm:

1. Experiment 1 uses the completed canonical 100-trial dataset;
2. Experiment 2 uses exactly `100` physical robots at every task load;
3. Experiment 2 task batches are exactly `100..1000` in steps of `100` unless the canonical spec is explicitly revised;
4. `capacity_per_robot = ceil(tasks/100)` is recorded and enforced by Oracle, local optimizers, and consensus;
5. formal Experiment 2 uses all `100` physical robots as voters;
6. preview runs with `--max-voters` or fewer than 100 trials are labeled preview;
7. paired scenario/communication inputs are shared by all enabled methods;
8. ACO uses separate algorithm-internal RNG and does not regenerate communication loss;
9. bounded zero-loss integration gates pass for all enabled method families;
10. all final report CSVs were generated by the final code version;
11. the main Experiment 2 figure uses direct cost error from the minimum;
12. calculations use raw CSV values even when presentation values are rounded;
13. task counts above 100 are described as allocation workload/batches, not simultaneous physical task execution.

## Current one-page report structure

The CACS one-page paper should tell two main stories:

1. **Single-task communication robustness** - how packet loss and fleet size affect majority execution success.
2. **Fixed-fleet workload scaling** - with 100 robots held constant, how Voting assignment cost error changes as the task batch grows from 100 to 1000 under a shared capacity and communication model.

The one-page draft may reserve the Experiment 2 figure area until the new fixed-100 workload rerun is complete. The Direct-vs-Voting ablation and older matched-scale preview remain supporting/historical material rather than current main-figure data.
