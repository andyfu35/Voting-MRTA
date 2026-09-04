# Canonical Report Experiment Suite

This file defines which experiments must be rerun before report writing and which outputs are report-authoritative.

## Global report rules

- Use seed `20260903` unless a canonical experiment states otherwise.
- Use the experiment-specific canonical trial count. Experiment 1 remains 100 trials; Experiment 2 is 20 trials per task point because its runtime contract was explicitly revised.
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

The completed 100-trial Experiment 1 dataset remains report-authoritative.

## Experiment 2 - Fixed 100 robots, fast Voting workload scaling

Canonical owner:

```text
run_multitask_peer_cost_all_optimizers.py
```

Fast heuristic owner:

```text
run_multitask_workload_heuristics.py
```

Canonical workload contract:

```text
physical robots = 100 fixed
task batches = 100, 200, ..., 1000
capacity_per_robot = ceil(tasks / 100)
30% independent directed scalar task-cost packet loss
20 trials per task point
all 100 physical robots vote
```

Task counts above 100 represent an allocation batch across the fixed fleet. They must not be described as one robot physically executing all assigned tasks simultaneously.

### Canonical methods

```text
Hungarian Oracle                full-information minimum-cost reference
Voting Sequential Greedy
Voting Global Greedy
Voting Static Regret-2 Greedy
Voting Hungarian
```

The three fast heuristics operate directly on physical receiver-local `100 x T` cost matrices and explicit robot capacity.

Voting Hungarian uses the existing Hungarian assignment owner after capacity-slot expansion of each receiver-local incomplete view.

MILP, ACO + Local Search, and Auction are no longer part of the report-facing lossy workload sweep. Their implementations remain in their existing owners/supporting studies. This is a deliberate runtime/research-scope decision, not deletion of those optimizer implementations.

### Why this method set is canonical

The earlier all-five-family design took about `85.44 s` on macOS for only `T=1000`, one trial, and one voter, showing that receiver-local MILP/ACO could not reasonably satisfy the approximately one-hour rerun budget.

The first fast redesign retained Voting Auction. Its measured all-voter preview used task points `100, 500, 1000`, two trials per point, and four process workers, and required `193.18 s` wall time. That implied roughly `1 h 47 min` for the ten-point, 20-trial workload if scaling remained similar.

Voting Auction is therefore replaced by Voting Hungarian as the one exact local assignment-family comparison. The underlying assignment objective is unchanged; the change targets receiver-local runtime. The Oracle and Voting Hungarian use the same Hungarian algorithm under different information conditions: the Oracle sees the full true cost matrix, while each Voting Hungarian voter sees only its own lossy receiver-local cost view before Voting aggregation.

### Primary report metric

The one-page paper should primarily plot direct cost error relative to the full-information capacitated minimum:

```text
Cost error (%) = 100 * (method_cost - oracle_cost) / oracle_cost
```

Lower is better. The Oracle remains in CSV data but is not plotted as a redundant `0%` curve.

The code/CSV field is:

```text
average_optimality_gap_percent
```

The `<=5%` rate is supporting only.

Exactly identical plotted method series may share one legend entry; numerically different series remain separate.

### Figure curve count

The primary Experiment 2 cost-error figure normally contains **four curves**:

```text
Voting Sequential Greedy
Voting Global Greedy
Voting Static Regret-2 Greedy
Voting Hungarian
```

`Hungarian Oracle` supplies the minimum-cost denominator and `0%` reference but is not drawn as a fifth curve.

### Canonical x-axis

```text
Task batch size (100 robots fixed)
```

The ten points are:

```text
100 200 300 400 500 600 700 800 900 1000
```

### Capacity contract

For task count `T`:

```text
K = ceil(T / 100)
```

Each physical robot may receive at most `K` tasks from the allocation batch.

The Oracle, every receiver-local heuristic, Voting Hungarian, evaluation, and final support consensus all enforce the same physical capacity contract.

### Communication and pairing contract

All Voting methods share the same:

- physical `100 x T` true cost matrix;
- physical voter identities;
- packet-loss realization;
- task order;
- tie-priority matrix;
- capacity value.

Parallel process scheduling never enters a random seed formula.

### Runtime contract

Independent `(task_count, trial)` jobs are parallelized with `ProcessPoolExecutor`.

Canonical default:

```text
workers = min(4, available CPU count)
receiver batch size = 4
```

Worker count and receiver batch size are runtime controls only. They must not change the paired experiment semantics.

Parallel mode suppresses receiver-level worker output and reports trial completions from the parent process. When workers exceed one, BLAS/OpenMP thread-count environment variables default to one thread unless the user explicitly set them, avoiding nested thread oversubscription.

### Zero-loss gates

Before the lossy sweep:

- all three heuristics must return valid capacity-feasible complete-information proposals at bounded loads up to 200 tasks;
- Voting Hungarian must match the capacitated Hungarian Oracle cost at bounded loads up to 200 tasks under the existing exact numerical tolerance.

Heuristics are not required to equal the Oracle.

### Authoritative output root

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

Raw/summary files must retain:

```text
robots
voters
tasks
capacity_per_robot
assignment_slots
method
method_label
```

### Real-machine timing sequence

After pulling the exact-solver swap, first run an all-voter parallel preview:

```bash
time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 100 500 1000 \
  --trials 2 \
  --workers 4
```

The canonical Experiment 2 command remains:

```bash
time python run_multitask_peer_cost_all_optimizers.py
```

That command uses ten task points, 20 trials per point, all 100 voters, the four report-facing Voting methods, and up to four process workers.

The one-hour runtime is an objective that must be re-measured after the Auction-to-Hungarian swap; it is not guaranteed by specification alone.

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
3. Experiment 2 task batches are exactly `100..1000` in steps of `100`;
4. Experiment 2 has `20` completed trials per task point unless a later canonical revision explicitly changes this;
5. formal Experiment 2 uses all `100` physical robots as voters;
6. `capacity_per_robot = ceil(tasks/100)` is recorded and enforced by all methods and consensus;
7. paired scenario/communication inputs are shared by all four Voting methods;
8. parallel worker scheduling does not change seeds or results;
9. bounded heuristic feasibility and Voting Hungarian exactness gates pass;
10. all final report CSVs were generated by the final code version;
11. the main Experiment 2 figure uses direct cost error from the minimum;
12. calculations use raw CSV values even when presentation values are rounded;
13. task counts above 100 are described as allocation workload/batches;
14. MILP/ACO/Auction claims come only from their separate supporting data, not from the new canonical lossy workload curve.

## Current one-page report structure

The CACS one-page paper should tell two main stories:

1. **Single-task communication robustness** - how packet loss and fleet size affect majority execution success.
2. **Fixed-fleet workload scaling** - with 100 robots held constant, how Voting assignment cost error changes as the task batch grows from 100 to 1000 using scalable local decision rules.

For the paper bibliography, retain the Hungarian assignment reference. It supports both the full-information Hungarian Oracle and Voting Hungarian, which differ in available information and Voting aggregation rather than in the underlying assignment algorithm. The three greedy baselines are explicitly defined by the experiment and should not be given unrelated optimizer references merely to fill the bibliography.
