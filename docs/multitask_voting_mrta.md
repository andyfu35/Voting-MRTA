# Multi-Task Lossy P2P Voting Scaling Experiment

This is the canonical report-facing Experiment 2 for the current one-page paper cycle.

## Research question

Under `30%` independent directed P2P scalar task-cost packet loss, how does Democracy/Voting assignment quality change as the multi-robot system scales toward `1000` simultaneous tasks?

The capacity-one assignment semantics remain unchanged by using a matched fleet at every point:

```text
robot_count = task_count
```

The current default scale points remain:

```text
50, 100, 200, 400, 600, 800, 1000 robots/tasks
```

For a denser figure, the CLI may explicitly request 50-step points from `50` through `1000`; this changes only which scale points are sampled.

## Canonical owner

```text
run_multitask_peer_cost_all_optimizers.py
```

The filename is retained for command compatibility.

## Compared methods

The default scalable methods are:

```text
Hungarian Oracle
Voting Greedy
Voting Hungarian
Voting Auction
```

The default command deliberately excludes expensive receiver-local methods.

An optional MILP probe is now available:

```bash
python run_multitask_peer_cost_all_optimizers.py --include-milp ...
```

This adds:

```text
Voting MILP
```

without changing the default method set. The MILP algorithm itself remains owned by:

```text
run_multitask_optimizer_screening.py::solve_milp_assignment
```

The scaling owner only routes each incomplete receiver matrix to that existing owner and accumulates the returned proposal support. No second MILP implementation is introduced.

`ACO + Local Search` remains excluded from the scaling sweep for now because its receiver-local runtime was substantially heavier than MILP in the preceding optimizer screening. It can be reconsidered after the MILP timing probe.

## Controlled settings

- Directed P2P scalar task-cost packet loss: `30%`.
- Robot capacity: one simultaneous task per robot.
- Matched scale: `robot_count == task_count`.
- Formal trials per reported scale point: `100` unless this contract is explicitly revised.
- Task delivery: reliable.
- Final proposal collection: reliable/in-window in this controlled stage.
- Route planning, execution noise, retransmission, permanent failure, deadlines, Robust Optimization, and multi-objective costs remain excluded.
- Scalar cost remains:

```text
C_ij = 0.05 + EuclideanDistance(robot_i, task_j)
```

## Paired trial contract

For every `(task_count, trial)` pair:

```text
robot_count = task_count
trial_seed = seed + task_count * 100003 + trial * 1009
```

Separate deterministic streams derive:

- scenario geometry, task order, and tie priority;
- optional voter selection;
- packet-loss visibility.

All enabled Voting methods receive the same:

- true cost matrix;
- voter identities;
- directed packet-loss views;
- task order;
- tie-priority matrix.

No optimizer samples or regenerates its own communication realization.

## P2P information model

Robot `i` owns its own task-cost row. For receiver `r` and task `j`, directed scalar visibility is sampled independently:

```text
sender i -> receiver r -> task j
```

Every receiver always knows its own sender row. Missing entries are represented as:

```text
+inf
```

and remain unavailable assignment edges.

## Scalable receiver batching

The owner does not materialize a full `receiver x sender x task` tensor at large scale. Voting receivers are streamed in bounded batches:

```text
DEFAULT_VOTER_BATCH_SIZE = 8
```

For each batch:

1. sample that batch's directed visibility;
2. materialize only that batch's incomplete cost views;
3. run every enabled optimizer on those same views;
4. accumulate proposal support per optimizer;
5. discard the local views before the next batch.

Batch size changes memory/runtime only. It does not change selected voters, packet-loss samples, proposals, support totals, or consensus for a fixed configuration.

## Optimizer routing boundary

Default Greedy/Hungarian/Auction proposals remain owned by:

```text
run_multitask_peer_cost_experiment.py::solve_local_optimizer_proposals
```

Optional MILP proposals are routed through:

```text
run_multitask_peer_cost_all_optimizers.py::solve_voter_batch_proposals
    -> run_multitask_peer_cost_all_optimizers.py::solve_milp_batch_proposals
    -> run_multitask_optimizer_screening.py::solve_milp_assignment
```

`solve_milp_batch_proposals` owns only receiver iteration and proposal validity bookkeeping. The optimization model, missing-edge treatment, HiGHS invocation, and numerical behavior remain owned by `solve_milp_assignment`.

## Voting support / consensus

Each valid receiver proposal is a complete capacity-one task assignment.

Support is:

```text
S_ij = number of valid receiver proposals assigning task j to robot i
```

The final assignment maximizes total support subject to:

```text
sum_i x_ij = 1   for every task j
sum_j x_ij <= 1  for every robot i
```

The paired tie-priority matrix is used only for equal-support ties. True cost is not used as a hidden consensus tie-break.

## Full-voter mode vs trend-preview mode

Canonical behavior uses all robots as voters:

```bash
python run_multitask_peer_cost_all_optimizers.py
```

For immediate trend inspection, a receiver cap may be used explicitly:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --tasks 50 100 150 200 250 300 350 400 450 500 \
          550 600 650 700 750 800 850 900 950 1000 \
  --trials 10 \
  --max-voters 100
```

This is preview data, not canonical full-voter report data.

## Optional MILP timing probe

Before attempting a dense MILP sweep, first measure a small real run:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --tasks 50 100 150 \
  --trials 2 \
  --max-voters 20 \
  --include-milp
```

If this is acceptably fast, increase voter count and scale gradually. Do not start a 1000-scale all-voter MILP sweep before measuring the smaller probe on the target machine.

## Zero-loss contracts

Before the lossy sweep:

- single-task Greedy must match the Hungarian Oracle;
- Voting Hungarian must match the Oracle at representative scales;
- Voting Auction must match the Oracle at representative scales;
- when `--include-milp` is enabled, Voting MILP is additionally checked against the Oracle on a bounded complete-information integration case up to 50 tasks.

The bounded MILP gate deliberately avoids turning preflight into a million-variable 1000-task MILP. Larger lossy MILP cases remain measured experiment points rather than preflight checks.

MILP objective matching uses its existing numerical tolerance:

```text
MILP_NUMERICAL_TOLERANCE_PERCENT = 1e-6
```

## Primary report metric

The primary Experiment 2 figure is now direct cost error relative to the full-information minimum-cost reference:

```text
Cost error (%) = 100 * (method_cost - oracle_cost) / oracle_cost
```

Lower is better; `0%` means the method reaches the minimum total cost.

The code field remains:

```text
average_optimality_gap_percent
```

but the report-facing y-axis is labeled:

```text
Cost error from minimum (%)
```

The script continues to record supporting metrics:

- optimal-cost match;
- near-optimal within 5%;
- exact optimal assignment;
- valid local proposal rate.

The `<=5%` metric is no longer the primary figure.

Generated figures use line-only curves with no point markers. If two method series are numerically identical at every plotted scale point, the plotting boundary merges them into one legend entry such as:

```text
Voting Hungarian / Auction
```

Near-overlapping but non-identical curves remain separate.

## Outputs

```text
results/multitask_peer_cost_scaling/
```

Raw and summary data:

```text
scaling_comparison_raw.csv
scaling_comparison_summary.csv
```

Report-ready CSVs:

```text
report_average_optimality_gap_percent.csv
report_optimal_cost_match_percent.csv
report_near_optimal_5pct_percent.csv
report_exact_optimal_assignment_percent.csv
report_valid_proposal_rate_percent.csv
```

Primary figure file:

```text
average_optimality_gap_percent.png
```

## Interpretation boundary

The x-axis represents matched system scale (`robots = simultaneous tasks`), not increasing task utilization inside a fixed fleet.

Runs using `--max-voters` measure a capped-voter preview and must not be described as full-voter scaling.

MILP scaling claims are allowed only for runs that explicitly enabled `--include-milp` and completed successfully.

ACO scaling claims are not supported by this experiment yet.

The proposal-support consensus stage remains a controlled centralized boundary and must not be described as fully asynchronous decentralized consensus.
