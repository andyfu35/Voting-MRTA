# Fixed-100-Robot Lossy P2P Voting Workload Experiment

This is the canonical report-facing Experiment 2 for the current one-page paper cycle.

## Research question

Under `30%` independent directed P2P scalar task-cost packet loss, how does Democracy/Voting assignment quality change as the task workload grows while the physical fleet remains fixed at `100` robots?

The physical fleet is always:

```text
robot_count = 100
```

The canonical task-batch sweep is:

```text
50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
550, 600, 650, 700, 750, 800, 850, 900, 950, 1000 tasks
```

These are **task batches / allocation workload**, not claims that one physical robot executes multiple tasks simultaneously.

For a task batch of size `T`, the uniform assignment capacity is:

```text
capacity_per_robot = ceil(T / 100)
```

Examples:

```text
T=50   -> capacity 1
T=100  -> capacity 1
T=150  -> capacity 2
T=500  -> capacity 5
T=1000 -> capacity 10
```

This preserves a bounded, balanced allocation model while allowing the workload to exceed the fleet size.

## Canonical owner

```text
run_multitask_peer_cost_all_optimizers.py
```

The filename is retained for command compatibility even though the experiment is no longer a matched robot/task scale sweep.

## Compared methods

Fast default methods remain:

```text
Hungarian Oracle
Voting Greedy
Voting Hungarian
Voting Auction
```

Optional optimizer families can be included without changing the fast default:

```bash
--include-milp
--include-aco
```

They add:

```text
Voting MILP
Voting ACO + Local Search
```

The intended full multi-optimizer rerun is:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --include-milp \
  --include-aco
```

`--only-milp` remains available for isolated timing or completion of missing MILP data.

The optimization algorithms themselves remain owned by their existing modules:

```text
Greedy / Hungarian / Auction:
run_multitask_peer_cost_experiment.py

MILP / ACO + Local Search:
run_multitask_optimizer_screening.py
```

Experiment 2 does not copy a second implementation of any optimizer.

## Capacity-slot representation

The existing optimizer owners solve capacity-one assignment matrices. Experiment 2 therefore owns one explicit representation transformation instead of changing or duplicating each optimizer.

For physical cost matrix:

```text
C in R^(100 x T)
```

and uniform capacity `K = ceil(T/100)`, each physical robot row is repeated `K` times to create capacity slots:

```text
C_slot in R^((100*K) x T)
```

Each capacity slot can receive at most one task under the existing assignment owners. Slot assignments are mapped back to physical robots after optimization.

This is equivalent to the physical constraint:

```text
sum_j x_ij <= K   for every physical robot i
sum_i x_ij = 1    for every task j
```

The same slot representation is used for:

- full-information Hungarian Oracle;
- receiver-local Greedy;
- receiver-local Hungarian;
- receiver-local Auction;
- receiver-local MILP;
- receiver-local ACO + Local Search;
- final support-maximizing consensus.

No optimizer receives a different capacity model.

## Controlled settings

- Physical robots: `100` fixed.
- Directed P2P scalar task-cost packet loss: `30%`.
- Task batches: `50..1000` in steps of `50`.
- Uniform per-robot batch capacity: `ceil(tasks/100)`.
- Formal trials per reported task count: `100` unless explicitly revised later.
- Canonical formal mode uses all `100` physical robots as voters.
- Task delivery: reliable.
- Final proposal collection: reliable/in-window in this controlled stage.
- Route execution noise, retransmission, permanent robot failure, deadlines, Robust Optimization, and multi-objective costs remain excluded.
- Scalar task cost remains:

```text
C_ij = 0.05 + EuclideanDistance(robot_i, task_j)
```

## Paired trial contract

For every `(task_count, trial)` pair:

```text
trial_seed = seed + task_count * 100003 + trial * 1009
```

Separate deterministic streams derive:

- physical robot/task geometry, task order, tie priority;
- optional voter selection;
- packet-loss visibility;
- ACO internal search randomness.

All enabled Voting methods receive the same:

- physical `100 x T` cost matrix;
- physical voter identities;
- directed packet-loss realization;
- task order;
- tie-priority matrix;
- uniform capacity value.

No optimizer samples or regenerates communication loss.

ACO uses a separate deterministic per-physical-receiver RNG stream, so enabling ACO does not change the communication realization or another optimizer's input.

## P2P information model

Robot `i` owns its own task-cost row. For physical receiver `r` and task `j`, visibility is sampled independently for the directed message:

```text
sender robot i -> receiver robot r -> task j
```

Every receiver always knows its own physical sender row.

Missing physical robot/task costs are represented as:

```text
+inf
```

before capacity-slot expansion. Therefore all slots belonging to the same physical robot inherit the same receiver-local visibility state for a task.

## Receiver batching

Receiver-local physical views are streamed in bounded batches:

```text
DEFAULT_VOTER_BATCH_SIZE = 8
```

For each physical receiver batch:

1. sample directed physical sender-to-receiver visibility;
2. materialize only that batch's incomplete `100 x T` views;
3. expand physical robot rows into capacity slots;
4. run every enabled optimizer on those same slot views;
5. map slot assignments back to physical robots;
6. accumulate physical robot/task proposal support;
7. discard the batch views.

Batch size changes memory/runtime only. It does not change voter identities, communication samples, ACO receiver seeds, support totals, or final assignments for a fixed experiment configuration.

## Voting support and capacitated consensus

Each valid receiver proposal assigns every task to one physical robot and respects:

```text
load(robot_i) <= capacity_per_robot
```

Support is accumulated on physical robots:

```text
S_ij = number of valid receiver proposals assigning task j to robot i
```

For final consensus, physical support/tie rows are expanded into the same capacity-slot representation and passed to the existing support-consensus owner.

The final physical assignment therefore maximizes proposal support subject to the same uniform robot capacity.

The paired tie-priority matrix is used only for equal-support ties. True cost is not used as a hidden consensus tie-break.

## Optimizer routing boundaries

Default Greedy/Hungarian/Auction:

```text
run_multitask_peer_cost_all_optimizers.py::solve_slot_voter_batch_proposals
    -> run_multitask_peer_cost_experiment.py::solve_local_optimizer_proposals
```

MILP:

```text
solve_slot_voter_batch_proposals
    -> solve_milp_batch_proposals
    -> run_multitask_optimizer_screening.py::solve_milp_assignment
```

ACO + Local Search:

```text
solve_slot_voter_batch_proposals
    -> solve_aco_batch_proposals
    -> run_multitask_optimizer_screening.py::solve_aco_assignment
```

The Experiment 2 adapter owns capacity representation, physical/slot mapping, paired RNG selection, and proposal bookkeeping only.

## Zero-loss integration gates

Preflight checks are intentionally bounded so they verify integration without turning startup into the expensive experiment itself.

- single-task Greedy is checked against the capacitated Hungarian reference;
- Voting Hungarian and Voting Auction are checked at bounded task loads up to `200`, including a capacity-greater-than-one case;
- Voting MILP is checked against the Oracle up to `100` tasks under its existing numerical tolerance;
- Voting ACO + Local Search is checked for a valid capacity-feasible complete-information proposal up to `150` tasks, but is not incorrectly required to equal the exact Oracle.

MILP objective matching uses:

```text
MILP_NUMERICAL_TOLERANCE_PERCENT = 1e-6
```

## Primary report metric

The primary Experiment 2 figure is direct cost error relative to the full-information capacitated minimum:

```text
Cost error (%) = 100 * (method_cost - oracle_cost) / oracle_cost
```

Lower is better; `0%` means the method reaches the minimum total assignment cost under the same capacity contract.

The CSV field remains:

```text
average_optimality_gap_percent
```

Supporting metrics remain:

- optimal-cost match;
- near-optimal within 5%;
- exact optimal physical assignment;
- valid local proposal rate.

Generated figures are line-only. Exactly identical curves share one legend entry; near-overlapping but non-identical curves remain separate.

The x-axis is:

```text
Task batch size (100 robots fixed)
```

## Output separation

The fixed-fleet workload experiment is stored separately from the superseded matched-scale data:

```text
results/multitask_peer_cost_fixed100_workload/
```

Raw and summary data:

```text
workload_comparison_raw.csv
workload_comparison_summary.csv
```

Report tables:

```text
report_average_optimality_gap_percent.csv
report_optimal_cost_match_percent.csv
report_near_optimal_5pct_percent.csv
report_exact_optimal_assignment_percent.csv
report_valid_proposal_rate_percent.csv
```

Primary figure:

```text
average_optimality_gap_percent.png
```

Raw/summary output records:

```text
robots
voters
tasks
capacity_per_robot
assignment_slots
method
method_label
```

so the fixed fleet and changing workload/capacity remain explicit in the data.

## Recommended real-machine rerun sequence

Before the complete rerun, use a small all-family smoke:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --tasks 50 150 300 \
  --trials 1 \
  --max-voters 5 \
  --voter-batch-size 1 \
  --include-milp \
  --include-aco
```

Then inspect a full-range low-trial trend:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --trials 3 \
  --max-voters 10 \
  --voter-batch-size 1 \
  --include-milp \
  --include-aco
```

If runtime is acceptable, the intended complete all-family Experiment 2 rerun is:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --include-milp \
  --include-aco
```

That command means:

```text
100 physical robots
20 task-batch points from 50 to 1000
100 trials per point
all 100 robots vote
Greedy + Hungarian + Auction + MILP + ACO/Local Search
30% directed P2P cost-message loss
```

This may be computationally expensive, especially because ACO and MILP execute receiver-locally. A completed smaller run must not be silently described as the canonical 100-trial all-voter run.

## Interpretation boundary

The new x-axis is increasing **task workload in a fixed 100-robot fleet**.

It is not matched robot/task scaling and it is not a claim that one robot physically executes `capacity_per_robot` tasks simultaneously.

Runs with `--max-voters` are capped-voter previews.

MILP or ACO workload-scaling claims are allowed only when those flags were enabled and the run completed successfully.

The proposal-support consensus stage remains a controlled centralized boundary and must not be described as fully asynchronous decentralized consensus.
