# Fixed-100-Robot Lossy P2P Voting Workload Experiment

This is the canonical report-facing Experiment 2 for the current one-page paper cycle.

## Research question

Under `30%` independent directed P2P scalar task-cost packet loss, how does Voting-based task allocation quality change as task workload grows while the physical fleet remains fixed at `100` robots?

The physical fleet is always:

```text
robot_count = 100
```

The canonical task-batch sweep is:

```text
100, 200, 300, 400, 500,
600, 700, 800, 900, 1000 tasks
```

These are allocation batches/workload, not a claim that one physical robot executes all assigned tasks simultaneously.

For a task batch of size `T`, the uniform assignment capacity is:

```text
capacity_per_robot = ceil(T / 100)
```

At the canonical 100-task spacing, this gives capacities `1..10`.

## Canonical owner

```text
run_multitask_peer_cost_all_optimizers.py
```

Fast heuristic algorithms are owned separately by:

```text
run_multitask_workload_heuristics.py
```

The Hungarian assignment algorithm used by both the full-information Oracle and the receiver-local exact comparison remains owned by:

```text
run_multitask_peer_cost_experiment.py::solve_hungarian_assignment
```

The Experiment 2 adapter only performs capacity-slot representation, receiver-local routing, physical mapping, Voting support, and evaluation.

## Why the optimizer set changed

A real macOS timing probe of the earlier five-family design used `1000` tasks, one trial, one voter, and Greedy/Hungarian/Auction/MILP/ACO + Local Search. It took about `85.44 s` wall time. Scaling receiver-local MILP/ACO to all voters and repeated trials was incompatible with the approximately one-hour experiment budget.

The first fast redesign retained Voting Auction. A real all-voter macOS timing preview using task points `100, 500, 1000`, two trials per point, and four worker processes took `193.18 s` wall time. Extrapolating that measured workload to ten task points and 20 trials was still roughly `1 h 47 min` before considering run-to-run variation.

Therefore the report-facing lossy workload experiment now uses three fast heuristics plus receiver-local Hungarian. MILP and ACO + Local Search remain in the separate complete-information optimizer screening. Auction also remains implemented in its existing owner but is no longer part of the canonical lossy workload sweep.

The reason for choosing Voting Hungarian instead of Voting Auction is runtime, not a change in objective. Both are exact assignment-family solvers under complete information in the tested integration contract, while prior paired runs repeatedly showed the same or nearly identical final Voting cost. Hungarian is expected to reduce receiver-local runtime substantially on this workload.

## Canonical compared methods

The report-facing methods are:

```text
Hungarian Oracle                full-information minimum-cost reference only
Voting Sequential Greedy       fast heuristic
Voting Global Greedy           fast heuristic
Voting Static Regret-2 Greedy  fast heuristic
Voting Hungarian               exact receiver-local assignment family
```

The Oracle is retained in CSV output but is not plotted as a quality curve; `0%` on the cost-error axis is the minimum-cost reference.

This means the primary Experiment 2 figure normally contains **four plotted optimizer curves**. The Oracle is a reference baseline, not a fifth plotted curve.

## Fast heuristic definitions

### Voting Sequential Greedy

For each receiver-local incomplete physical `100 x T` cost matrix, process tasks in the paired trial task order and assign each task to the cheapest finite-cost robot with remaining batch capacity.

Owner:

```text
run_multitask_workload_heuristics.py::solve_sequential_greedy_capacitated
```

### Voting Global Greedy

Sort all finite receiver-visible physical robot/task edges once by cost. Consume the cheapest edge whose task is still unassigned and whose robot still has capacity until all tasks are assigned or no feasible edge remains.

Owner:

```text
run_multitask_workload_heuristics.py::solve_global_greedy_capacitated
```

### Voting Static Regret-2 Greedy

For each task, compute a one-time priority from the gap between its best and second-best receiver-visible robot costs. Tasks with only one visible candidate have infinite urgency. Ties use the paired trial task order. Then process tasks in descending regret priority and assign each to its cheapest robot with remaining capacity.

This is deliberately named **Static Regret-2 Greedy** because the two-best regret order is computed once per receiver rather than recomputed after every assignment.

Owner functions:

```text
run_multitask_workload_heuristics.py::compute_static_regret2_priority
run_multitask_workload_heuristics.py::solve_regret2_greedy_capacitated
```

### Voting Hungarian

Each physical receiver first obtains its own incomplete `100 x T` physical cost matrix. Capacity-slot expansion then converts it to the existing capacity-one Hungarian representation. The existing Hungarian owner solves that receiver-local incomplete assignment and the slot result is mapped back to physical robot IDs before Voting support is counted.

Routing:

```text
run_multitask_peer_cost_all_optimizers.py::solve_voter_batch_proposals
    -> run_multitask_peer_cost_experiment.py::solve_local_optimizer_proposals
    -> run_multitask_peer_cost_experiment.py::solve_hungarian_assignment
```

`Voting Hungarian` is not the same experimental condition as `Hungarian Oracle`:

```text
Hungarian Oracle  = full true cost matrix, one minimum-cost reference solve per trial
Voting Hungarian  = one incomplete receiver-local solve per voter, then Voting consensus
```

They share the same mathematical assignment algorithm but differ in information availability and aggregation.

## Capacity representation

Heuristics operate directly on physical `100 x T` receiver-local matrices and explicit physical robot capacities. This avoids expanding every heuristic problem to a large slot matrix.

Voting Hungarian, the full-information Oracle, and final support consensus use capacity slots because the existing exact Hungarian/support owners are capacity-one assignment solvers.

For cost matrix:

```text
C in R^(100 x T)
```

and uniform capacity `K`, slot expansion is:

```text
C_slot in R^((100*K) x T)
```

This represents the physical constraints:

```text
sum_i x_ij = 1    for every task j
sum_j x_ij <= K   for every physical robot i
```

All compared methods are evaluated under the same physical capacity contract.

## Controlled settings

- Physical robots: `100` fixed.
- Task batches: `100..1000` in steps of `100`.
- Directed P2P scalar cost-message loss: `30%`.
- Uniform per-robot batch capacity: `ceil(tasks/100)`.
- Canonical report trials per task point: `20`.
- Canonical mode uses all `100` physical robots as voters.
- Default process workers: up to `4`, bounded by available CPU count.
- Default receiver batch size inside each trial: `4`.
- Task delivery: reliable.
- Final proposal collection: reliable/in-window in this controlled stage.
- Route execution noise, retransmission, permanent robot failure, deadlines, Robust Optimization, and multi-objective costs remain excluded.
- Scalar cost remains:

```text
C_ij = 0.05 + EuclideanDistance(robot_i, task_j)
```

The reduction from 100 to 20 trials is a deliberate time-budget change. A larger value may be supplied with `--trials` if measured runtime permits, but such a rerun must be reported with its actual trial count.

## Paired trial contract

For every `(task_count, trial)` pair:

```text
trial_seed = seed + task_count * 100003 + trial * 1009
```

Separate deterministic streams derive:

- physical robot/task geometry, task order, and tie priority;
- optional voter selection;
- directed packet-loss visibility.

All Voting methods receive the same:

- physical `100 x T` true cost matrix;
- voter identities;
- packet-loss realization;
- task order;
- tie-priority matrix;
- capacity value.

No optimizer regenerates communication loss. Parallel worker scheduling does not enter any seed formula.

## P2P information model

Robot `i` owns its own task-cost row. For physical receiver `r` and task `j`, visibility is sampled independently for the directed message:

```text
sender robot i -> receiver robot r -> task j
```

Every receiver always knows its own physical sender row. Missing physical robot/task costs are represented as:

```text
+inf
```

The heuristics treat `+inf` as unavailable. Voting Hungarian slot copies inherit the same physical visibility state, so capacity expansion never creates information that the receiver did not receive.

## Voting support and final consensus

Every valid local proposal assigns each task to one physical robot and respects the same uniform capacity.

Physical support is:

```text
S_ij = number of valid receiver proposals assigning task j to robot i
```

Final consensus expands physical support/tie rows into the same capacity-slot representation and calls the existing support-consensus owner. True task cost is not used as a hidden consensus tie-break.

The proposal-support consensus stage remains a controlled centralized boundary and must not be described as fully asynchronous decentralized consensus.

## Runtime architecture

### Receiver batching

Receivers are streamed in bounded batches inside each trial. Batch size changes memory/runtime only; it does not alter selected voters, visibility samples, proposals, or final support for a fixed configuration.

### Trial process parallelism

Independent `(task_count, trial)` jobs are parallelized with:

```text
ProcessPoolExecutor
```

Owner boundaries:

```text
build_trial_jobs
run_trial_job
execute_trial_jobs
configure_worker_thread_environment
report_trial_completion
```

Default worker count is:

```text
min(4, available CPU count)
```

`--workers 1` provides serial diagnostic mode. Worker count is runtime-only and must not change results because every trial is fully seeded before runtime scheduling.

When process parallelism is used, child BLAS/OpenMP thread environment variables are defaulted to one thread when the user has not explicitly set them. This avoids process x BLAS oversubscription.

Parallel mode reports completed trials from the parent process instead of interleaving receiver-level progress from multiple workers.

## Zero-loss integration gates

Preflight is bounded and separated by optimizer family:

```text
validate_zero_loss_heuristic_contracts
validate_zero_loss_hungarian_contract
validate_zero_loss_optimizer_contract
```

The heuristic gate requires all three fast heuristics to return valid capacity-feasible proposals with complete information at bounded task loads up to 200.

The Voting Hungarian gate additionally requires complete-information receiver-local Hungarian cost to match the capacitated Hungarian Oracle at bounded task loads up to 200 within the existing exact numerical tolerance.

Heuristics are not incorrectly required to equal the Oracle.

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

Supporting CSV metrics remain:

- optimal-cost match;
- near-optimal within 5%;
- exact optimal physical assignment;
- valid local proposal rate.

The `<=5%` measure is supporting only and is not the main paper figure.

Generated figures are line-only. Exactly identical curves may share one legend entry; near-overlapping but non-identical curves remain separate.

The x-axis is:

```text
Task batch size (100 robots fixed)
```

## Outputs

Authoritative Experiment 2 output root remains:

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

Raw/summary records include:

```text
robots
voters
tasks
capacity_per_robot
assignment_slots
method
method_label
```

## Canonical run and timing probes

First pull the final code:

```bash
git pull
```

Fast smoke with all four report-facing Voting methods:

```bash
time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 100 500 1000 \
  --trials 1 \
  --max-voters 5 \
  --workers 1
```

Parallel timing preview with all 100 voters:

```bash
time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 100 500 1000 \
  --trials 2 \
  --workers 4
```

Canonical one-hour-oriented rerun:

```bash
time python run_multitask_peer_cost_all_optimizers.py
```

The no-argument command means:

```text
100 physical robots
10 task-batch points: 100..1000 by 100
20 trials per point
all 100 robots vote
30% directed P2P cost-message loss
Voting Sequential Greedy
Voting Global Greedy
Voting Static Regret-2 Greedy
Voting Hungarian
Hungarian Oracle as full-information minimum reference
up to 4 independent trial worker processes
```

The one-hour target remains a runtime objective, not a guarantee. The user's Mac should rerun the three-point all-voter timing preview after this exact-method swap before starting the full 20-trial run.

## Interpretation and paper-reference boundary

The report should describe Experiment 2 as **fixed-fleet workload scaling under incomplete peer cost information and Voting aggregation**.

MILP, ACO, and Auction results belong to prior/supporting optimizer studies unless a new explicit experiment is defined later. Do not imply that the new canonical lossy workload curve evaluated those solvers.

For the paper bibliography, cite the Hungarian assignment reference for both the full-information Oracle and Voting Hungarian. They are two information/aggregation conditions using the same underlying assignment algorithm. The three greedy variants are explicitly defined experiment baselines in this project; do not attach an unrelated optimizer citation merely because their names contain `Greedy` or `Regret-2`.
