# Fixed-100-Robot Lossy P2P Voting Workload Experiment

This is the canonical report-facing Experiment 2 for the current one-page paper cycle.

## Research question

Under `30%` independent directed P2P scalar task-cost packet loss, how does Voting-based task allocation quality change as task workload grows while the physical fleet remains fixed at `100` robots?

Canonical workload:

```text
robot_count = 100
tasks = 100, 200, 300, ..., 1000
capacity_per_robot = ceil(tasks / 100)
```

Task counts are allocation batches/workload, not a claim that one physical robot executes all assigned tasks simultaneously.

## Canonical owners

Experiment routing, communication, Voting, evaluation, and runtime:

```text
run_multitask_peer_cost_all_optimizers.py
```

Single Greedy baseline:

```text
run_multitask_workload_heuristics.py::solve_sequential_greedy_capacitated
```

Min-Cost Flow and Sinkhorn:

```text
run_multitask_workload_optimizers.py
```

Hungarian assignment algorithm:

```text
run_multitask_peer_cost_experiment.py::solve_hungarian_assignment
```

The Experiment 2 adapter does not duplicate these optimizer implementations.

## Canonical compared methods

The report-facing Voting methods are deliberately from four different optimization families:

```text
Voting Greedy
Voting Hungarian
Voting Min-Cost Flow
Voting Sinkhorn + Rounding
```

`Hungarian Oracle` remains the full-information minimum-cost reference. It is retained in CSV output but is not plotted as a fifth quality curve; `0%` on the cost-error axis is the reference minimum.

The main figure therefore normally contains four optimizer curves. Exactly identical measured curves may be merged only under the existing exact-overlap plotting rule.

The previous Global Greedy and Static Regret-2 Greedy implementations remain in the repository for supporting studies but are no longer canonical Experiment 2 methods. Auction, MILP, and ACO also remain implemented but are not part of this lossy workload sweep.

## Method definitions

### Voting Greedy

For each receiver-local incomplete physical `100 x T` cost matrix, tasks are processed in the paired trial task order and assigned to the cheapest finite-cost robot with remaining capacity.

Owner:

```text
run_multitask_workload_heuristics.py::solve_sequential_greedy_capacitated
```

This is the only Greedy curve in the canonical figure.

### Voting Hungarian

Each receiver obtains its incomplete physical `100 x T` cost matrix. The Experiment 2 adapter expands physical robot rows into capacity slots, calls the existing Hungarian owner, maps slot IDs back to physical robots, and contributes that proposal to Voting support.

Routing:

```text
run_multitask_peer_cost_all_optimizers.py::solve_voter_batch_proposals
    -> run_multitask_peer_cost_experiment.py::solve_local_optimizer_proposals
    -> run_multitask_peer_cost_experiment.py::solve_hungarian_assignment
```

`Voting Hungarian` is not the same condition as `Hungarian Oracle`:

```text
Hungarian Oracle = full true cost matrix, one reference solve per trial
Voting Hungarian = incomplete receiver-local solve per voter, then Voting consensus
```

### Voting Min-Cost Flow

Owner:

```text
run_multitask_workload_optimizers.py::solve_min_cost_flow_capacitated
```

The receiver-local physical problem is represented directly as a capacitated network:

```text
source -> robot_i -> task_j -> sink
```

with:

```text
capacity(source, robot_i) = capacity_per_robot
capacity(robot_i, task_j) = 1 for finite receiver-visible edges
capacity(task_j, sink) = 1
```

The implementation uses OR-Tools `SimpleMinCostFlow` through the project dependency `ortools`.

OR-Tools requires integer arc costs. Receiver-visible float costs are therefore deterministically scaled and rounded at the named boundary:

```text
MIN_COST_FLOW_COST_SCALE = 1_000_000
run_multitask_workload_optimizers.py::quantize_min_cost_flow_costs
```

All report metrics are still evaluated on the original unquantized float costs. Min-Cost Flow is exact for the integer-scaled network objective and is required by preflight to agree with the float Hungarian Oracle within:

```text
MIN_COST_FLOW_ORACLE_TOLERANCE_PERCENT = 0.01
```

No capacity-slot expansion is used for Min-Cost Flow.

### Voting Sinkhorn + Rounding

Owner functions:

```text
run_multitask_workload_optimizers.py::compute_sinkhorn_transport_plan
run_multitask_workload_optimizers.py::round_sinkhorn_plan_to_capacity
run_multitask_workload_optimizers.py::solve_sinkhorn_capacitated
```

Sinkhorn is a soft entropic optimal-transport approximation on the receiver-local physical `100 x T` matrix. Missing costs have zero transport kernel weight.

Canonical configuration:

```text
epsilon = 0.08
max_iterations = 30
tolerance = 1e-5
```

The soft row target is `tasks / 100`, which equals the physical capacity at the canonical task points because every point is a multiple of 100. Each task has unit column mass.

The continuous transport plan is not itself a robot assignment, so discretization is an explicit separate boundary. `round_sinkhorn_plan_to_capacity` prioritizes scarce/high-confidence tasks and chooses the highest-plan finite robot that still has capacity. It never exceeds the physical capacity and never uses hidden full-information costs.

Sinkhorn is not claimed to be an exact discrete assignment solver. Its preflight requires feasibility, not Oracle equality.

## Controlled settings

- Physical robots: `100` fixed.
- Task batches: `100..1000` in steps of `100`.
- Directed P2P scalar cost-message loss: `30%`.
- Uniform per-robot batch capacity: `ceil(tasks/100)`.
- Canonical report trials per task point: `20`.
- Canonical mode uses all `100` physical robots as voters.
- Default process workers: up to `4`, bounded by available CPU count.
- Default receiver batch size: `4`.
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

Separate deterministic streams derive scenario geometry/task order/tie priority, optional voter selection, and packet-loss visibility.

All four Voting methods receive the same:

- physical `100 x T` true cost matrix;
- voter identities;
- directed packet-loss realization;
- task order;
- tie-priority matrix;
- capacity value.

No optimizer regenerates communication loss. Parallel worker scheduling does not enter any seed formula.

## P2P information model

Robot `i` owns its own task-cost row. For receiver `r` and task `j`, the directed scalar message `i -> r` is independently lost according to the fixed packet-loss rate. Every receiver always knows its own sender row.

Missing receiver-local costs are represented as:

```text
+inf
```

Greedy, Hungarian, Min-Cost Flow, and Sinkhorn all treat those edges as unavailable. Capacity expansion for Hungarian never creates visibility that the receiver did not receive.

## Voting support and final consensus

Every valid local proposal assigns each task to one physical robot and obeys the same uniform physical capacity.

Physical support is:

```text
S_ij = number of valid receiver proposals assigning task j to robot i
```

Final consensus expands support/tie rows into capacity slots and calls the existing support-consensus owner. True task cost is not used as a hidden consensus tie-break.

The proposal-support consensus stage remains a controlled centralized boundary and must not be described as fully asynchronous decentralized consensus.

## Runtime architecture

Independent `(task_count, trial)` jobs use `ProcessPoolExecutor`.

Canonical default:

```text
workers = min(4, available CPU count)
receiver batch size = 4
```

Worker count and receiver batch size are runtime controls only. They must not change scenario seeds, packet-loss samples, proposals, or final assignments for a fixed configuration.

BLAS/OpenMP thread-count environment variables default to one when process parallelism is enabled unless the user explicitly set them.

## Zero-loss integration gates

Preflight uses separate named boundaries:

```text
validate_zero_loss_greedy_contract
validate_zero_loss_hungarian_contract
validate_zero_loss_min_cost_flow_contract
validate_zero_loss_sinkhorn_contract
validate_zero_loss_optimizer_contract
```

The requirements are:

- Greedy: valid capacity-feasible complete-information proposal;
- Hungarian: equal to the capacitated Oracle within the existing exact tolerance;
- Min-Cost Flow: equal to the float Oracle within the explicit integer-quantization tolerance;
- Sinkhorn + Rounding: valid capacity-feasible complete-information proposal.

The bounded preflight task size is at most `200` for each family.

## Primary report metric

The main Experiment 2 figure uses:

```text
Cost error (%) = 100 * (method_cost - oracle_cost) / oracle_cost
```

Lower is better. `0%` means the method reaches the full-information capacitated minimum under the same physical capacity contract.

The CSV field is:

```text
average_optimality_gap_percent
```

Supporting CSV metrics remain optimal-cost match, near-optimal within 5%, exact physical assignment, and valid local proposal rate.

The x-axis is:

```text
Task batch size (100 robots fixed)
```

## Dependencies

The canonical four-method experiment now requires OR-Tools for Min-Cost Flow. After pulling the code, synchronize the virtual environment with:

```bash
pip install -r requirements.txt
```

If OR-Tools is missing, execution fails at the explicit dependency boundary:

```text
owner=run_multitask_workload_optimizers
function=require_min_cost_flow_dependency
category=dependency
code=ORTOOLS_NOT_AVAILABLE
```

## Outputs

Authoritative output root:

```text
results/multitask_peer_cost_fixed100_workload/
```

Primary files:

```text
data/workload_comparison_raw.csv
data/workload_comparison_summary.csv
figures/average_optimality_gap_percent.png
```

## Rerun sequence

Install/update dependencies first:

```bash
git pull
pip install -r requirements.txt
```

Small dependency/function smoke:

```bash
time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 100 500 1000 \
  --trials 1 \
  --max-voters 5 \
  --workers 1
```

All-voter timing preview:

```bash
time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 100 500 1000 \
  --trials 2 \
  --workers 4
```

If the timing preview is acceptable, canonical Experiment 2 is:

```bash
time python run_multitask_peer_cost_all_optimizers.py
```

No-argument meaning:

```text
100 robots
100..1000 tasks by 100
20 trials per point
100 voters
30% directed packet loss
Voting Greedy
Voting Hungarian
Voting Min-Cost Flow
Voting Sinkhorn + Rounding
Hungarian Oracle reference
up to 4 trial worker processes
```

The approximately one-hour runtime remains a target that must be measured on the user's Mac; it is not guaranteed by specification.

## Interpretation and paper-reference boundary

The report should describe Experiment 2 as fixed-fleet workload scaling under incomplete peer-cost information and Voting aggregation.

The paper bibliography must include method-appropriate references when the final Experiment 2 figure is inserted: Hungarian assignment for Oracle/Voting Hungarian, a standard minimum-cost-flow reference for Voting Min-Cost Flow, and an entropic optimal-transport/Sinkhorn reference for Voting Sinkhorn. The Greedy baseline is explicitly defined by this experiment.
