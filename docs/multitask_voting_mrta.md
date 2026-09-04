# Multi-Task Lossy P2P Voting Optimizer Comparison

This is the canonical report-facing multi-task experiment. It compares five optimizer families inside the same lossy P2P Voting pipeline.

## Research question

With a fixed fleet of 100 robots and 30% independent directed P2P task-cost packet loss, how does increasing simultaneous task load affect Voting-based assignment quality across different optimizer families?

Every task-count/method point is summarized over 100 paired trials.

## Canonical owner

```text
run_multitask_peer_cost_all_optimizers.py
```

The historical three-optimizer owner remains:

```text
run_multitask_peer_cost_experiment.py
```

Its Greedy/Hungarian/Auction scenario RNG schedule is deliberately preserved by the canonical owner, so those three columns should reproduce when seed, task counts, packet loss, and trial count are unchanged.

## Controlled settings

- Robots: `100`
- Directed P2P scalar task-cost packet loss: `30%`
- Trials per task-count/method point: `100`
- Simultaneous task counts: `5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100`
- Robot capacity: one simultaneous task per robot
- Task delivery: reliable
- Final proposal collection for the controlled Voting stage: reliable/in-window
- Route planning, execution noise, retransmission, permanent robot failure, deadlines, Robust Optimization, and multi-objective costs are excluded.

## Scalar cost model

Each trial samples robot and task positions in one normalized 2-D workspace.

```text
C_ij = 0.05 + distance(robot_i, task_j)
```

This scalar spatial cost is the only optimization objective in this report cycle.

## Paired scenario contract

For every `(task_count, trial)` pair, the experiment generates exactly once:

- one true robot-by-task cost matrix;
- one directed P2P visibility tensor;
- one task order;
- one consensus tie-priority matrix.

The same objects are reused by every optimizer family.

The legacy task-count RNG schedule is preserved:

```text
rng = default_rng(seed + task_count * 100003)
```

Each trial consumes that shared stream only in this order:

```text
generate cost matrix
generate packet-loss visibility tensor
generate task order
generate tie priority
```

No optimizer consumes this scenario RNG afterward.

Therefore the fairness contract is:

```text
same trial
-> same robot/task geometry
-> same true cost matrix
-> same packet-loss realization
-> same task order / tie priority
-> different optimizer only
```

## P2P information model

Robot `i` computes only its own task-cost row. For every task `j`, sender `i` attempts to transmit that scalar cost directly to receiver `r`:

```text
sender i -> receiver r -> task j
```

Each directed scalar delivery is an independent Bernoulli event. Every robot always knows its own row.

A receiver reconstructs its own incomplete robot-by-task cost matrix. Missing entries are represented as:

```text
+inf
```

at the optimizer boundary and are unavailable assignment edges.

## Compared methods

The report-facing methods are:

```text
Hungarian Oracle
Voting Greedy
Voting Hungarian
Voting Auction
Voting MILP
Voting ACO + Local Search
```

### Hungarian Oracle

Full-information centralized minimum-total-cost assignment. It is the evaluation reference only and is unaffected by packet loss.

### Voting Greedy

Each receiver runs the existing sequential Greedy assignment on its incomplete matrix. Greedy is a heuristic baseline and is not expected to be globally optimal for multiple tasks.

### Voting Hungarian

Each receiver solves the exact linear assignment problem on its own incomplete matrix using the existing Hungarian owner.

### Voting Auction

Each receiver solves the same local assignment objective using the existing epsilon-scaling Bertsekas-style Auction owner.

The existing epsilon schedule remains:

```text
1e-2 -> 1e-3 -> 1e-4 -> 1e-5 -> 1e-6 -> 1e-7 -> 1e-8
```

When task count is below robot count, zero-cost dummy tasks are used internally exactly as in the preceding Auction implementation.

### Voting MILP

Each receiver solves its incomplete capacity-one assignment as a binary MILP using the existing `solve_milp_assignment` owner.

For every missing `+inf` edge, the corresponding binary decision variable is forbidden:

```text
x_ij <= 0
```

No large artificial objective penalty is substituted for missing information.

The existing HiGHS setting remains:

```text
mip_rel_gap = 0.0
```

MILP retains its solver-specific objective-equivalence tolerance:

```text
MILP_NUMERICAL_TOLERANCE_PERCENT = 1e-6
```

### Voting ACO + Local Search

Each receiver uses the existing ACO + Local Search owner with the same fixed search budget used in optimizer screening:

```text
ants = 12
iterations = 15
alpha = 1.0
beta = 3.0
evaporation = 0.20
candidate_list_size = 20
elite_weight = 2.0
local_search_moves = 25
```

Unavailable `+inf` edges are excluded from ACO candidate sets. If the Greedy seed cannot form a complete assignment from an incomplete view, ACO may continue its ant search rather than immediately declaring the receiver invalid.

ACO randomness is independent from the scenario RNG. Receiver `r` receives the deterministic seed:

```text
seed
+ 7000003
+ task_count * 100003
+ trial * 1009
+ receiver * 10000019
```

Changing execution order therefore does not change the ACO random stream assigned to any receiver.

## Local proposal validity

Every valid receiver proposal must satisfy:

- one robot index for every task;
- robot indices in range;
- no robot assigned to two simultaneous tasks;
- every selected robot-task edge is finite in that receiver's local matrix.

MILP/ACO outputs are checked at `validate_local_proposal` before they enter Voting support.

A solver returning no complete feasible local assignment marks that receiver proposal invalid. It is not repaired with the Hungarian Oracle and no missing edge is restored.

## Shared Voting / consensus boundary

Every valid local proposal is a complete map:

```text
task j -> robot i
```

Proposal support is:

```text
S_ij = number of valid receiver proposals assigning task j to robot i
```

The final team assignment maximizes total proposal support subject to:

```text
sum_i x_ij = 1   for every task j
sum_j x_ij <= 1  for every robot i
```

A tiny paired random priority is used only to break equal-support assignments. True cost is not used as a hidden consensus tie-break.

## Receiver-local runtime execution

MILP and ACO are much more expensive than the previous Greedy/Hungarian/Auction paths because each of the 100 receiver-local incomplete matrices requires its own solve.

The canonical owner therefore parallelizes only these independent receiver-local MILP/ACO solves with a persistent `ProcessPoolExecutor`.

Default runtime control:

```text
workers = min(4, available CPU count)
```

This parallelism changes execution scheduling only. It does not change:

- the cost matrix;
- packet-loss realization;
- task order;
- tie priority;
- optimizer parameters;
- ACO per-receiver seed;
- proposal validation;
- support counts;
- Voting consensus.

Worker completion order is irrelevant because every returned proposal is written back to its original receiver index before support is built.

A serial regression mode remains available:

```bash
python run_multitask_peer_cost_all_optimizers.py --workers 1
```

The serial and parallel modes are required to be decision-equivalent for fixed seed/configuration.

The default command prints liveness for MILP/ACO every 25 completed receiver solves. The interval can be changed with:

```text
--progress-every-receivers N
```

Runtime-control validation failures are owned by `validate_parallel_config`. Unexpected worker/pool failures are reported at the runtime boundary, for example:

```text
owner=run_multitask_peer_cost_all_optimizers
function=solve_parallel_receiver_proposals
category=runtime
code=PARALLEL_RECEIVER_SOLVE_FAILED
```

No optimizer failure is hidden by this boundary: named `ValueError` diagnostics from the real solver owner are propagated unchanged.

## Zero-loss contract

Before the formal 30% loss sweep, `validate_zero_loss_optimizer_contract` checks task counts `1, 5, 50, 100`.

At complete information:

- Voting Hungarian must match the Hungarian Oracle cost;
- Voting Auction must match the Hungarian Oracle cost;
- Voting MILP must match the Hungarian Oracle cost within the MILP numerical tolerance;
- single-task Voting Greedy must match the Oracle;
- all proposals for the exact methods above must be valid.

ACO is intentionally not required to match the Oracle because it is a stochastic metaheuristic.

## Evaluation metrics

### Average optimality gap

```text
gap (%) = 100 * (method_cost - oracle_cost) / oracle_cost
```

Lower is better.

### Optimal-cost match

Percentage of the 100 trials where the final assignment reaches the Oracle cost within the numerical tolerance owned by that optimizer family.

### Near-optimal within 5%

Percentage of trials satisfying:

```text
gap <= 5%
```

### Exact optimal assignment

Percentage of trials whose complete task-to-robot assignment exactly equals the Hungarian Oracle assignment.

### Valid local proposal rate

Percentage of the 100 receiver-local solvers that returned a complete feasible proposal. This is diagnostic and must be interpreted together with final quality.

## Canonical run

First run a smoke check with parallel execution:

```bash
python run_multitask_peer_cost_all_optimizers.py --tasks 5 20 100 --trials 3
```

For a serial-vs-parallel regression on a small case:

```bash
python run_multitask_peer_cost_all_optimizers.py --tasks 5 --trials 1 --workers 1
python run_multitask_peer_cost_all_optimizers.py --tasks 5 --trials 1 --workers 4
```

Formal report run:

```bash
python run_multitask_peer_cost_all_optimizers.py
```

The formal run uses 100 robots, 30% packet loss, all canonical task counts, 100 paired trials per point, and the default receiver-parallel runtime configuration.

## Outputs

```text
results/multitask_peer_cost_all_optimizers/data/
results/multitask_peer_cost_all_optimizers/figures/
```

Raw and summary data:

```text
optimizer_comparison_raw.csv
optimizer_comparison_summary.csv
```

Report-ready CSVs:

```text
report_average_optimality_gap_percent.csv
report_optimal_cost_match_percent.csv
report_near_optimal_5pct_percent.csv
report_exact_optimal_assignment_percent.csv
report_valid_proposal_rate_percent.csv
```

## Report interpretation boundary

This experiment supports statements about optimizer-family behavior inside the same controlled lossy P2P Voting pipeline.

It does not by itself prove that the gain is caused by Voting. Causal Direct-vs-Voting claims remain owned by `run_multitask_voting_ablation.py`.

The current Direct-vs-Voting ablation still covers Hungarian and Auction only. Extending that ablation to additional optimizer families is a separate bounded experiment.

The shared proposal-support consensus remains a controlled centralized boundary, so this experiment must not be described as fully asynchronous decentralized consensus.
