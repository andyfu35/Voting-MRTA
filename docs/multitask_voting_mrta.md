# Multi-Task Peer-Cost Optimizer Comparison

This is the canonical controlled multi-task experiment that follows the single-task peer-cost packet-loss study.

## Research question

With a fixed fleet of 100 robots and 30% independent directed P2P task-cost packet loss, how does increasing the number of simultaneous tasks affect assignment quality for real assignment optimizers?

Every task-count/method data point is summarized over 100 paired trials.

## Controlled experiment settings

- Robots: `100`
- Directed P2P task-cost packet loss: `30%`
- Trials per task-count/method point: `100`
- Simultaneous task counts: `5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100`
- Robot capacity: one simultaneous task per robot
- Task delivery: reliable
- Final assignment-proposal collection: reliable/in-window in this controlled stage
- Route planning, execution noise, retransmission, permanent robot failure, and task deadlines are excluded.

## Cost owner

Each trial samples robot and task positions in one normalized 2-D workspace. The ground-truth cost is Euclidean robot-to-task travel distance plus a small positive floor:

```text
C_ij = 0.05 + distance(robot_i, task_j)
```

Every method is evaluated against exactly the same true cost matrix inside a paired trial.

## P2P communication model

Robot `i` computes only its own row of task costs. For each task `j`, sender `i` sends that scalar cost directly to every receiver `r`.

```text
sender i -> receiver r -> task j
```

Each directed scalar delivery is an independent Bernoulli event. Every robot always knows its own task-cost row.

The receiver therefore reconstructs an incomplete robot-by-task cost matrix. Missing cost entries are unavailable to that receiver and are represented as `+inf` at the optimizer boundary.

## Compared methods

### Hungarian Oracle

Full-information centralized minimum-total-cost assignment. This is the ground-truth optimal reference and is not affected by packet loss.

### P2P Hungarian

Every receiver solves the exact linear assignment problem on its own incomplete P2P cost matrix using the Hungarian/linear-assignment solver. The receiver proposes one complete capacity-one task assignment.

### P2P Auction

Every receiver solves the same local assignment objective using a Bertsekas-style auction algorithm with iterative prices and bids. Missing cost edges cannot be bid on. The implementation is batched across receivers but preserves an independent price/ownership state per receiver.

### P2P Greedy

Heuristic baseline. Every receiver processes the paired task order sequentially and assigns each task to the cheapest still-available robot in its incomplete view. Greedy is not claimed to be globally optimal for multiple tasks.

## Shared proposal-to-consensus boundary

Every valid receiver produces one complete assignment proposal:

```text
task j -> robot i
```

Proposal support is counted as:

```text
S_ij = number of receiver assignment proposals assigning task j to robot i
```

The final team assignment maximizes total proposal support subject to:

```text
sum_i x_ij = 1   for every task j
sum_j x_ij <= 1  for every robot i
```

A tiny paired random priority is used only to break equal-support assignments. True task cost is not used as a hidden consensus tie-break.

## Zero-loss optimizer contract

Before the 30% packet-loss sweep starts, the script runs a mandatory complete-information contract check.

- P2P Hungarian must match the Hungarian Oracle total cost.
- P2P Auction must match the Hungarian Oracle total cost.
- For the single-task case, P2P Greedy must also select the oracle minimum-cost robot.
- All local proposals must be valid at zero packet loss.

If any exact optimizer violates this contract, the experiment aborts with the first failing owner/function/category/code diagnostic.

This requirement exists specifically to prevent the earlier probabilistic single-vote failure mode where a method could miss the known optimum even with complete information.

## Rejected previous policy comparison

The prior `Inverse`, `Softmax`, and `Rank` methods sampled one candidate from a probability distribution. Those experiments are retained only as historical screening data. They are not treated as optimization methods because stochastic vote dispersion can fail to choose the known optimum at 0% packet loss.

They are therefore removed from the canonical optimizer comparison.

## Evaluation metrics

### Average optimality gap

```text
gap (%) = 100 * (method_cost - oracle_cost) / oracle_cost
```

Lower is better.

### Optimal-cost match

Percentage of the 100 trials where the final assignment reaches the same total cost as the Hungarian Oracle within numerical tolerance.

This is preferred over exact assignment identity when two different assignments have the same optimum cost.

### Near-optimal within 5%

Percentage of trials where:

```text
gap <= 5%
```

### Exact optimal assignment

Percentage of trials where the complete task-to-robot assignment exactly equals the Hungarian Oracle assignment.

### Valid local proposal rate

Percentage of the 100 receivers that could construct a complete capacity-one local assignment from their incomplete P2P view. This is a diagnostic metric, not the primary quality metric.

## Run

```bash
python run_multitask_peer_cost_experiment.py
```

Default configuration is 100 robots, 30% packet loss, all task counts above, and 100 trials per point.

For a smoke test:

```bash
python run_multitask_peer_cost_experiment.py --tasks 5 20 --trials 3
```

For an explicit no-loss verification run:

```bash
python run_multitask_peer_cost_experiment.py --tasks 5 50 100 --trials 10 --packet-loss 0
```

## Outputs

```text
results/multitask_peer_cost/data/
results/multitask_peer_cost/figures/
```

Report-ready CSVs:

```text
report_average_optimality_gap_percent.csv
report_optimal_cost_match_percent.csv
report_near_optimal_5pct_percent.csv
report_exact_optimal_assignment_percent.csv
report_valid_proposal_rate_percent.csv
```

Raw and summary data:

```text
optimizer_comparison_raw.csv
optimizer_comparison_summary.csv
```

## Scope boundary

This experiment compares assignment optimizers under receiver-specific incomplete P2P cost matrices and a shared proposal-consensus boundary. It is not yet a fully asynchronous CBBA implementation. A later experiment can replace the proposal-consensus boundary with a true asynchronous decentralized auction/bundle protocol without changing the cost, communication, or evaluation owners defined here.
