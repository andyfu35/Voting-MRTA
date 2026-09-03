# Multi-Task Peer-Cost Voting-MRTA

This is the canonical controlled multi-task experiment that follows the single-task peer-cost packet-loss study.

## Research question

With a fixed fleet of 100 robots and 30% independent directed P2P task-cost packet loss, how does increasing the number of simultaneous tasks affect assignment quality for different allocation/voting methods?

Every task-count/method data point is summarized over 100 paired trials.

## Controlled experiment settings

- Robots: `100`
- Directed P2P task-cost packet loss: `30%`
- Trials per task-count/method point: `100`
- Simultaneous task counts: `5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100`
- Robot capacity: one simultaneous task per robot
- Task delivery: reliable
- Final vote collection: reliable/in-window in this controlled stage
- Route planning, execution noise, retransmission, permanent robot failure, and task deadlines are excluded.

## Cost owner

Each trial samples robot and task positions in one normalized 2-D workspace. The shared ground-truth cost is Euclidean robot-to-task travel distance plus a small positive floor:

```text
C_ij = 0.05 + distance(robot_i, task_j)
```

Every method is evaluated against exactly the same true cost matrix inside a paired trial.

## P2P communication model

Robot `i` computes only its own row of task costs. For every task `j`, the scalar cost message from sender `i` to receiver `r` is an independent directed Bernoulli delivery event.

Therefore:

```text
sender i -> receiver r -> task j
```

can be delivered while the reverse direction or another task-cost message is lost. Every robot always knows its own task costs.

This produces a different incomplete candidate-cost view for each receiver and each task.

## Compared methods

### Hungarian

Full-information centralized minimum-total-cost assignment. This is the optimal reference, not the proposed decentralized method.

### Sequential Greedy

Full-information heuristic baseline. Tasks are processed in task-index order; each task takes the cheapest robot that remains available.

### Greedy

For every task, each receiver votes for the cheapest candidate visible in its own incomplete P2P cost view.

### Inverse

Each receiver converts visible task costs to inverse-cost weights and samples exactly one candidate vote. The parameter selected from the preceding single-task tuning experiment is used internally; report tables show only `Inverse`.

### Softmax

Each receiver locally normalizes visible costs, converts them to Boltzmann/softmax weights, and samples exactly one candidate vote. The selected parameter is internal; report tables show only `Softmax`.

### Rank

Each receiver ranks only the candidates it actually sees, converts rank to inverse-rank weights, and samples exactly one candidate vote. The selected parameter is internal; report tables show only `Rank`.

The selected single-task parameters are fixed before this multi-task evaluation rather than re-tuned separately for every task count.

## Shared vote-to-assignment boundary

The four P2P voting methods produce a candidate-by-task vote-support matrix:

```text
V_ij = number of receiver votes supporting robot i for task j
```

A shared capacity-one matching owner then solves:

```text
maximize sum_ij V_ij x_ij
```

subject to:

```text
sum_i x_ij = 1   for every task j
sum_j x_ij <= 1  for every robot i
```

This support matching is common to all four voting policies so the experiment isolates the effect of the local cost-to-vote method. True task cost is not used as a hidden secondary tie-break; a tiny paired random priority resolves equal-support assignments only.

## Evaluation metrics

### Average optimality gap

```text
gap (%) = 100 * (method_cost - Hungarian_cost) / Hungarian_cost
```

Lower is better. Hungarian is always `0%` by definition.

### Near-optimal within 5%

Percentage of the 100 trials where:

```text
gap <= 5%
```

Higher is better.

### Exact optimal assignment

Percentage of trials where the complete robot-task assignment exactly matches the Hungarian assignment.

This is intentionally strict and may approach zero as task count rises even when the total cost remains close to optimal.

## Run

```bash
python run_multitask_peer_cost_experiment.py
```

The default run uses the full agreed experiment configuration. Optional smoke tests can reduce task counts or trials:

```bash
python run_multitask_peer_cost_experiment.py --tasks 5 50 100 --trials 10
```

## Outputs

```text
results/multitask_peer_cost/data/
results/multitask_peer_cost/figures/
```

Report-ready CSV tables contain parameter-free method labels:

```text
report_average_optimality_gap_percent.csv
report_near_optimal_5pct_percent.csv
report_exact_optimal_assignment_percent.csv
```

## Scope boundary

This experiment is not yet a full asynchronous CBBA/auction protocol. It first tests whether the cost-to-vote policies that were characterized in the single-task experiment remain useful when simultaneous tasks introduce robot-capacity conflicts. A separate later experiment can add true decentralized auction/bundle negotiation without changing this controlled baseline.
