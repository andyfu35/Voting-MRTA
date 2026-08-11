# Multi-Task Voting-MRTA

This experiment extends Voting-MRTA from one task to multiple simultaneous tasks while keeping one shared cost definition across all voting policies.

## Architecture

The multi-task pipeline is intentionally separated into four stages:

1. **Local cost evaluation** - every robot/task pair uses the same cost model `C_ij`.
2. **Distributed voting** - active robots convert the shared cost information into votes using a voting policy.
3. **Vote aggregation** - the terminal receives the vote matrix under the existing packet-loss/retransmission model.
4. **Constraint-aware global assignment** - the terminal chooses a feasible robot-task matching from the vote support matrix.

For the first experiment, `C_ij` is Euclidean robot-to-task travel distance plus a small positive floor. The numerical costs change with robot/task positions, but every method uses the same formula and the same cost matrix inside a paired trial.

## Why assignment is separate from cost

A robot can correctly report that it is the cheapest candidate for several tasks at the same time. However, when those tasks are allocated simultaneously, independent per-task winners can overload the same robot.

The first multi-task experiment therefore does **not** pretend that a robot knows which other tasks it will receive in the future. Current task cost and future assignment load are kept separate.

With binary assignment variable `x_ij`:

```text
x_ij = 1 if robot i is assigned task j, otherwise 0
```

The terminal enforces:

```text
sum_i x_ij = 1      for every task j
sum_j x_ij <= 1     for every active robot i
```

The second constraint is the first controlled capacity model. Later experiments can replace `1` with robot-specific capacities.

## Terminal objective

Voting methods produce a support matrix `V_ij`, where `V_ij` is the number of delivered votes supporting robot `i` for task `j`.

The terminal solves the maximum-support feasible assignment:

```text
maximize sum_ij V_ij x_ij
```

subject to the assignment constraints above.

A tiny paired random priority is used only when two feasible global assignments have exactly equal vote support. Task cost is not used as a hidden tie-breaking objective in the Voting-MRTA allocation, so the experiment measures how informative each voting policy is.

## Shared cost, different voting policies

The same `C_ij` matrix is used by:

- Inverse voting: `alpha = 1, 2, 3`
- Softmax voting: `beta = 1, 2`
- Greedy voting
- Heterogeneous voting using a balanced mix of Inverse, Softmax, and Greedy voters

This isolates the effect of the **cost-to-vote policy** from the effect of the cost model itself.

An important hypothesis is that very deterministic voting may lose useful second-choice information under multi-task conflicts. Softer probability distributions can preserve support for alternative robots, which may help the terminal construct a better feasible global assignment.

## Task pressure

The experiment uses simultaneous task-load ratios:

```text
25%, 50%, 75% of the currently active robot count
```

Each robot has capacity one in this first version. Increasing task load therefore increases competition for the same attractive robots and makes assignment conflicts more important.

## Baselines

### Centralized Optimal

Uses the true shared cost matrix directly and minimizes total assignment cost subject to the same capacity constraints. This is an optimization upper bound / lower-cost reference, not the proposed voting method.

### Sequential Greedy

Processes tasks in a paired random order and assigns each task to the cheapest robot that is still available. It always produces a feasible assignment, but early decisions can prevent a lower-cost global allocation.

### Independent minimum-cost diagnostic

Each task independently picks its cheapest robot without considering other tasks. This is not treated as a feasible allocation method. Its conflict rate measures how often a global assignment layer is necessary.

## Metrics

The experiment reports:

- Total allocation cost
- Optimality gap relative to centralized optimal
- Near-optimal allocation rate (`gap <= 5%`)
- Exact assignment match rate
- Complete-vote vs post-communication assignment preservation
- Independent minimum-cost conflict rate
- Maximum load produced by independent per-task winners
- Received vote rate after retransmission

## Run

```bash
python3 run_multitask_experiment.py
```

Outputs are written to:

```text
results/multitask/data/
results/multitask/figures/
```

## Is this decentralized?

Not fully. The architecture is best described as **distributed local evaluation and voting with centralized constraint-aware arbitration**.

The robot side is distributed because multiple robots independently participate in evaluation/voting and the system does not rely on one robot making the local preference decision for everyone. The final matching is centralized because the terminal receives the votes and solves the global assignment constraints.

A fully decentralized MRTA version would also remove the terminal as the final decision authority and require robots to reach a consistent allocation through peer-to-peer negotiation, auctions, consensus, token passing, or another distributed coordination protocol.
