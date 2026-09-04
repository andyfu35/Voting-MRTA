# Multi-Task Lossy P2P Voting Scaling Experiment

This is the canonical report-facing Experiment 2 for the current one-page paper cycle.

## Research question

Under `30%` independent directed P2P scalar task-cost packet loss, how does Democracy/Voting assignment quality change as the multi-robot system scales from tens to `1000` simultaneous tasks?

The scaling sweep keeps the capacity-one assignment semantics unchanged by using a matched fleet at every point:

```text
robot_count = task_count
```

Canonical scale points are:

```text
50, 100, 200, 400, 600, 800, 1000 robots/tasks
```

This is therefore a **system-scale** experiment at full simultaneous utilization, not a 100-robot system overloaded with more tasks than robots.

## Canonical owner

```text
run_multitask_peer_cost_all_optimizers.py
```

The filename is retained for command compatibility. The current main Experiment 2 no longer runs all previously screened optimizer families.

## Compared methods

The main scaling experiment now keeps only the methods that are practical for a large trend sweep:

```text
Hungarian Oracle
Voting Greedy
Voting Hungarian
Voting Auction
```

`MILP` and `ACO + Local Search` are removed from the main Experiment 2 sweep because their per-receiver local solves made the report-facing run prohibitively slow. Their earlier complete-information screening remains supporting optimizer-characterization evidence; removing them from this scaling experiment does not alter their implementations.

Hungarian and Auction remain separate result columns even if they overlap numerically.

## Controlled settings

- Directed P2P scalar task-cost packet loss: `30%`.
- Robot capacity: one simultaneous task per robot.
- Matched scale: `robot_count == task_count`.
- Canonical task/robot counts: `50, 100, 200, 400, 600, 800, 1000`.
- Formal trials per scale point: `100` unless a later report decision explicitly changes this contract.
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

One true cost matrix and one task order/tie-priority realization are generated once.

All Voting methods then receive the same selected voter identities and the same directed packet-loss realization for those voters.

Separate deterministic random streams are used for:

```text
scenario geometry / task order / tie priority
voter selection
packet-loss visibility
```

so changing the preview voter cap does not regenerate optimizer-specific scenarios.

No optimizer samples its own communication realization.

## P2P information model

Robot `i` owns its own task-cost row. For receiver `r` and task `j`, directed scalar visibility is sampled independently:

```text
sender i -> receiver r -> task j
```

Every receiver always knows its own sender row.

Missing entries remain unavailable and are represented as:

```text
+inf
```

at the local optimizer boundary.

## Scalable receiver batching

A direct `1000 receivers x 1000 senders x 1000 tasks` tensor is too large for a laptop-scale experiment. Experiment semantics are therefore preserved while memory ownership is changed.

The canonical owner processes voting receivers in bounded batches:

```text
DEFAULT_VOTER_BATCH_SIZE = 8
```

For one voter batch:

1. sample that batch's directed visibility;
2. materialize only that batch's incomplete cost views;
3. run Greedy, Hungarian, and Auction on the **same** batch views;
4. accumulate each method's proposal-support matrix;
5. discard the batch views before the next batch.

The final support matrix is exactly the sum of batch supports. Batch size changes memory/runtime only; it does not change voter identities, visibility sequence, proposals, support totals, or consensus for a fixed configuration.

## Voting support / consensus

Each valid receiver proposal is a complete capacity-one task assignment.

Support is accumulated as:

```text
S_ij = number of valid receiver proposals assigning task j to robot i
```

The final assignment maximizes total support subject to:

```text
sum_i x_ij = 1   for every task j
sum_j x_ij <= 1  for every robot i
```

The paired tie-priority matrix is used only for equal-support ties. True cost is not used as a hidden consensus tie-break.

A voter batch with zero valid proposals is allowed and contributes zero support. If an entire method has zero valid proposals across all selected voters, Experiment 2 fails at:

```text
owner=run_multitask_peer_cost_all_optimizers
function=finalize_voting_assignments
category=planning
code=NO_VALID_PROPOSALS
```

## Full-voter mode vs trend-preview mode

The canonical behavior uses **all robots as voting receivers**:

```bash
python run_multitask_peer_cost_all_optimizers.py
```

For immediate trend inspection, a receiver sample can be explicitly capped:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --tasks 50 100 200 400 600 800 1000 \
  --trials 10 \
  --max-voters 100
```

`--max-voters 100` means:

- use the whole fleet for scales up to 100;
- randomly select 100 receiver identities for larger fleets;
- all compared optimizers use that exact same voter sample and packet-loss views.

This is a **trend preview**, not canonical full-voter report data. Preview CSVs must not be silently mixed with full-voter formal data.

## Zero-loss contract

Before the lossy sweep, the owner checks:

- single-task Greedy against the Hungarian Oracle;
- Voting Hungarian against the Oracle at representative matched scales;
- Voting Auction against the Oracle at representative matched scales, including the largest configured scale.

Failures identify the first owner/function/category/code boundary.

## Primary metric for the one-page paper

The primary Experiment 2 curve is now:

```text
Trials within 5% of optimum (%)
```

where:

```text
gap (%) = 100 * (method_cost - oracle_cost) / oracle_cost
near-optimal = gap <= 5%
```

Higher is better.

The script still records supporting metrics:

- average optimality gap;
- optimal-cost match;
- exact optimal assignment;
- valid local proposal rate.

Generated report plots are line-only (no point markers), and the Oracle is retained in CSV tables rather than drawn as a trivial `100%` quality curve.

## Outputs

The new scale experiment is intentionally stored separately from the historical 100-robot data:

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

## Interpretation boundary

The new x-axis represents **matched system scale** (`robots = simultaneous tasks`), not increasing task utilization within a fixed 100-robot fleet.

The experiment supports statements about how Voting assignment quality scales under a fixed 30% P2P information-loss model.

It does not make MILP/ACO lossy-scaling claims, because those methods are intentionally excluded from this main sweep.

The support-consensus stage remains a controlled centralized boundary and must not be described as fully asynchronous decentralized consensus.
