# Multi-Task Voting Ablation

This is the canonical ablation experiment for the proposal-consensus mechanism used by the multi-task peer-cost optimizer comparison.

## Research question

With the optimizer, true cost matrix, directed P2P packet-loss realization, robot count, task count, and random seed held fixed, how much assignment quality is gained by aggregating the independent receiver-local optimizer proposals instead of directly adopting one receiver's incomplete-information result?

The only intended experimental difference is:

```text
Direct: one fixed receiver proposal -> final assignment
Voting: all receiver proposals -> proposal support -> consensus assignment
```

No external paper method is added in this ablation.

## Controlled settings

- Robots: `100`
- Directed P2P scalar task-cost packet loss: `30%`
- Trials per task-count/method point: `100`
- Simultaneous task counts: `5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100`
- Robot simultaneous capacity: one task
- Direct decision maker: receiver `0`, fixed before all trials
- Task delivery: reliable
- Final proposal collection for the Voting path: reliable/in-window
- Route planning, execution noise, retransmission, permanent failure, asynchronous delay, and deadlines remain excluded.

## Shared cost and communication owners

This experiment imports the canonical owners from `run_multitask_peer_cost_experiment.py` rather than copying a second implementation.

- ground-truth spatial cost: `generate_spatial_cost_matrix`
- directed P2P visibility: `sample_p2p_cost_visibility`
- receiver-local incomplete matrices: `build_receiver_cost_views`
- Hungarian optimizer: `solve_hungarian_assignment`
- Auction optimizer: `solve_local_optimizer_proposals` -> `solve_batched_auction_assignments`
- proposal support: `build_assignment_support`
- consensus assignment: `solve_support_consensus`
- assignment feasibility / true cost: `assignment_total_cost`

The ablation owner is `run_multitask_voting_ablation.py`.

## Compared paths

### Hungarian Oracle

The full-information minimum-total-cost assignment. This is the reference only.

### Direct Hungarian

All 100 receiver-local Hungarian proposals are computed once for pairing with the Voting path, but the final decision uses only the fixed receiver-0 proposal:

```text
assignment = proposals[0]
```

There is no fallback to another receiver and no proposal aggregation. If receiver 0 cannot produce a complete assignment, the trial is recorded as an invalid Direct result and counts as a failure for optimal-cost-match and near-optimal-rate metrics.

### Voting Hungarian

Uses the exact same batch of receiver-local Hungarian proposals as Direct Hungarian. All valid proposals are converted to robot-by-task support and the existing capacity-one consensus owner produces the final team assignment.

### Direct Auction

Uses only receiver 0's assignment from the same batch of receiver-local epsilon-scaling Auction proposals used by Voting Auction. No fallback and no consensus are allowed.

### Voting Auction

Uses all valid receiver-local Auction proposals and the existing proposal-support consensus boundary.

## Paired fairness contract

Within a trial, Direct and Voting variants of the same optimizer share:

- the exact same true cost matrix
- the exact same directed P2P visibility tensor
- the exact same receiver-local incomplete cost matrices
- the exact same local optimizer proposal batch
- the exact same task order where required

The local proposals are solved only once per optimizer. Direct selects receiver 0 from that batch; Voting aggregates the same batch. Therefore the ablated concern is the proposal-consensus mechanism rather than optimizer randomness or a second communication realization.

The task-count RNG schedule is kept identical to the canonical optimizer experiment:

```text
seed + task_count * 100003
```

Therefore the Voting Hungarian and Voting Auction columns should reproduce the corresponding P2P Hungarian and P2P Auction results from the preceding canonical run when the same seed, packet-loss rate, task counts, and trial count are used.

## Zero-loss ablation contract

Before the formal 30% loss sweep, `validate_zero_loss_ablation_contract` checks task counts `1, 5, 50, 100`.

With complete communication:

- Direct Hungarian must match Hungarian Oracle cost.
- Voting Hungarian must match Hungarian Oracle cost.
- Direct Auction must match Hungarian Oracle cost.
- Voting Auction must match Hungarian Oracle cost.

A violation aborts before formal data generation.

## Evaluation metrics

### Optimal-cost match

Percentage of all trials that reach the same total cost as the full-information Hungarian Oracle within numerical tolerance.

Invalid Direct assignments count as failures.

### Average optimality gap

```text
gap (%) = 100 * (method_cost - oracle_cost) / oracle_cost
```

The average is computed across valid final assignments. Valid-assignment rate is reported separately so infeasibility is not hidden.

### Near-optimal within 5%

Percentage of all trials with a valid assignment whose gap is at most `5%`. Invalid assignments count as failures.

### Valid final assignment

Percentage of trials in which the Direct or Voting path returns a complete capacity-one assignment.

### Voting uplift

The report also computes paired aggregate improvements:

```text
match uplift = Voting optimal-cost-match - Direct optimal-cost-match
gap reduction = Direct average gap - Voting average gap
```

Positive values mean the Voting path improved the corresponding metric.

## Run

Canonical full run:

```bash
python run_multitask_voting_ablation.py
```

Smoke test:

```bash
python run_multitask_voting_ablation.py --tasks 5 20 100 --trials 3
```

The designated Direct receiver may be changed for sensitivity analysis, but receiver `0` is the canonical ablation setting:

```bash
python run_multitask_voting_ablation.py --direct-receiver 0
```

## Outputs

```text
results/multitask_voting_ablation/data/
results/multitask_voting_ablation/figures/
```

Primary report CSVs:

```text
report_optimal_cost_match_percent.csv
report_average_optimality_gap_percent.csv
report_near_optimal_5pct_percent.csv
report_valid_assignment_percent.csv
report_voting_uplift.csv
```

Raw and summary data:

```text
voting_ablation_raw.csv
voting_ablation_summary.csv
```

## Scope boundary

This ablation does not claim to represent another published MRTA method. It isolates whether multi-view proposal aggregation improves robustness to incomplete P2P cost information relative to directly trusting one designated receiver using the same exact optimizer.
