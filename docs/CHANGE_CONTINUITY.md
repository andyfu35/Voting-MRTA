# Change Continuity

Historical continuity entries through `2026-09-03 - Fix 100-Task Auction Convergence` are preserved verbatim in `docs/CHANGE_CONTINUITY_ARCHIVE_20260903.md`.

## 2026-09-03 - Direct vs Voting Optimizer Ablation

### Purpose
Isolate the value of the existing multi-view proposal-consensus mechanism before adding any external published baseline. The experiment holds the optimizer, true cost matrix, 30% directed P2P cost-loss realization, task count, robot count, and RNG pairing fixed, and changes only whether the final team decision trusts one designated incomplete-information receiver or aggregates all receiver proposals.

The preceding canonical optimizer run completed successfully for 100 robots, 30% packet loss, 100 trials per point, and task counts 5 through 100. That run established the Voting Hungarian/Auction reference columns used for the reproducibility check in this ablation.

### Files
- `run_multitask_voting_ablation.py`
- `docs/multitask_voting_ablation.md`
- `docs/CHANGE_CONTINUITY.md`
- `docs/CHANGE_CONTINUITY_ARCHIVE_20260903.md` (verbatim archive of prior continuity entries)

### Owner and named functions
The ablation has a separate experiment owner: `run_multitask_voting_ablation.py`.

- ablation diagnostic boundary: `fail`
- direct receiver contract: `validate_ablation_config`
- no-consensus decision owner: `select_direct_assignment`
- voting aggregation path: `solve_voting_assignment`
- result/invalid-decision evaluation: `evaluate_ablation_assignment`
- one-optimizer shared-proposal branching: `append_optimizer_ablation_records`
- complete-information ablation gate: `validate_zero_loss_ablation_contract`
- paired trial owner: `run_trial`
- summary owner: `summarize_results`
- full sweep: `run_experiment`
- paired report tables: `ablation_pair_table`
- voting-effect report: `voting_uplift_table`
- persistence/plots: `save_outputs`, `save_metric_plot`

The canonical cost, packet-loss, Hungarian, Auction, support, and consensus owners remain in `run_multitask_peer_cost_experiment.py` and are imported rather than copied.

### Responsibility movement
No responsibility moved out of the canonical optimizer owner. This is a new bounded experiment owner only.

- Local Hungarian/Auction proposals are solved exactly once per optimizer and trial.
- `Direct` selects receiver `0` from that same proposal batch.
- `Voting` aggregates the same batch through the existing `build_assignment_support` and `solve_support_consensus` owners.
- Direct has no fallback to another receiver when receiver 0 cannot produce a complete proposal.
- No new optimizer, communication model, or second state machine was introduced.

The continuity file itself was compacted only at the documentation layer: all prior entries were moved verbatim to `docs/CHANGE_CONTINUITY_ARCHIVE_20260903.md` and the canonical `docs/CHANGE_CONTINUITY.md` now points to that archive before recording new work. No code behavior changed because of this documentation compaction.

### Preserved behavior
- 100 robots.
- 30% independent directed sender->receiver scalar task-cost packet loss.
- 100 trials per task-count/method point.
- task counts `5,10,20,30,40,50,60,70,80,90,100`.
- same Euclidean spatial true-cost model and one-task-per-robot capacity.
- same Hungarian and epsilon-scaling Auction implementations.
- same packet-loss owner and receiver-local incomplete matrices.
- same proposal-support consensus owner for the Voting path.
- same RNG schedule `seed + task_count * 100003`, allowing Voting columns to reproduce the preceding canonical run.
- no external literature baseline is added in this block.

### Deliberately added behavior
The compared final-decision paths are:

- `Direct Hungarian`: fixed receiver 0 Hungarian proposal, no consensus.
- `Voting Hungarian`: all valid receiver Hungarian proposals plus existing consensus.
- `Direct Auction`: fixed receiver 0 Auction proposal, no consensus.
- `Voting Auction`: all valid receiver Auction proposals plus existing consensus.
- `Hungarian Oracle`: full-information reference only.

If Direct receiver 0 is locally infeasible, the final Direct assignment is invalid. It is counted as failure for optimal-cost-match and near-optimal metrics instead of silently choosing another receiver. Average optimality gap is averaged across valid final assignments only, while `valid_assignment_percent` exposes failure separately.

Added paired-effect metrics:

```text
match uplift = Voting optimal-cost-match - Direct optimal-cost-match
gap reduction = Direct average gap - Voting average gap
```

Positive values indicate benefit from the Voting/consensus path.

### Zero-loss diagnostic contract
`validate_zero_loss_ablation_contract` checks task counts `1,5,50,100` before formal data generation.

At complete communication, Direct and Voting Hungarian/Auction must all match Hungarian Oracle cost. Failure aborts at the ablation owner with a named `planning` diagnostic such as `ZERO_LOSS_ABLATION_NOT_ORACLE_CONSISTENT`.

### Validation performed
- `run_multitask_voting_ablation.py` passed Python compilation.
- A local interface-compatible control-flow harness passed the zero-loss ablation gate and 30%-loss smoke flow for task counts 5 and 20 with 3 trials. This harness verified ablation branching, invalid-result accounting, summaries, CSV/plot generation, and terminal tables; it was not a substitute for the repository's real Auction implementation.
- Full integration with the repository's canonical Hungarian/Auction owners is pending the user's `git pull` run.

### Diagnostic contract
- `contract`: invalid direct receiver/proposal shapes fail at `validate_ablation_config` or `select_direct_assignment`.
- `planning`: zero-loss optimizer inconsistency fails at `validate_zero_loss_ablation_contract`.
- `state`: optimizer proposal feasibility remains owned by the imported canonical optimizer functions; Direct invalidity is reported as an experiment outcome, not hidden by fallback.
- communication loss remains owned only by `sample_p2p_cost_visibility` in the canonical optimizer module.

### Known limitations / unfinished risks
- Receiver 0 is deliberately fixed as the Direct decision maker; receiver sensitivity/rotation is not part of the canonical ablation and can be tested later if needed.
- Direct average gap excludes invalid final assignments, so it must be interpreted together with `valid_assignment_percent`.
- Hungarian and Auction may again produce identical quality because both solve the same receiver-local assignment objective; the primary question here is Direct vs Voting, not Hungarian vs Auction.
- The Voting path still uses the controlled centralized proposal-support consensus boundary; this ablation does not claim full asynchronous decentralization.

### Next step
Run the full paired ablation with the canonical command and verify that Voting Hungarian/Auction reproduce the preceding full optimizer results. Then compare Direct vs Voting using optimal-cost-match rate and average optimality gap as the primary table.

### Commit SHA
- `fdc5099d74f5e79e682230fe34c47492de905fbc` - added `run_multitask_voting_ablation.py`
- `04a172b1dc0ca69095d04d827b48927414738b8a` - added canonical ablation specification
- continuity/archive commit: this file's commit
