# Change Continuity

## 2026-09-03 - Peer Cost Majority Packet-Loss Experiment

### Purpose
Realign the controlled Voting-MRTA experiment with the intended research question: every robot receives the same task, computes only its own local task cost, exchanges that scalar cost directly with every other robot over independent directed P2P links, makes a local greedy decision from the cost information it actually received, and commits an executor only when one candidate receives a strict majority (>50%) of robot votes.

### Files
- `run_peer_cost_majority_experiment.py`
- `docs/CHANGE_CONTINUITY.md`

### Owner and named functions
The new experiment is intentionally isolated in `run_peer_cost_majority_experiment.py`.

- configuration validation: `validate_experiment_config`
- local deterministic cost generation: `generate_costs`
- explicit directed P2P cost exchange model: `simulate_cost_exchange_round`
- local greedy decision from a receiver-specific cost view: `choose_local_greedy_votes`
- exact vectorized sampling equivalent for the full sweep: `sample_local_greedy_votes`
- vote aggregation / fixed vote-window collection: `collect_votes_within_window`
- strict-majority decision boundary: `resolve_strict_majority`
- execution outcome summary: `summarize_execution_outcome`
- one configuration evaluation: `run_configuration`
- full experiment sweep: `run_experiment`
- persistence/plots: `save_outputs`

### Responsibility movement
No existing experiment owner was modified. The previous preference-matrix, multi-task, hierarchical, routing, and execution experiments remain unchanged. This new owner isolates the redesigned single-task P2P cost-information-loss experiment instead of wrapping or altering those unrelated systems.

### Preserved behavior
- Existing repository experiments and output directories are untouched.
- Existing dependency set is sufficient; no new package is required.
- Costs remain deterministic and strictly ordered (`10 + 5*i`) so the globally optimal robot is unambiguous.
- Vote-message delay/loss remains disabled in this controlled stage, so all robots currently contribute one vote inside the vote window.

### Deliberately changed behavior in this new experiment
- Task delivery is assumed reliable and is not part of packet-loss testing.
- Packet loss applies only to directed robot-to-robot scalar cost messages.
- `sender -> receiver` and `receiver -> sender` loss events are independent.
- Every robot always knows its own cost.
- Every robot applies the same local greedy rule to its own incomplete cost view.
- Majority now uses the actual number `M` of votes collected before the fixed cutoff: `floor(M/2) + 1`; with vote delay/loss disabled, `M = N` and numerical behavior is preserved.
- If no candidate reaches the strict majority, the round produces no executor.
- Default trials per `(robots, packet_loss)` point were set to 100 and the terminal summary reports successful optimal executions directly as a count and percentage.

### Diagnostic contract / reported categories
The current experiment reports statistical outcomes rather than application exceptions:
- communication/data visibility: `mean_optimal_vote_share`, compared against the analytical expected value
- time/collection boundary: `mean_votes_in_window`, `vote_window_participation_rate`, `mean_required_majority_votes`
- state: `majority_commit_rate`, `no_majority_rate`
- planning/decision correctness: `successful_optimal_executions`, `optimal_execution_success_percent`, `wrong_execution_count`, `no_execution_count`

Input contract failures raise explicit `ValueError` from the owning named validation/function boundary.

### Validation performed
A local smoke test over robots 5-8 and packet loss 0-20% passed.
A complete validation sweep over robots 5-100 and packet loss 0-99% with 100 trials/configuration completed successfully.
Sanity checks included:
- 0% packet loss -> 100% optimal strict-majority commit for N=5, 30, and 100.
- Mean optimal vote share tracks the analytical expectation `(1 + (N-1)*(1-loss))/N`.
- Near 99% loss, large teams generally fail to form a strict majority rather than forcing an arbitrary winner.
- Replacing fixed fleet membership with the in-window vote count preserved results while all votes are configured in-window.

### Known limitations / unfinished risks
- The first experiment deliberately uses one task only.
- Cost values are deterministic and strictly ordered; random or context-dependent cost models are future controlled experiments.
- Task-delivery loss and final vote-message loss are intentionally excluded so cost-information loss is measured in isolation.
- The vectorized greedy sweep uses an exact decision-equivalent geometric sampler for strictly ordered costs; `simulate_cost_exchange_round` remains available as the explicit N x N directed-link reference model.
- No robot execution, route planning, multi-task conflicts, retransmission, or permanent robot failure is included in this isolated experiment.

### Commit SHA
- `53ad37fcf33952b0f0bb76626e1e1e9e38233dff` - added `run_peer_cost_majority_experiment.py`
- `9db9bb485aeed7a9dfca2b857e2b346de0c79c3f` - changed strict majority to use in-window vote count
- `cf0801fddaea248daf4f3b443fbc4c4b89a5571c` - standardized 100-trial execution summaries

## 2026-09-03 - Paired Single-Task Voting Policy Comparison

### Purpose
Compare alternative local voting policies under exactly the same single-task directed P2P cost-message loss model before moving to multi-task MRTA. The experiment is report-oriented: every `(robot_count, packet_loss, policy)` point is summarized over 100 trials by the percentage of trials in which the globally minimum-cost robot actually obtains a strict majority and executes.

### Files
- `run_peer_cost_policy_comparison.py`
- `docs/CHANGE_CONTINUITY.md`

### Owner and named functions
The comparison is isolated in `run_peer_cost_policy_comparison.py`.

- experiment contract validation: `validate_experiment_config`
- paired directed P2P visibility realization: `sample_paired_cost_visibility`
- generic probabilistic candidate sampling: `sample_from_weight_tensor`
- Greedy local decision: `greedy_votes`
- Inverse-cost local decision: `inverse_cost_votes`
- Softmax/Boltzmann local decision: `softmax_votes`
- rank-based local decision: `rank_votes`
- policy dispatch: `generate_policy_votes`
- execution summary for one policy: `summarize_policy`
- paired configuration evaluation: `run_configuration`
- experiment sweep: `run_experiment`
- report CSV tables: `save_robot_success_tables`
- report plots: `save_success_plot`
- pasteable terminal tables: `printable_success_table`

### Responsibility movement
The existing Greedy baseline owner remains unchanged and supplies the shared cost generation, vote-window collection, strict-majority decision, and execution-summary functions. The new comparison owner adds policy generation and paired comparison only; it does not alter the old multi-task, preference-matrix, hierarchical, route, or execution experiments.

### Preserved behavior
- Task delivery remains reliable.
- Only directed robot-to-robot cost messages are lossy.
- Every robot always knows its own cost.
- Vote messages remain reliable/in-window in this controlled experiment.
- Every robot contributes exactly one final vote for every policy.
- A task executes only when one candidate obtains a strict majority of the votes received in-window.
- The deterministic costs remain `10 + 5*i`, so Robot 0 is the unambiguous ground-truth optimum for this controlled comparison.

### Deliberately changed / added behavior
Ten policy settings are compared:
- Greedy
- Inverse cost: alpha = 1, 2, 3
- Softmax: beta = 1, 2, 5 using receiver-local min/max cost normalization
- Rank: gamma = 1, 2, 3 using `1 / rank^gamma`

Inverse, Softmax, and Rank remain one-robot/one-vote methods by sampling one candidate from their receiver-local probability distribution. All policies use the same P2P delivery tensor for a paired trial, and the stochastic policies also share common random voter uniforms to reduce comparison noise.

Default report robot counts are `5, 10, 20, 30, 50, 75, 100`; packet loss sweeps 0-99%; every data point uses 100 trials. Full 0-99% tables are saved per robot count, while the terminal prints the same representative packet-loss rows used in the previous report table.

### Diagnostic contract
- data/communication: every policy receives the same receiver-specific visibility tensor
- planning: each named policy owns only its cost-to-vote transformation
- state: shared strict-majority owner decides whether an executor exists
- contract: invalid policy parameters, visibility shapes, or sampling weights fail at the first named function with `ValueError`

Primary report metric:
- `optimal_execution_success_percent`

Supporting diagnostics retained in the long-form CSV:
- successful/failed optimal execution counts
- wrong execution count
- no-execution count
- mean/min/max optimal-robot votes

### Validation performed
Local smoke tests passed for:
- N=5 and N=10, loss 0-5%, 20 trials/point
- N=5, N=30, and N=100, loss 30-50%, 100 trials/point

Greedy reproduces the expected high single-task success region; softer stochastic policies intentionally expose the cost of vote dispersion in a strict-majority single-task setting.

### Known limitations / unfinished risks
- These probability-based policies are voting policies, not global multi-task assignment optimizers.
- Soft policies can fail to form a majority even at low packet loss because they deliberately distribute votes; that behavior is part of this experiment rather than an implementation error.
- The deterministic ordered cost model makes rank computation especially efficient; a later random/contextual cost experiment must rank by actual visible costs instead of relying on candidate-ID order.
- This comparison does not yet test multi-task conflicts or robot capacity constraints.

### Next step
Create a separate multi-task owner rather than extending this single-task comparison. The agreed next experiment is:
- 100 robots fixed
- directed P2P cost-message packet loss fixed at 30%
- 100 trials per task-count/method point
- vary simultaneous task count
- compare multi-task allocation optimizers/baselines, with centralized Hungarian as the full-information reference and decentralized/heuristic methods evaluated under the same communication realization

### Commit SHA
- `5673b81007c7a21c1ad518e32af318f79a37b21a` - added `run_peer_cost_policy_comparison.py`
- `3b1d17c20067be15635dd3262a77453603a41b59` - continuity update

## 2026-09-03 - 100-Robot Multi-Task Peer-Cost Comparison

### Purpose
Introduce the next controlled experiment without modifying the single-task owners: fix the fleet at 100 robots, fix independent directed P2P task-cost packet loss at 30%, vary simultaneous task count, and compare assignment quality over 100 paired trials per task-count/method point.

### Files
- `run_multitask_peer_cost_experiment.py`
- `docs/multitask_voting_mrta.md`
- `docs/CHANGE_CONTINUITY.md`

### Canonical experiment contract
- robot count: 100
- packet loss: 30% on directed sender->receiver scalar task-cost messages
- trials: 100 per task-count/method point
- task counts: 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
- robot simultaneous capacity: one task
- task delivery and final vote collection: reliable in this controlled stage
- excluded: route planning, execution noise, retransmission, permanent failure, asynchronous delay, deadlines

### Owner and named functions
The new owner is `run_multitask_peer_cost_experiment.py`.

- contract validation: `validate_experiment_config`
- shared ground-truth spatial cost: `generate_spatial_cost_matrix`
- P2P task-cost communication visibility: `sample_p2p_cost_visibility`
- probabilistic one-vote sampler: `sample_weighted_candidates`
- local Greedy voting: `greedy_task_votes`
- local Inverse voting: `inverse_task_votes`
- local Softmax voting: `softmax_task_votes`
- local Rank voting: `rank_task_votes`
- policy dispatch: `generate_task_votes`
- candidate/task support construction: `build_vote_support`
- full-information Hungarian reference: `solve_hungarian_optimal`
- full-information sequential greedy baseline: `solve_sequential_greedy`
- capacity-one support matching: `solve_support_assignment`
- assignment feasibility/cost boundary: `assignment_total_cost`
- evaluation against Hungarian: `evaluate_assignment`
- paired trial owner: `run_trial`
- task-count sweep: `run_experiment`
- report persistence: `save_report_tables`, `save_outputs`

### Responsibility movement
The prior `docs/multitask_voting_mrta.md` canonical specification described the older terminal-centric preference/vote experiment. It has been deliberately replaced as canonical by the raw peer-cost model. Older scripts are retained for historical reproducibility and are not wrapped or silently altered.

### Preserved behavior
- The selected single-task policy parameters are reused rather than re-tuned per task count: Inverse alpha=3, Softmax beta=5, Rank gamma=3.
- Report labels intentionally hide tuning parameters and display only `Greedy`, `Inverse`, `Softmax`, and `Rank`.
- All compared P2P voting policies receive the same task costs, the same P2P visibility realization, and shared voter random uniforms inside each paired trial.

### Deliberately changed / added behavior
- Costs are now robot-task spatial costs rather than a deterministic single-task ordered vector.
- Each receiver can have a different incomplete cost view for each task.
- Every receiver casts one candidate vote for every simultaneous task.
- Multi-task capacity conflicts are resolved through one shared maximum-vote-support matching owner; true task cost is not used as a hidden vote-assignment tie-break.
- Hungarian uses the complete true cost matrix and is the `0%` optimality-gap reference.
- Sequential Greedy is retained as a full-information heuristic baseline.

### Diagnostic contract
- data: invalid task count/cost/assignment shapes fail at their named owner
- communication: `sample_p2p_cost_visibility` is the only packet-loss owner
- planning: each voting policy owns only its local cost-to-vote transformation
- state/constraint: `solve_support_assignment` and `assignment_total_cost` own capacity-one feasibility
- planning correctness: `evaluate_assignment` reports total cost, optimality gap, near-optimal-within-5%, and exact Hungarian assignment match
- contract failures raise `ValueError` at the first named boundary

### Validation performed
Local smoke tests passed for:
- task counts 5 and 10, 5 trials per point
- task counts 5, 50, and 100, 10 trials per point

The smoke tests produced feasible capacity-one assignments for every method and showed the expected distinction between exact assignment match and low-cost near-optimal solutions.

### Known limitations / unfinished risks
- This is not yet an asynchronous CBBA or bundle-auction implementation.
- Task count is capped at 100 because robot simultaneous capacity is one in this first controlled experiment.
- Vote delivery itself is still reliable; only cost-information exchange is lossy.
- The support matching stage is a shared centralized arbitration boundary for all four voting policies; later work can replace that boundary with a true decentralized auction/consensus mechanism.
- Spatial travel cost is the only task cost component in this experiment.

### Next step
Run the full 100-trial task-count sweep on the user's machine, inspect average optimality gap / near-optimal rate / exact assignment match, then decide which policy families deserve a true decentralized CBBA/auction implementation.

### Commit SHA
- `af952d2630898be006fe52249b3ef4d9453a86d4` - added `run_multitask_peer_cost_experiment.py`
- `fb6b740263e82fe6909a1b7d158c7a38b68a3f86` - updated multi-task canonical specification
- continuity update commit: this file's commit

## 2026-09-03 - Redefine Multi-Task Comparison Around Real Optimizers

### Purpose
Reject the probabilistic single-vote policy families as the canonical optimizer comparison and replace them with actual assignment solvers. The trigger was the observed failure of Inverse/Softmax/Rank sampling to select the known best robot even at 0% packet loss. A method that intentionally randomizes away from a known optimum is not treated as an optimizer in the canonical experiment.

### Files
- `run_multitask_peer_cost_experiment.py`
- `docs/multitask_voting_mrta.md`
- `docs/CHANGE_CONTINUITY.md`

### Owner and named functions
The canonical owner remains `run_multitask_peer_cost_experiment.py`, but its planning responsibility has been replaced rather than wrapped.

- diagnostic boundary: `fail`
- contract validation: `validate_experiment_config`
- shared ground-truth cost: `generate_spatial_cost_matrix`
- P2P communication visibility: `sample_p2p_cost_visibility`
- receiver-specific incomplete matrix: `build_receiver_cost_views`
- exact assignment cost/feasibility: `assignment_total_cost`
- exact Hungarian solver: `solve_hungarian_assignment`
- Greedy heuristic solver: `solve_sequential_greedy`
- batched Bertsekas-style auction solver: `solve_batched_auction_assignments`
- local optimizer dispatch: `solve_local_optimizer_proposals`
- proposal support: `build_assignment_support`
- team consensus assignment: `solve_support_consensus`
- end-to-end P2P optimizer proposal path: `optimizer_consensus_assignment`
- mandatory complete-information check: `validate_zero_loss_optimizer_contract`
- evaluation: `evaluate_assignment`
- paired trial: `run_trial`
- task-count sweep: `run_experiment`
- report tables/figures: `report_table`, `save_report_tables`, `save_outputs`

### Responsibility movement
- Removed Inverse, Softmax, and Rank stochastic vote generation from the canonical multi-task owner.
- The historical `run_peer_cost_policy_comparison.py` is retained only for reproducibility/screening and is no longer an optimizer benchmark.
- Multi-task decisions now come from complete assignment proposals produced by actual assignment solvers over each receiver's own incomplete cost matrix.
- P2P Hungarian and P2P Auction optimize the same one-task-per-robot assignment objective; P2P Greedy remains explicitly a heuristic baseline.
- The final shared consensus boundary aggregates complete assignment proposals, not random candidate ballots.

### Preserved behavior
- 100 robots.
- 30% independent directed sender->receiver task-cost packet loss by default.
- 100 trials per task-count/method point.
- task counts `5,10,20,30,40,50,60,70,80,90,100`.
- every receiver always knows its own task-cost row.
- task delivery and final proposal collection remain reliable/in-window.
- one simultaneous task per robot.
- all compared methods receive paired cost matrices and paired P2P loss realizations.
- route planning, execution noise, retransmission, permanent failure, asynchronous delay, and deadlines remain excluded.

### Deliberately changed behavior
- Canonical methods are now `Hungarian Oracle`, `P2P Hungarian`, `P2P Auction`, and `P2P Greedy` only.
- Missing receiver-local cost entries are unavailable (`+inf`) to the optimizer rather than converted to stochastic vote weights.
- Every receiver proposes a complete feasible assignment when possible.
- Proposal support is aggregated and the final team assignment maximizes proposal support under capacity-one constraints.
- Added `optimal_cost_match_percent` so different assignments with the same optimal total cost count as successful optimization.
- Added `average_valid_proposal_rate_percent` to expose local infeasibility separately from optimization quality.
- Added `--packet-loss` CLI control for explicit zero-loss verification while keeping 30% as the canonical default.

### Zero-loss diagnostic contract
Before every formal sweep, `validate_zero_loss_optimizer_contract` checks complete-information cases at task counts 1, 5, 50, and 100.

- P2P Hungarian must reach Hungarian Oracle total cost.
- P2P Auction must reach Hungarian Oracle total cost.
- In the single-task case P2P Greedy must select the oracle minimum-cost robot.
- All local proposals must be valid at 0% loss.

A violation aborts at the first failing owner/function with category `planning` or `contract` and a named code such as `ZERO_LOSS_NOT_ORACLE_CONSISTENT`.

### Validation performed
Local smoke tests passed after the redesign:
- mandatory zero-loss contract passed for task counts 1, 5, 50, and 100.
- 30% packet loss, tasks 5 and 20, 3 trials per point produced feasible final assignments for all methods.
- at 5 tasks in that smoke test all P2P methods reached oracle cost.
- at 20 tasks P2P Hungarian/Auction remained substantially closer to oracle than P2P Greedy, confirming that the multi-task optimizer distinction is now measurable.

### Known limitations / unfinished risks
- P2P Auction is an assignment auction over each receiver-local matrix; it is not yet a fully asynchronous network-level CBBA protocol with lossy bid messages.
- P2P Hungarian and P2P Auction can produce very similar quality because both optimize the same local assignment objective; their future distinction should include communication/computation cost when the asynchronous protocol layer is added.
- P2P Greedy can fail to produce a complete local assignment at high task load even when a feasible matching exists; this is intentionally reported through valid proposal rate rather than hidden by fallback logic.
- The final proposal-consensus boundary remains shared and centralized for this controlled experiment.

### Next step
Run the canonical 100-robot, 30%-loss, 100-trial task-count sweep and use the printed tables for report analysis. The primary tables are average optimality gap, optimal-cost match rate, near-optimal-within-5% rate, and valid local proposal rate.

### Commit SHA
- `3763166ba052b9f809bc50147c3eb010d4c7a679` - replaced stochastic voting policies with real assignment optimizers
- `095ffcd19fc8ccbd5ac16f26e32c501727191510` - updated canonical multi-task specification
- continuity update commit: this file's commit

## 2026-09-03 - Fix 100-Task Auction Convergence

### Purpose
Fix the first formal 100-trial run stopping at the 100-task boundary with `owner=run_multitask_peer_cost_experiment function=solve_batched_auction_assignments category=planning code=AUCTION_DID_NOT_CONVERGE details=tasks=100 active_receivers=1`. The failure came from using one fixed extremely small auction epsilon (`1e-8`) from the start; at the square 100x100 boundary one receiver could require more than the previous 200,000 iteration cap even though the assignment problem was feasible.

### Files
- `run_multitask_peer_cost_experiment.py`
- `docs/multitask_voting_mrta.md`
- `docs/CHANGE_CONTINUITY.md`

### Owner and named functions
Auction planning remains owned by `run_multitask_peer_cost_experiment.py`.

- one epsilon-stage state machine: `run_batched_auction_stage`
- epsilon-scaling owner: `solve_batched_auction_assignments`
- unchanged zero-loss correctness gate: `validate_zero_loss_optimizer_contract`

### Responsibility movement
The iterative bid/price state for one epsilon level was separated from the multi-stage auction owner. No wrapper was added around another optimizer and no fallback to Hungarian was introduced for Auction results. `solve_batched_auction_assignments` remains the true P2P Auction owner.

### Preserved behavior
- 100 robots, 30% directed P2P task-cost loss, 100 trials per point.
- same task counts and same random pairing rules.
- same receiver-specific incomplete cost matrices.
- same proposal support and consensus assignment boundary.
- same Hungarian Oracle, P2P Hungarian, P2P Greedy, and P2P Auction method labels.
- same optimality-gap, optimal-cost-match, near-optimal, exact-assignment, and valid-proposal metrics.
- zero-loss Auction is still required to match Hungarian Oracle cost before the formal sweep starts.

### Deliberately changed behavior
- Replaced one fixed `AUCTION_EPSILON=1e-8` run with epsilon scaling: `1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8`.
- Each epsilon stage resets assignment/ownership while retaining the preceding stage's price vector, so coarse stages approach the solution quickly and fine stages refine it.
- For rectangular cases with fewer tasks than robots, zero-cost dummy tasks internally square the auction problem. Dummy tasks consume otherwise unused robots and do not change real-task objective cost.
- The former global `AUCTION_DID_NOT_CONVERGE` diagnostic is replaced at the narrower boundary by `run_batched_auction_stage / planning / AUCTION_STAGE_DID_NOT_CONVERGE`, including `tasks`, `epsilon`, `active_receivers`, and `iterations`.

### Diagnostic contract
If a stage still fails to converge, the first failure now reports:

```text
owner=run_multitask_peer_cost_experiment
function=run_batched_auction_stage
category=planning
code=AUCTION_STAGE_DID_NOT_CONVERGE
```

with the exact task count, epsilon stage, active receiver count, and iteration count. Data visibility, consensus state, and other optimizers remain separate owners.

### Validation performed
- Python compile check passed.
- Mandatory zero-loss contract passed at task counts 1, 5, 50, and 100.
- 30% packet loss smoke test passed for task counts 5, 20, and 100 with 3 trials per point.
- A focused 100-task, 30%-loss run completed 20 trials without auction convergence failure.
- In that 20-trial check, P2P Hungarian and P2P Auction produced the same average optimality gap (`1.474198%`) and both were near-optimal within 5% in `95%` of trials; P2P Greedy gap was `26.430752%`.
- The user's full 100-trial sweep still needs to be rerun after pulling this commit; no partial 5-90 task output from the aborted run is treated as the final report dataset.

### Known limitations / unfinished risks
- Auction remains a receiver-local assignment optimizer, not a network-level asynchronous CBBA protocol.
- The full canonical 100-trial sweep is pending user verification after this fix.
- Extremely different future cost scales may warrant deriving epsilon levels from the observed cost range rather than fixed levels.

### Next step
Pull the convergence fix and rerun the canonical command. Use only a run that reaches `tasks=100/100 complete` and prints all report tables as the report dataset.

### Commit SHA
- `01d3920d80cf3d9376d0c1f1395d97d672b9cb6c` - epsilon-scaling Auction implementation
- `3090d07d4188ecb0a659ff9a0f7b1e5e570def0f` - canonical Auction convergence specification update
- continuity update commit: this file's commit
