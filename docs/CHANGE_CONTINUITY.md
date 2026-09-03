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
- continuity update commit: this file's commit
