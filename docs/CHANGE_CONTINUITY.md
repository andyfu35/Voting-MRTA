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
- vote aggregation: `count_votes`
- strict-majority decision boundary: `resolve_strict_majority`
- one configuration evaluation: `run_configuration`
- full experiment sweep: `run_experiment`
- persistence/plots: `save_outputs`

### Responsibility movement
No existing experiment owner was modified. The previous preference-matrix, multi-task, hierarchical, routing, and execution experiments remain unchanged. This new owner isolates the redesigned single-task P2P cost-information-loss experiment instead of wrapping or altering those unrelated systems.

### Preserved behavior
- Existing repository experiments and output directories are untouched.
- Existing dependency set is sufficient; no new package is required.
- Costs remain deterministic and strictly ordered (`10 + 5*i`) so the globally optimal robot is unambiguous.

### Deliberately changed behavior in this new experiment
- Task delivery is assumed reliable and is not part of packet-loss testing.
- Packet loss applies only to directed robot-to-robot scalar cost messages.
- `sender -> receiver` and `receiver -> sender` loss events are independent.
- Every robot always knows its own cost.
- Every robot applies the same local greedy rule to its own incomplete cost view.
- Final execution requires a strict majority: `floor(N/2) + 1` votes.
- If no candidate reaches the strict majority, the round produces `no majority` and no executor is selected.
- The experiment sweeps robot count from 5 through 100 and packet-loss percentage from 0 through 99 by default.

### Diagnostic contract / reported categories
The current experiment reports statistical outcomes rather than application exceptions:
- communication/data visibility: `mean_optimal_vote_share`, compared against the analytical expected value
- decision/state: `majority_commit_rate`, `no_majority_rate`
- planning/decision correctness: `optimal_commit_rate`, `wrong_commit_rate`, `conditional_commit_accuracy`

Input contract failures raise explicit `ValueError` from the owning named validation/function boundary.

### Validation performed
A local smoke test over robots 5-8 and packet loss 0-20% passed.
A complete validation sweep over robots 5-100 and packet loss 0-99% with 100 trials/configuration completed successfully.
Sanity checks included:
- 0% packet loss -> 100% optimal strict-majority commit for N=5, 30, and 100.
- Mean optimal vote share tracks the analytical expectation `(1 + (N-1)*(1-loss))/N`.
- Near 99% loss, large teams generally fail to form a strict majority rather than forcing an arbitrary winner.

### Known limitations / unfinished risks
- The first experiment deliberately uses one task only.
- Cost values are deterministic and strictly ordered; random or context-dependent cost models are future controlled experiments.
- Task-delivery loss and final vote-message loss are intentionally excluded so cost-information loss is measured in isolation.
- The vectorized full sweep uses an exact decision-equivalent geometric sampler for strictly ordered costs; `simulate_cost_exchange_round` remains available as the explicit N x N directed-link reference model.
- No robot execution, route planning, multi-task conflicts, retransmission, or permanent robot failure is included in this isolated experiment.

### Next step
Run higher-trial sweeps (for example 1000 trials/configuration), inspect the transition region near 50% packet loss, and decide whether the observed majority threshold behavior matches the intended research hypothesis before adding any second source of communication loss.

### Commit SHA
- `53ad37fcf33952b0f0bb76626e1e1e9e38233dff` - added `run_peer_cost_majority_experiment.py`
- continuity document commit: this file's commit
