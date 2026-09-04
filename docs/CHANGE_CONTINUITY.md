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
- `41647907d7f2b4c6049a9979de774c3f60d14041` - archived prior continuity and recorded this ablation

## 2026-09-04 - Multi-Task Optimizer Family Screening

### Purpose
Screen the user-selected optimizer families before integrating another expensive solver into the lossy P2P Voting path. Robust Optimization is deliberately excluded for now. The bounded question is pure optimizer behavior under increasing multi-task load, without packet loss or proposal consensus confounding the result.

Compared methods:
- `Hungarian` - specialized exact linear assignment reference.
- `MILP` - general binary mixed-integer linear programming formulation of the same capacity-one assignment problem.
- `ACO + Local Search` - stochastic ant-colony construction with explicitly named local replacement/swap refinement.
- `Greedy Baseline` - existing sequential heuristic baseline, retained only for context.

Pre-change protocol lookup reconfirmed that repository-root `AGENTS.md` and `docs/AI_CHANGE_PROTOCOL.md` are absent; `docs/CHANGE_CONTINUITY.md`, `docs/multitask_voting_mrta.md`, and the actual canonical owner functions in `run_multitask_peer_cost_experiment.py` were read before implementation.

### Files
- `run_multitask_optimizer_screening.py`
- `docs/multitask_optimizer_screening.md`
- `docs/CHANGE_CONTINUITY.md`

### Owner and named functions
New bounded owner: `run_multitask_optimizer_screening.py`.

- screening diagnostic boundary: `fail`
- ACO parameter contract: `validate_aco_config`
- cached MILP structural model: `build_milp_assignment_model`
- MILP optimizer owner: `solve_milp_assignment`
- ACO candidate boundary: `select_aco_candidates`
- one-ant construction owner: `construct_aco_assignment`
- ACO local refinement owner: `improve_aco_assignment_locally`
- ACO optimizer owner: `solve_aco_assignment`
- result evaluation: `evaluate_method`
- exact-solver gate: `validate_exact_optimizer_contract`
- paired trial owner: `run_trial`
- aggregation: `summarize_results`
- sweep owner: `run_experiment`
- report table: `report_table`
- persistence/plots: `save_outputs`, `save_metric_plot`

Shared owners imported rather than copied from `run_multitask_peer_cost_experiment.py`:
- `generate_spatial_cost_matrix`
- `solve_hungarian_assignment`
- `solve_sequential_greedy`
- `assignment_total_cost`
- `validate_experiment_config`

### Responsibility movement
No existing owner was modified and no state machine was copied.

This screening intentionally does not run one optimizer per P2P receiver. It evaluates one complete-information assignment problem per paired trial so the first experiment isolates optimizer quality and runtime. This avoids multiplying MILP/ACO work by 100 receivers before knowing whether the methods are worth integrating.

MILP owns only the general exact mathematical-programming solve. ACO construction and ACO local refinement are separate named functions. The local refinement is not hidden under a generic `ACO` label; the report label is explicitly `ACO + Local Search`.

### Preserved behavior
- fleet size remains 100 robots.
- task counts remain `5,10,20,30,40,50,60,70,80,90,100`.
- default trials remain 100 per task-count/method point.
- same normalized 2-D Euclidean spatial cost owner and positive cost floor.
- same one-task-per-robot simultaneous capacity.
- same Hungarian implementation and assignment feasibility/cost diagnostics.
- no Robust Optimization uncertainty model is introduced.
- previous P2P optimizer, Voting ablation, single-task, and Auction experiments remain unchanged and reproducible.

### Deliberately added behavior
This screening disables packet loss and Voting/consensus. It adds optimizer runtime as a primary supporting metric because Hungarian and MILP are expected to have identical solution quality on the present linear assignment formulation.

MILP formulation:

```text
x_ij in {0,1}
sum_i x_ij = 1   for each task j
sum_j x_ij <= 1  for each robot i
minimize sum_ij C_ij x_ij
```

MILP uses `scipy.optimize.milp` / HiGHS. Shape-only sparse constraints are cached across repeated trials.

ACO fixed default search budget:

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

ACO construction uses pheromone and inverse-cost desirability. The bounded local-search owner then applies unused-robot replacements and task-pair robot swaps only when they reduce total cost. These ACO parameters are fixed across task counts and can be changed only through explicit CLI sensitivity runs.

### Diagnostic contract
- `contract`: invalid ACO controls or cost/model shapes fail at `validate_aco_config`, `build_milp_assignment_model`, `solve_milp_assignment`, or `solve_aco_assignment`.
- `data`: non-finite optimizer-screening costs fail as `NONFINITE_MILP_COST` or `NONFINITE_ACO_COST`; this experiment intentionally requires complete information.
- `planning`: infeasible solver results fail at `run_trial` with method-specific codes; MILP disagreement with Hungarian fails as `MILP_NOT_EXACT`.
- exact pre-sweep contract: `validate_exact_optimizer_contract` checks tasks `1,5,50,100` and requires MILP total cost to equal Hungarian within the existing tolerance.
- ACO is stochastic and is not required to match Hungarian.

### Evaluation metrics
- `average_optimality_gap_percent`
- `optimal_cost_match_percent`
- `near_optimal_5pct_percent`
- `exact_optimal_assignment_percent`
- `average_runtime_ms`
- `median_runtime_ms`

Hungarian defines the exact optimal-cost reference for the screening.

### Validation performed
- Python compilation passed for `run_multitask_optimizer_screening.py` in a local interface-compatible harness.
- Exact optimizer contract passed for tasks `1,5,50,100`: MILP matched Hungarian cost.
- 3-trial smoke sweep passed for tasks `5,20,100`.
- Smoke average gaps were approximately:
  - tasks 5: Hungarian `0%`, MILP `0%`, ACO+Local Search `0%`, Greedy `0%`.
  - tasks 20: Hungarian `0%`, MILP `0%`, ACO+Local Search `0%`, Greedy `3.20%`.
  - tasks 100: Hungarian `0%`, MILP `0%`, ACO+Local Search `2.54%`, Greedy `25.56%`.
- In the same 100-task smoke, ACO+Local Search was near-optimal within 5% in all 3 trials.
- Approximate 100-task smoke runtimes on the validation environment were Hungarian `0.39 ms`, MILP `95 ms`, ACO+Local Search `484 ms`, Greedy `0.46 ms` per solve. Runtime is environment-dependent and these values are validation diagnostics, not formal report results.
- A separate 20-trial prototype check at tasks `20,50,100` produced ACO average gaps of about `0.014%`, `0.073%`, and `2.17%`; all 20 task-100 ACO solutions were within 5% of Hungarian. These prototype numbers are not the canonical 100-trial dataset.

### Known limitations / unfinished risks
- Full 100-trial canonical screening has not yet been run on the user's repository environment.
- MILP is mathematically expected to duplicate Hungarian solution quality in this simple formulation; its useful distinction here is runtime/general modeling flexibility, not a different optimum.
- `ACO + Local Search` is a hybrid stochastic metaheuristic, not a pure Ant System implementation; the method label and canonical document state this explicitly.
- ACO performance depends on its fixed search budget. Changing ant count, iterations, or local-search moves creates a sensitivity experiment and must not be mixed into the canonical dataset.
- Screening has no packet loss or Voting. No claim about communication robustness should be made from these results.

### Next step
Run the canonical 100-trial screening on the user's machine. Inspect optimal-cost match, average gap, near-optimal rate, and runtime. Only after that result should MILP and/or ACO be promoted into a separate lossy P2P Voting comparison.

### Commit SHA
- `d3fb0d7143cd0367585dc95305a5417fc162458f` - added `run_multitask_optimizer_screening.py`
- `a8dd3bdf243b3fe63d666a012802c9739e23fba7` - added canonical optimizer-screening specification
- continuity update commit: this file's commit
