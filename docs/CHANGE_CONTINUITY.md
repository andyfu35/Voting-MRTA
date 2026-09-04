# Change Continuity

Historical continuity through `2026-09-03 - Fix 100-Task Auction Convergence` remains preserved verbatim in `docs/CHANGE_CONTINUITY_ARCHIVE_20260903.md`.

The subsequent entries for `2026-09-03 - Direct vs Voting Optimizer Ablation`, `2026-09-04 - Multi-Task Optimizer Family Screening`, and `2026-09-04 - Fix MILP Numerical Exactness Boundary` are preserved exactly in Git at:

```text
commit: 1917f5bf449bb6daad2b13a17476084cdad245aa
blob:   62015eca6577cdd5cf2150791309bdd412dedcbe
```

This second documentation compaction changes no experiment or runtime behavior; it only keeps the active continuity file small enough for reliable future updates.

## 2026-09-04 - All-Optimizer Report Table and Canonical Rerun Suite

### Purpose
Prepare the final single-cost report cycle so every reported optimizer comparison is paired and reproducible before report writing begins.

The user explicitly requested that all canonical experiments be rerun and that the multi-task optimizer table include multiple algorithm families. The new report-facing optimizer comparison therefore places `Hungarian`, `Auction`, `MILP`, `ACO + Local Search`, and `Greedy Baseline` on the same 100 paired complete-information scenarios at each task count.

Robust Optimization and multi-objective success/time/energy cost models remain deliberately excluded from this report cycle. The scalar spatial cost remains the only task cost.

Before this change, repository-root `AGENTS.md` and `docs/AI_CHANGE_PROTOCOL.md` were rechecked and remain absent. The current continuity, `docs/multitask_voting_mrta.md`, `docs/multitask_optimizer_screening.md`, `run_multitask_peer_cost_experiment.py`, `run_multitask_optimizer_screening.py`, and `run_multitask_voting_ablation.py` owner functions were read before implementation.

### Files
- `run_multitask_all_optimizer_experiment.py`
- `docs/multitask_all_optimizer.md`
- `docs/report_experiment_suite.md`
- `docs/CHANGE_CONTINUITY.md`

### Owner and named functions
New bounded report-facing experiment owner: `run_multitask_all_optimizer_experiment.py`.

- diagnostic boundary: `fail`
- solver-family cost equality boundary: `cost_match_tolerance_percent`
- one-matrix Auction adaptation: `solve_single_auction_assignment`
- per-method result evaluation: `evaluate_method`
- exact optimizer pre-sweep gate: `validate_exact_optimizer_contract`
- timed optimizer execution boundary: `solve_timed`
- paired scenario/trial owner: `run_trial`
- aggregation owner: `summarize_results`
- full sweep owner: `run_experiment`
- report table owner: `report_table`
- readable terminal table owner: `readable_report_table`
- persistence/plot owners: `save_outputs`, `save_metric_plot`

No optimizer implementation was copied into the new experiment.

Shared owners reused directly:

From `run_multitask_peer_cost_experiment.py`:
- `generate_spatial_cost_matrix`
- `solve_hungarian_assignment`
- `solve_batched_auction_assignments`
- `solve_sequential_greedy`
- `assignment_total_cost`
- `validate_experiment_config`

From `run_multitask_optimizer_screening.py`:
- `ACOConfig`
- `solve_aco_assignment`
- `solve_milp_assignment`
- `validate_aco_config`
- `MILP_NUMERICAL_TOLERANCE_PERCENT`

### Responsibility movement
No existing optimizer, communication, Voting, or ablation responsibility moved.

The new experiment is orchestration only: it gives one generated complete-information scenario to five existing optimizer families and records quality/runtime metrics. Auction is adapted only by adding a singleton receiver dimension before calling the existing batched Auction owner; no second Auction implementation exists.

The canonical rerun sequence is documented separately in `docs/report_experiment_suite.md`; no wrapper changes existing experiment behavior.

### Paired-scenario contract
For every `(task_count, trial)` pair in the all-optimizer table:

```text
trial_seed = seed + task_count * 100003 + trial * 1009
```

One cost matrix is generated exactly once and shared by all five methods.

ACO receives a deterministic independent search stream:

```text
aco_seed = trial_seed + 7000003
```

The ACO search RNG does not regenerate or alter robot/task geometry, costs, or other methods' inputs.

This establishes the report rule:

```text
same scenario -> same cost matrix -> different algorithms
```

For communication experiments, the existing owners continue to enforce:

```text
same scenario -> same cost matrix + same packet-loss realization -> different algorithms/paths
```

### Preserved behavior
- `RANDOM_SEED = 20260903`.
- 100 robots.
- task counts `5,10,20,30,40,50,60,70,80,90,100`.
- default 100 trials per reported task-count point.
- spatial scalar cost `0.05 + EuclideanDistance`.
- one simultaneous task per robot.
- existing Hungarian, Auction, MILP, ACO + Local Search, and Greedy implementations.
- existing ACO fixed search budget.
- existing MILP numerical tolerance and `mip_rel_gap=0.0` behavior.
- existing single-task P2P experiment.
- existing 30% lossy P2P multi-task experiment.
- existing Direct-vs-Voting ablation.
- no Robust Optimization or multi-objective cost model.

### Deliberately added behavior
A new report-facing complete-information table contains all five optimizer families:

```text
Hungarian
Auction
MILP
ACO + Local Search
Greedy Baseline
```

The terminal report tables round display values for readability while saved raw/summary CSV values remain unrounded.

The report-facing rerun suite is now explicitly defined as:

```bash
python run_peer_cost_majority_experiment.py
python run_multitask_all_optimizer_experiment.py
python run_multitask_peer_cost_experiment.py
python run_multitask_voting_ablation.py
```

The older four-method `run_multitask_optimizer_screening.py` dataset remains valid development evidence but is superseded as the final report optimizer-family table because Auction was absent from it.

### Exact optimizer diagnostic contract
Before the all-optimizer sweep, `validate_exact_optimizer_contract` checks task counts `1,5,50,100`.

- Auction must match Hungarian within `OPTIMAL_COST_TOLERANCE_PERCENT`.
- MILP must match Hungarian within `MILP_NUMERICAL_TOLERANCE_PERCENT`.
- ACO + Local Search and Greedy are intentionally not exact-gated.

Representative failure boundaries:

```text
owner=run_multitask_all_optimizer_experiment
function=validate_exact_optimizer_contract
category=planning
code=EXACT_OPTIMIZER_NOT_HUNGARIAN
```

Per-trial exact failures use:

```text
owner=run_multitask_all_optimizer_experiment
function=run_trial
category=planning
code=EXACT_OPTIMIZER_NOT_EXACT
```

Unknown method/configuration failures remain `contract` category diagnostics.

### Report data acceptance contract
Only 100-trial canonical datasets may be used in the report.

Smoke runs using 3, 10, or 20 trials are validation only and must not be merged into final tables or plots.

Before report writing starts:

1. every canonical rerun command must finish without exception;
2. every reported configuration must contain 100 trials;
3. all exact optimizer pre-sweep gates must pass;
4. all final CSVs must be regenerated after the final code version;
5. partial datasets from a run that terminated before `save_outputs` must be discarded;
6. report tables may round for presentation, but calculations must use raw values.

### Validation performed
- `run_multitask_all_optimizer_experiment.py` passed Python bytecode compilation locally.
- The new experiment only imports already-used canonical optimizer owners; full integration execution is pending the user's repository run.
- The user's preceding full four-method optimizer screening successfully completed all task counts and 100 trials per point after the MILP numerical-boundary fix. That completed run confirmed the existing Hungarian/MILP/ACO/Greedy owners execute successfully on the user's environment.
- Auction's zero-loss/full-information exactness was already validated by the canonical P2P experiment; the new all-optimizer owner additionally rechecks it before its own sweep.

### Known limitations / unfinished risks
- The new five-method all-optimizer experiment has not yet completed a smoke or full run on the user's machine.
- The complete-information all-optimizer table is not evidence of communication robustness; P2P loss and Voting remain separate controlled experiments.
- The lossy P2P multi-task table still compares P2P Greedy/Hungarian/Auction only. MILP and ACO are not silently presented as lossy-P2P results.
- Direct-vs-Voting ablation still isolates Hungarian/Auction only. Generalizing the ablation to ACO/MILP is a later explicit experiment, not implied by the new table.
- Runtime measurements are machine-dependent and should be reported with the user's execution environment noted.

### Next step
Pull the new report experiment, run a 3-trial smoke for task counts `5 20 100`, then run the four canonical report experiments from `docs/report_experiment_suite.md` from the beginning. Paste the final terminal tables/results back for report analysis. Do not begin the final report from mixed old/new datasets.

### Commit SHA
- `eaa8aef458913d4adfbe52f97d902b68fdbb5f98` - added `run_multitask_all_optimizer_experiment.py`
- `2868f2f62551fc82395e345a1367bad415afdcdc` - added `docs/multitask_all_optimizer.md`
- `f67e8b7a8cd414b0b09eeb26a5da3e91aff437c8` - added `docs/report_experiment_suite.md`
- `5f431da22c3c6c06b0e6f6d26a264573d19531de` - recorded this continuity entry
