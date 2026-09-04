# Change Continuity

Historical continuity through `2026-09-04 - Narrow Experiment 2 to 100-Task Steps and Hand Off to Windows Runtime` is preserved exactly in Git at:

```text
commit: 5534ccfbb57cae6bdf0d31aa0a1b881f2e577b92
blob:   d96ca1fc86de4a0c45a5e570a28bdbbcd6b9af14
```

## 2026-09-04 - Replace Slow Lossy Optimizers with Fast Heuristics and Parallel Trials

### Purpose
The user decided to keep Experiment 2 on the Mac and target an approximately one-hour end-to-end runtime. The preceding fixed-100-robot design still executed receiver-local Hungarian, MILP, and ACO + Local Search. A real macOS probe at `1000` tasks, one trial, one voter, with all five optimizer families took about `85.44 s`, making the full all-voter repeated experiment impractical.

The report-facing lossy workload experiment now focuses on communication loss + Voting with:

```text
Voting Sequential Greedy
Voting Global Greedy
Voting Static Regret-2 Greedy
Voting Auction
Hungarian Oracle as full-information minimum reference
```

The canonical Experiment 2 trial count is deliberately reduced from `100` to `20` per task point while retaining all `100` physical voters. Independent `(task_count, trial)` jobs are parallelized with a bounded process pool, defaulting to up to four workers.

Before modification, repository-root `AGENTS.md` and `docs/AI_CHANGE_PROTOCOL.md` were checked in that order and remain absent. Then `docs/CHANGE_CONTINUITY.md`, canonical `docs/multitask_voting_mrta.md`, the actual Experiment 2 owner `run_multitask_peer_cost_all_optimizers.py`, its exact capacity/routing/trial/report functions, the existing Greedy/Auction owner functions in `run_multitask_peer_cost_experiment.py`, and `docs/report_experiment_suite.md` were read before writing.

### Files
- `run_multitask_workload_heuristics.py` - new true owner for the three fast capacitated heuristic algorithms.
- `run_multitask_peer_cost_all_optimizers.py` - canonical Experiment 2 routing, Voting, evaluation, and parallel trial execution.
- `docs/multitask_voting_mrta.md` - canonical specification.
- `docs/report_experiment_suite.md` - report-authoritative rerun contract.
- `docs/CHANGE_CONTINUITY.md` - this continuity record.

The existing optimizer-screening module and Experiment 1/3 owners were not modified.

### Owner and named functions

Fast heuristic owner:

```text
run_multitask_workload_heuristics.py
```

Named boundaries:

- `validate_capacitated_problem`
- `solve_sequential_greedy_capacitated`
- `solve_global_greedy_capacitated`
- `compute_static_regret2_priority`
- `solve_regret2_greedy_capacitated`
- `solve_capacitated_heuristic`
- `solve_capacitated_heuristic_batch`

Experiment 2 owner:

```text
run_multitask_peer_cost_all_optimizers.py
```

Important named boundaries:

- method selection: `resolve_voting_methods`
- workload/runtime validation: `validate_workload_config`
- capacity representation: `resolve_robot_capacity`, `build_capacity_slot_cost_matrix`, `build_capacity_slot_cost_views`, `map_slot_assignments_to_robots`
- capacity validation/evaluation: `validate_capacity_assignment`, `assignment_total_cost_with_capacity`
- Oracle: `solve_capacity_oracle`
- communication: `sample_voter_batch_visibility`, `build_voter_batch_cost_views`
- local routing: `solve_voter_batch_proposals`
- support/Voting: `accumulate_proposal_support`, `collect_voting_support`, `solve_capacity_support_consensus`, `finalize_voting_assignments`
- bounded preflight: `validate_zero_loss_heuristic_contracts`, `validate_zero_loss_auction_contract`, `validate_zero_loss_optimizer_contract`
- paired trial: `run_trial`
- parallel runtime: `run_trial_job`, `build_trial_jobs`, `configure_worker_thread_environment`, `report_trial_completion`, `execute_trial_jobs`
- sweep/report: `run_experiment`, `summarize_results`, `save_outputs`

The Auction algorithm remains owned by:

```text
run_multitask_peer_cost_experiment.py::solve_batched_auction_assignments
```

The full-information minimum remains owned by:

```text
run_multitask_peer_cost_experiment.py::solve_hungarian_assignment
```

### Responsibility movement
The three new greedy algorithms are not wrappers around another owner. They now have one dedicated optimizer owner and operate directly on physical `100 x T` receiver-local cost matrices plus explicit physical capacity.

Sequential/Global/Static-Regret heuristics therefore no longer require slot expansion. Capacity slots remain only where required by existing capacity-one exact owners: Hungarian Oracle, Voting Auction, and final support consensus.

Trial semantics remain in `run_trial`; deterministic job construction is in `build_trial_jobs`; process execution is in `execute_trial_jobs`. Worker scheduling never enters a seed formula.

### Preserved behavior
- `100` physical robots fixed.
- Canonical task batches `100, 200, ..., 1000`.
- `30%` directed P2P scalar cost-message loss.
- Scalar cost `0.05 + EuclideanDistance`.
- `capacity_per_robot = ceil(tasks/100)`.
- Task counts are allocation batches, not simultaneous execution claims.
- Every receiver always knows its own sender row.
- Missing receiver-local costs remain `+inf` unavailable edges.
- Task delivery and final proposal collection remain reliable/in-window in this controlled stage.
- All Voting methods share the same cost matrix, voters, packet loss, task order, tie priority, and capacity within each trial.
- Support remains physical robot/task support; final consensus uses the same capacity contract and no hidden true-cost tie-break.
- Hungarian Oracle remains the minimum-cost reference.
- Primary report metric remains direct cost error from the minimum.
- Supporting `<=5%`, optimal-cost match, exact assignment, and valid-proposal-rate CSV fields remain.
- Output root remains `results/multitask_peer_cost_fixed100_workload/`.
- Experiment 1 and Experiment 3 remain unchanged.

### Deliberately changed behavior

Report-facing receiver-local methods removed:

```text
Voting Hungarian
Voting MILP
Voting ACO + Local Search
```

Their implementations were not deleted. MILP and ACO remain in the separate complete-information optimizer screening. Voting Auction is retained as the exact local assignment-family comparison.

New report-facing methods:

```text
Voting Sequential Greedy
Voting Global Greedy
Voting Static Regret-2 Greedy
Voting Auction
```

`Static Regret-2` computes the best-vs-second-best receiver-visible regret once per task, uses paired task order for ties, then assigns tasks in that fixed priority to cheapest robots with remaining capacity. It is not dynamic regret recomputation.

Canonical trials changed from `100` to `20` per task point. All `100` physical voters remain enabled.

New runtime option:

```text
--workers N
```

Default is `min(4, available CPU count)`. `--workers 1` is serial diagnostic mode.

When multiple workers are used, standard BLAS/OpenMP thread-count environment variables default to `1` only if the user did not explicitly set them. Parallel workers suppress receiver-level console progress; the parent process reports trial completion.

Default receiver batch size changes from `8` to `4` to bound per-process memory.

The report-facing CLI no longer exposes `--include-milp`, `--only-milp`, or `--include-aco` because those solvers are no longer part of this experiment's canonical question.

### Diagnostic contract
Experiment-owner failures keep the standard format:

```text
owner=run_multitask_peer_cost_all_optimizers
function=<named function>
category=<data|time|state|dependency|planning|safety|runtime|contract>
code=<named code>
```

New runtime contract example:

```text
function=validate_workload_config category=contract code=INVALID_WORKER_COUNT
```

Fast heuristic failures use:

```text
owner=run_multitask_workload_heuristics
```

Representative first-failure diagnostics:

```text
function=validate_capacitated_problem category=data code=INVALID_HEURISTIC_COST
function=validate_capacitated_problem category=state code=CAPACITY_EXCEEDED
function=solve_capacitated_heuristic category=contract code=UNKNOWN_METHOD
```

Auction diagnostics still propagate from its existing true owner.

### Validation performed
- The new heuristic owner was re-fetched from GitHub and inspected across validation, all three algorithms, routing, and receiver-batch execution.
- The rewritten Experiment 2 owner was re-fetched and inspected across imports/constants, capacity functions, solver routing, bounded zero-loss gates, deterministic trial seeding, process-job creation, process execution, aggregation, and CLI defaults.
- Canonical/report documents were synchronized with the new method set, 20-trial formal contract, and parallel runtime design.
- A container-side repository clone/compile attempt was made, but the container still cannot resolve `github.com`; no real repository bytecode/runtime test could be executed there.
- The previous `85.44 s` Mac result is runtime motivation only; it does not validate the new fast method set.

### Unfinished risks
- The one-hour target is not yet validated. Voting Auction may still dominate large workloads.
- Process parallelism increases peak memory; lower `--workers` or `--voter-batch-size` may be needed on smaller Macs.
- macOS process startup and BLAS behavior depend on the installed Python/NumPy/SciPy stack.
- Global/Sequential/Static-Regret proposals may legitimately be infeasible under packet loss and tight capacity; this is measured via `valid_proposal_rate_percent` and is never repaired by a hidden fallback.
- Formal files are still written only after `run_experiment` completes; checkpoint/resume remains a separate future concern.
- The current one-page placeholder draft still describes the previously considered optimizer set and must be synchronized before final submission after new results exist.

### Next step
Pull and run the serial smoke:

```bash
git pull

time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 100 500 1000 \
  --trials 1 \
  --max-voters 5 \
  --workers 1
```

Then measure the all-voter parallel path:

```bash
time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 100 500 1000 \
  --trials 2 \
  --workers 4
```

If the extrapolation fits the budget, the canonical run is:

```bash
time python run_multitask_peer_cost_all_optimizers.py
```

Canonical no-argument meaning:

```text
100 robots
100..1000 tasks by 100
20 trials per point
100 voters
30% directed packet loss
Sequential Greedy + Global Greedy + Static Regret-2 Greedy + Auction
Hungarian Oracle reference
up to 4 trial worker processes
```

### Commit SHA
- `7148921855d63644dc39bf250acec350ac36e7c7` - dedicated fast capacitated heuristic owner.
- `a9ada9d2afaac53030952ed114df77eacb8e8f5c` - fast report-facing Experiment 2 owner and process-parallel runtime.
- `67aba0cc7394e5e04e5e60ac46ce7151dd0cbf0d` - canonical Experiment 2 specification.
- `83d8651f933f4620d5d47f53a4a1a1ff49131994` - report experiment suite.
- `9e8417aee1f946bb6341df196e3c717c5cbf6817` - continuity content for this change block.
