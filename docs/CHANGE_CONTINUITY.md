# Change Continuity

Historical continuity through `2026-09-04 - Narrow Experiment 2 to 100-Task Steps and Hand Off to Windows Runtime` is preserved exactly in Git at:

```text
commit: 5534ccfbb57cae6bdf0d31aa0a1b881f2e577b92
blob:   d96ca1fc86de4a0c45a5e570a28bdbbcd6b9af14
```

## 2026-09-04 - Replace Slow Lossy Optimizers with Fast Heuristics and Parallel Trials

### Purpose
The user decided to keep Experiment 2 on the Mac and target an approximately one-hour end-to-end runtime. The preceding fixed-100-robot design still executed receiver-local Hungarian, MILP, and ACO + Local Search. A real macOS probe at `1000` tasks, one trial, one voter, with all five optimizer families took about `85.44 s`, making the full all-voter repeated experiment impractical.

This change narrows the report-facing research question to communication loss + Voting workload scaling and replaces the expensive/redundant receiver-local solver set with computationally scalable local decision rules:

```text
Voting Sequential Greedy
Voting Global Greedy
Voting Static Regret-2 Greedy
Voting Auction
```

`Hungarian Oracle` remains the full-information minimum-cost reference only.

The canonical Experiment 2 trial count is deliberately reduced from `100` to `20` per task point, while retaining all `100` physical robots as voters. Independent `(task_count, trial)` jobs are parallelized with a bounded process pool, defaulting to up to four workers.

Before modification, repository-root `AGENTS.md` and `docs/AI_CHANGE_PROTOCOL.md` were checked in that order and remain absent. Then `docs/CHANGE_CONTINUITY.md`, canonical `docs/multitask_voting_mrta.md`, the actual Experiment 2 owner `run_multitask_peer_cost_all_optimizers.py`, the exact capacity/routing/trial/report functions, the existing Greedy/Auction owner functions in `run_multitask_peer_cost_experiment.py`, and `docs/report_experiment_suite.md` were read before writing.

### Files
- new optimizer owner: `run_multitask_workload_heuristics.py`
- canonical Experiment 2 owner: `run_multitask_peer_cost_all_optimizers.py`
- canonical specification: `docs/multitask_voting_mrta.md`
- report suite: `docs/report_experiment_suite.md`
- continuity: `docs/CHANGE_CONTINUITY.md`

The existing optimizer-screening module and historical Experiment 1/3 owners were not modified.

### Owner and named functions

#### Fast heuristic owner

```text
run_multitask_workload_heuristics.py
```

Named boundaries:

- input/data/capacity validation: `validate_capacitated_problem`
- sequential heuristic: `solve_sequential_greedy_capacitated`
- global cheapest-edge heuristic: `solve_global_greedy_capacitated`
- one-time two-best regret ordering: `compute_static_regret2_priority`
- static regret heuristic: `solve_regret2_greedy_capacitated`
- method routing: `solve_capacitated_heuristic`
- receiver batch execution: `solve_capacitated_heuristic_batch`

#### Experiment 2 owner

```text
run_multitask_peer_cost_all_optimizers.py
```

Important changed/added named boundaries:

- canonical method selection: `resolve_voting_methods`
- workload/runtime validation: `validate_workload_config`
- capacity representation: `resolve_robot_capacity`, `build_capacity_slot_cost_matrix`, `build_capacity_slot_cost_views`, `map_slot_assignments_to_robots`
- physical capacity validation/evaluation: `validate_capacity_assignment`, `assignment_total_cost_with_capacity`
- full-information reference: `solve_capacity_oracle`
- communication view ownership: `sample_voter_batch_visibility`, `build_voter_batch_cost_views`
- local solver routing: `solve_voter_batch_proposals`
- proposal support: `accumulate_proposal_support`, `collect_voting_support`
- final capacitated Voting consensus: `solve_capacity_support_consensus`, `finalize_voting_assignments`
- bounded heuristic gate: `validate_zero_loss_heuristic_contracts`
- bounded exact Auction gate: `validate_zero_loss_auction_contract`
- paired trial: `run_trial`
- process worker boundary: `run_trial_job`
- deterministic job planning: `build_trial_jobs`
- nested-thread control: `configure_worker_thread_environment`
- parent progress boundary: `report_trial_completion`
- runtime execution: `execute_trial_jobs`
- sweep/report: `run_experiment`, `summarize_results`, `save_outputs`

The existing Auction algorithm itself remains owned by:

```text
run_multitask_peer_cost_experiment.py::solve_batched_auction_assignments
```

The full-information Oracle remains owned by:

```text
run_multitask_peer_cost_experiment.py::solve_hungarian_assignment
```

### Responsibility movement
The three new greedy algorithms are not added as wrappers around an unrelated optimizer owner. They have a dedicated optimizer owner, `run_multitask_workload_heuristics.py`, and operate directly on the physical `100 x T` receiver-local matrix plus explicit physical capacity.

This removes unnecessary slot expansion from Sequential/Global/Static-Regret heuristics. Capacity slots remain only where required by existing capacity-one exact owners: full-information Hungarian Oracle, Voting Auction, and final support consensus.

Trial scheduling is separated from trial semantics. `run_trial` still owns one deterministic paired trial. `build_trial_jobs` creates independent jobs, and `execute_trial_jobs` owns serial/process-pool execution. Worker count never enters a seed formula.

### Preserved behavior
- physical robots remain exactly `100`;
- canonical task loads remain `100, 200, ..., 1000`;
- directed P2P scalar cost-message loss remains `30%`;
- scalar cost remains `0.05 + EuclideanDistance`;
- workload capacity remains `ceil(tasks/100)`;
- task counts remain allocation batches, not simultaneous physical execution claims;
- every physical receiver always knows its own sender row;
- missing receiver-local costs remain `+inf` unavailable edges;
- task delivery remains reliable;
- final proposal collection remains reliable/in-window;
- all Voting methods share identical cost matrix, voter IDs, packet-loss realization, task order, tie priority, and capacity within a trial;
- support is accumulated on physical robot/task pairs;
- final consensus maximizes proposal support under the same physical capacity;
- true task cost is not used as a hidden consensus tie-break;
- full-information Hungarian remains the minimum-cost reference;
- primary report metric remains direct percentage cost error from the minimum;
- `<=5%`, optimal-cost match, exact assignment, and valid-proposal-rate metrics remain supporting CSV outputs;
- output root remains `results/multitask_peer_cost_fixed100_workload/`;
- Experiment 1 and Experiment 3 remain unchanged.

### Deliberately changed behavior

#### Canonical Voting optimizer set
Removed from report-facing lossy Experiment 2:

```text
Voting Hungarian
Voting MILP
Voting ACO + Local Search
```

These implementations were not deleted. MILP and ACO remain in the separate complete-information optimizer screening. Receiver-local Hungarian is omitted because Voting Auction already provides the exact local assignment-family comparison and previous measurements showed the exact curves were largely redundant.

New report-facing heuristic set:

```text
Voting Sequential Greedy
Voting Global Greedy
Voting Static Regret-2 Greedy
Voting Auction
```

`Static Regret-2` is intentionally named: its best-vs-second-best receiver-visible regret is computed once per task, ties use the paired task order, and capacity-aware cheapest assignment then follows that fixed priority. It is not claimed to be dynamically recomputed Regret-2.

#### Canonical trial count
Experiment 2 formal default changed from:

```text
100 trials / task point
```

to:

```text
20 trials / task point
```

All `100` physical voters remain enabled in canonical mode. `--trials` can increase the count if measured runtime permits, but results must be reported with the actual trial count.

#### Trial parallelism
New CLI/runtime control:

```text
--workers N
```

Default:

```text
min(4, available CPU count)
```

Independent paired trials execute through `ProcessPoolExecutor`. `--workers 1` is the serial diagnostic path.

When more than one worker is used, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` default to `1` only when the user has not explicitly set them. This is a runtime oversubscription guard, not an experiment-semantic change.

Parallel workers suppress receiver-level progress output. The parent reports completed trials to avoid interleaved console output.

#### Receiver batching
Default receiver batch size changes from `8` to `4` to reduce per-process peak memory when several trials execute concurrently. It remains a runtime/memory control only.

#### CLI simplification
The report-facing Experiment 2 owner no longer exposes `--include-milp`, `--only-milp`, or `--include-aco`. Those methods are no longer part of this experiment's canonical question.

### Diagnostic contract
Existing Experiment 2 diagnostics retain the standard form:

```text
owner=run_multitask_peer_cost_all_optimizers
function=<named function>
category=<data|time|state|dependency|planning|safety|runtime|contract>
code=<named code>
```

New runtime validation example:

```text
owner=run_multitask_peer_cost_all_optimizers
function=validate_workload_config
category=contract
code=INVALID_WORKER_COUNT
```

New heuristic-owner diagnostics use:

```text
owner=run_multitask_workload_heuristics
function=<named function>
category=<data|state|contract>
code=<named code>
```

Representative examples:

```text
function=validate_capacitated_problem category=data code=INVALID_HEURISTIC_COST
function=validate_capacitated_problem category=state code=CAPACITY_EXCEEDED
function=solve_capacitated_heuristic category=contract code=UNKNOWN_METHOD
```

Existing Auction diagnostics continue to propagate from `run_multitask_peer_cost_experiment`; no wrapper replaces the first failing owner.

### Validation performed
- The newly created `run_multitask_workload_heuristics.py` was re-fetched from GitHub and inspected across validation, Sequential Greedy, Global Greedy, static Regret-2 priority, routing, and receiver-batch execution.
- The rewritten Experiment 2 owner was re-fetched and inspected across imports/constants, capacity functions, solver routing, bounded zero-loss gates, deterministic trial seeding, process-job creation, process execution, report aggregation, and CLI defaults.
- Canonical/report documents were synchronized with the new method set, 20-trial formal contract, and parallel runtime design.
- A container-side repository clone/compile attempt was made, but the container still cannot resolve `github.com`; therefore no real repository bytecode/runtime test could be executed there.
- The preceding real macOS all-five-family probe remains useful only as the runtime motivation for this redesign, not as validation of the new method set.

### Known limitations / unfinished risks
- The approximately one-hour target is not yet validated. Voting Auction may still dominate runtime at 700-1000 tasks even after MILP/ACO/Hungarian local solves are removed.
- Process parallelism increases peak memory. The default of four workers and receiver batch size four is intentionally bounded, but a lower `--workers` or `--voter-batch-size` may be required on memory-constrained Macs.
- macOS process startup and BLAS behavior depend on the installed Python/NumPy/SciPy stack; actual scaling must be measured on the user's environment.
- Static Regret-2 is a fixed-priority heuristic, not dynamic regret recomputation; the paper must use the exact method name/definition.
- Global/Sequential/Static-Regret local proposals can legitimately become infeasible under packet loss and tight capacity. This is measured through `valid_proposal_rate_percent` rather than repaired with a hidden fallback.
- Formal output is still written after `run_experiment` finishes; checkpoint/resume is not part of this change.
- The existing one-page placeholder draft still mentions the previously considered optimizer set and must be synchronized before final submission after the new results are available.

### Next step
Pull the new code on the Mac and run the new serial smoke:

```bash
git pull

time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 100 500 1000 \
  --trials 1 \
  --max-voters 5 \
  --workers 1
```

If that passes, measure the actual all-voter parallel path:

```bash
time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 100 500 1000 \
  --trials 2 \
  --workers 4
```

If the measured extrapolation is within the available budget, run the canonical Experiment 2:

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
- `7148921855d63644dc39bf250acec350ac36e7c7` - added the dedicated fast capacitated heuristic owner.
- `a9ada9d2afaac53030952ed114df77eacb8e8f5c` - rewrote report-facing Experiment 2 for the fast method set, physical heuristic routing, 20-trial default, and bounded process parallelism.
- `67aba0cc7394e5e04e5e60ac46ce7151dd0cbf0d` - synchronized the canonical Experiment 2 specification.
- `83d8651f933f4620d5d47f53a4a1a1ff49131994` - synchronized the report experiment suite.
- continuity update commit: pending metadata follow-up.
