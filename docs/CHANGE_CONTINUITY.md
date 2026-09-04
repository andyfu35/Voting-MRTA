# Change Continuity

Historical continuity through `2026-09-03 - Fix 100-Task Auction Convergence` remains preserved verbatim in `docs/CHANGE_CONTINUITY_ARCHIVE_20260903.md`.

The subsequent active continuity through `2026-09-04 - Expand Experiment 2 to Five Optimizers under Lossy P2P Voting` is preserved exactly in Git at:

```text
commit: 7a4a1c247874f94589629134b25737521f7ff94e
blob:   50a0fd241e8bc61e04b8d3091112882135819e33
```

This documentation compaction changes no experiment behavior. It keeps the active continuity file bounded while preserving the full preceding state in Git history.

## 2026-09-04 - Parallelize Heavy Receiver-Local MILP/ACO Execution

### Purpose
Reduce the wall-clock cost of canonical Experiment 2 after the user observed that:

```bash
python run_multitask_peer_cost_all_optimizers.py
```

appeared to stall for a long time.

The cause is computational rather than a change in the experimental model: for every paired trial, both MILP and ACO + Local Search must solve up to 100 independent receiver-local incomplete assignment problems. Across 11 task counts and 100 trials, each added optimizer can therefore require up to 110,000 local solves.

This change improves only runtime execution and liveness reporting. It deliberately does not change scenario generation, packet loss, optimizer parameters, ACO seeds, proposal validity, Voting support, consensus, metrics, or report interpretation.

Before modification, repository-root `AGENTS.md` and `docs/AI_CHANGE_PROTOCOL.md` were checked in order and remain absent. `docs/CHANGE_CONTINUITY.md`, canonical `docs/multitask_voting_mrta.md`, and the actual owner `run_multitask_peer_cost_all_optimizers.py`, including `solve_extended_local_optimizer_proposals`, `optimizer_consensus_assignment`, `validate_zero_loss_optimizer_contract`, `run_trial`, and `run_experiment`, were read before the change.

### Files
- `run_multitask_peer_cost_all_optimizers.py`
- `docs/multitask_voting_mrta.md`
- `docs/CHANGE_CONTINUITY.md`

### Owner and named functions
Runtime execution remains owned by the canonical Experiment 2 module:

```text
run_multitask_peer_cost_all_optimizers.py
```

New/updated named boundaries:

- parallel runtime configuration contract: `validate_parallel_config`
- deterministic one-receiver heavy solve: `solve_receiver_local_proposal`
- proposal validation/storage boundary: `store_receiver_proposal`
- progress-only output owner: `report_receiver_progress`
- serial regression execution: `solve_serial_receiver_proposals`
- process-parallel execution: `solve_parallel_receiver_proposals`
- optimizer-family routing: `solve_extended_local_optimizer_proposals`
- shared Voting path: `optimizer_consensus_assignment`
- process-pool creation: `create_receiver_executor`
- process-pool shutdown: `shutdown_receiver_executor`
- paired scenario owner: `run_trial`
- sweep owner: `run_experiment`

The actual MILP algorithm remains owned by `run_multitask_optimizer_screening.solve_milp_assignment` and ACO remains owned by `run_multitask_optimizer_screening.solve_aco_assignment`. No optimizer implementation was copied.

### Responsibility movement
Previously, `solve_extended_local_optimizer_proposals` both selected the optimizer family and executed all MILP/ACO receiver solves serially.

Runtime execution is now separated:

```text
solve_extended_local_optimizer_proposals
    -> existing Greedy/Hungarian/Auction owner
    -> solve_serial_receiver_proposals        when workers=1
    -> solve_parallel_receiver_proposals      when workers>1
```

`solve_receiver_local_proposal` owns exactly one receiver-local heavy solver call. `store_receiver_proposal` owns post-solve validation and writes the result back to the original receiver index. `report_receiver_progress` owns terminal liveness output only.

No communication owner, cost owner, Voting owner, MILP owner, ACO owner, or assignment state machine moved.

### Preserved behavior
The following experimental behavior is unchanged:

- `RANDOM_SEED = 20260903`;
- 100 robots;
- 30% independent directed P2P scalar task-cost packet loss;
- 100 trials per task-count/method point;
- task counts `5,10,20,30,40,50,60,70,80,90,100`;
- scalar cost `0.05 + EuclideanDistance`;
- one simultaneous task per robot;
- one true cost matrix, one packet-loss tensor, one task order, and one tie-priority matrix generated per paired trial;
- existing task-count scenario RNG schedule `seed + task_count * 100003`;
- Greedy/Hungarian/Auction implementations and execution behavior;
- MILP objective, constraints, unavailable-edge handling, `mip_rel_gap=0.0`, and numerical tolerance;
- ACO ant count, iteration count, pheromone controls, local-search budget, candidate rules, and unavailable-edge handling;
- deterministic ACO per-receiver seed formula;
- local proposal validation;
- support counting and Voting consensus;
- report metrics and output file names.

No trial count, solver search budget, or problem size was reduced to obtain the speedup.

### Deliberately changed behavior
MILP and ACO receiver-local solves now use a persistent `ProcessPoolExecutor` when `parallel_workers > 1`.

Default runtime setting:

```text
DEFAULT_PARALLEL_WORKERS = min(4, available CPU count)
```

Only independent receiver-local MILP/ACO solves are submitted to the pool. Existing Greedy/Hungarian/Auction execution is untouched.

Worker completion order may differ, but every result returns its receiver index and is stored into that original index before proposal support is constructed. Therefore parallel completion order cannot change the support matrix.

ACO randomness remains deterministic per receiver:

```text
seed
+ 7000003
+ task_count * 100003
+ trial * 1009
+ receiver * 10000019
```

so process scheduling does not change its random stream.

A serial reference path is retained:

```bash
python run_multitask_peer_cost_all_optimizers.py --workers 1
```

The default parallel path is:

```bash
python run_multitask_peer_cost_all_optimizers.py
```

Terminal progress for MILP/ACO is now emitted every 25 completed receivers by default and each trial prints a start line. The interval can be changed with `--progress-every-receivers`.

### Diagnostic contract
Invalid runtime configuration fails at:

```text
owner=run_multitask_peer_cost_all_optimizers
function=validate_parallel_config
category=contract
code=INVALID_PARALLEL_WORKERS
```

or:

```text
owner=run_multitask_peer_cost_all_optimizers
function=validate_parallel_config
category=contract
code=INVALID_PROGRESS_INTERVAL
```

Unexpected process-worker failures fail at:

```text
owner=run_multitask_peer_cost_all_optimizers
function=solve_parallel_receiver_proposals
category=runtime
code=PARALLEL_RECEIVER_SOLVE_FAILED
```

A worker returning a result under the wrong receiver identity fails at:

```text
owner=run_multitask_peer_cost_all_optimizers
function=solve_parallel_receiver_proposals
category=state
code=WORKER_RECEIVER_MISMATCH
```

Named `ValueError` diagnostics raised by the true MILP/ACO owner are propagated rather than wrapped, preserving the first failing owner/function/category/code.

### Validation performed
- The updated source was re-fetched from GitHub after the write and inspected across the modified function boundaries and CLI wiring.
- A process-pool determinism harness compared serial and 4-worker execution using the same receiver-indexed deterministic RNG formula. Despite asynchronous completion order, the final receiver-indexed result arrays were exactly equal.
- The harness verified that result collection by receiver index is order-independent.
- The real repository smoke on the user's macOS environment is still required because that environment exercises SciPy HiGHS and the full ACO implementation inside spawned worker processes.

### Known limitations / unfinished risks
- Parallel execution reduces wall-clock time but does not reduce total computational work; the formal 100-trial sweep remains heavy.
- The default of 4 worker processes is intentionally conservative for a laptop. Users may increase `--workers`, but excessive process count can cause CPU/memory contention or solver-thread oversubscription.
- Runtime speedup is expected to be strongest for ACO and high task counts; exact speedup is machine-dependent.
- The current code does not yet persist partial formal results mid-sweep. If a later runtime failure aborts before `save_outputs`, that run remains non-authoritative.
- Direct-vs-Voting ablation remains unchanged and still covers Hungarian/Auction only.

### Next step
On the user's machine:

```bash
git pull
python run_multitask_peer_cost_all_optimizers.py --tasks 5 20 100 --trials 3 --workers 4
```

Confirm that spawned workers run successfully, progress is visible, and all six report columns are produced.

For a small serial/parallel decision-equivalence check, run:

```bash
python run_multitask_peer_cost_all_optimizers.py --tasks 5 --trials 1 --workers 1
python run_multitask_peer_cost_all_optimizers.py --tasks 5 --trials 1 --workers 4
```

The quality tables should be identical for the fixed seed/configuration. Runtime is not expected to match.

Only after the smoke passes should the canonical 100-trial run be restarted from the beginning.

### Commit SHA
- `f4e347ab54db7529a99eca07723835913f323010` - parallelized receiver-local MILP/ACO execution and added progress/runtime controls.
- `e9c270ae0ec0a10effc81ed267715358069b21e2` - updated canonical Experiment 2 runtime execution specification.
- continuity update commit: this file's commit.
