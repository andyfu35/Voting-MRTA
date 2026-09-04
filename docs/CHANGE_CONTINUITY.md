# Change Continuity

Historical continuity through `2026-09-04 - Replace Multiple Greedy Curves with Four Distinct Fast Optimizer Families` is preserved exactly in Git at:

```text
commit: a9ebad1112434fe22c1ac01e02c0fd5222b00afa
blob:   dad71c50d9dd74f8394e94aaba917b9f2c3879fd
```

## 2026-09-04 - Add Separate Per-Optimizer Runtime Benchmark

### Purpose
The user wants computation time for each of the four canonical Experiment 2 local optimizer families so those measured values can later be added to the one-page CACS paper.

The existing formal Experiment 2 terminal output only reported whole-run wall/user/system time. It did not separate Greedy, Hungarian, Min-Cost Flow, and Sinkhorn compute time, so no per-method runtime values may be inferred from the previous `9:22.56` whole-run measurement.

A separate runtime benchmark was added instead of changing the formal assignment-quality state machine. It measures the actual existing receiver-local optimizer routing boundary on paired incomplete cost views.

Before modification, repository-root `AGENTS.md` and `docs/AI_CHANGE_PROTOCOL.md` were checked in that order and remain absent. Then `docs/CHANGE_CONTINUITY.md`, canonical `docs/multitask_voting_mrta.md`, report suite `docs/report_experiment_suite.md`, actual Experiment 2 owner `run_multitask_peer_cost_all_optimizers.py`, and exact function `solve_voter_batch_proposals` plus its surrounding receiver-view functions were read before writing.

### Files
- `run_multitask_optimizer_runtime_benchmark.py` - new runtime-measurement owner.
- `docs/multitask_optimizer_runtime_benchmark.md` - canonical timing-boundary and reporting contract.
- `docs/CHANGE_CONTINUITY.md` - this continuity record.

The formal Experiment 2 owner and all optimizer implementations are unchanged.

### Owner and named functions
Runtime measurement is owned by:

```text
run_multitask_optimizer_runtime_benchmark.py
```

Named boundaries:

- timing input validation: `validate_timing_config`
- deterministic method-order rotation: `rotate_method_order`
- local solver timing boundary: `measure_optimizer_batch_runtime`
- one-time excluded warm-up: `warm_up_optimizer_paths`
- one paired timing trial: `run_timing_trial`
- timing aggregation: `summarize_timings`
- report table construction: `runtime_report_table`
- timing persistence: `save_timing_outputs`
- benchmark orchestration: `run_benchmark`

The timed solver route remains owned by:

```text
run_multitask_peer_cost_all_optimizers.py::solve_voter_batch_proposals
```

That route still delegates to the true Greedy, Hungarian, Min-Cost Flow, and Sinkhorn owners. No second optimizer implementation or assignment state machine was created.

### Responsibility movement
No existing responsibility moved.

Timing is a new, separate concern with its own owner. The benchmark imports the canonical Experiment 2 communication/view helpers and the canonical `solve_voter_batch_proposals` routing function rather than copying those implementations.

The timing boundary surrounds only the receiver-local optimizer route. Communication sampling, receiver-view materialization, Voting support/consensus, Oracle computation, plotting, and file I/O are deliberately outside the measured interval.

### Preserved behavior
- formal Experiment 2 remains 100 physical robots;
- formal task batches remain `100, 200, ..., 1000`;
- formal message-loss setting remains 30%;
- formal quality experiment remains 20 trials per task point and all 100 voters;
- Greedy/Hungarian/Min-Cost-Flow/Sinkhorn algorithms are unchanged;
- paired seed, voter selection, and directed message-loss helpers are reused;
- formal assignment-quality CSVs and figures are unchanged;
- Voting consensus behavior is unchanged;
- Experiment 1 and Experiment 3 are unchanged.

### Deliberately added behavior
A new command is available:

```bash
python run_multitask_optimizer_runtime_benchmark.py
```

Canonical runtime-benchmark settings are:

```text
tasks = 100, 200, ..., 1000
runtime trials = 5 per task point
voters = all 100 robots
message loss = 30%
receiver batch size = 4
single process
```

The benchmark performs a one-time excluded warm-up for all methods, then uses `time.perf_counter()` around each call to `solve_voter_batch_proposals`.

To reduce fixed-order timing bias, method order is rotated deterministically by receiver batch. This order rotation does not change scenario, communication, or optimizer inputs.

Primary runtime field:

```text
local_optimizer_runtime_ms_per_voter
```

This is the mean receiver-local optimizer compute time and is the value intended for the paper after the user runs the benchmark on the target Mac.

New outputs:

```text
results/multitask_peer_cost_fixed100_workload/data/optimizer_runtime_raw.csv
results/multitask_peer_cost_fixed100_workload/data/optimizer_runtime_summary.csv
results/multitask_peer_cost_fixed100_workload/data/optimizer_runtime_overall.csv
```

### Diagnostic contract
Runtime-benchmark failures use:

```text
owner=run_multitask_optimizer_runtime_benchmark
function=<named function>
category=<data|time|state|dependency|planning|safety|runtime|contract>
code=<named code>
```

Important first-failure boundaries:

```text
function=validate_timing_config category=contract code=INVALID_TASK_COUNT
function=validate_timing_config category=contract code=INVALID_MESSAGE_LOSS
function=rotate_method_order category=contract code=EMPTY_METHOD_SET
function=measure_optimizer_batch_runtime category=time code=INVALID_MEASURED_RUNTIME
function=warm_up_optimizer_paths category=contract code=WARMUP_OUTPUT_SHAPE_MISMATCH
```

Diagnostics from the existing optimizer owners continue to propagate unchanged.

### Validation performed
- Re-fetched `run_multitask_optimizer_runtime_benchmark.py` from GitHub after creation and inspected the complete owner structure and named boundaries.
- Confirmed the benchmark calls the existing `solve_voter_batch_proposals` route rather than duplicating the four optimizer algorithms.
- Confirmed timing begins immediately before that route and ends immediately after it.
- Confirmed communication sampling and receiver-view construction occur before the timer.
- Confirmed the benchmark writes separate runtime files and does not overwrite the formal Experiment 2 quality CSVs.
- Real OR-Tools/SciPy runtime must still be executed on the user's Mac; this environment is not being used as the authoritative timing machine.

### Unfinished risks
- Timing values do not yet exist; the user must run the benchmark on the target Mac.
- `perf_counter()` measures elapsed local solve time on that machine and can vary with CPU load, thermal state, and background activity.
- The reported value is local optimizer compute time per receiver, not end-to-end multi-robot allocation latency.
- Because the benchmark is intentionally single-process, its values should not be compared directly with the previous four-worker whole-experiment wall time.
- The CACS paper must not be updated with per-method runtime until the measured benchmark output is returned and checked.

### Next step
On the Mac:

```bash
git pull
pip install -r requirements.txt
python run_multitask_optimizer_runtime_benchmark.py
```

Then send the terminal section:

```text
Average local optimizer compute time per voter (ms):
...
Overall mean across the selected task points (ms per voter):
...
```

The three generated runtime CSVs may also be supplied if detailed verification is needed.

### Commit SHA
- `1f3b3f8af15c054e58836fda633d7ba498eebbbb` - add paired local optimizer runtime benchmark owner.
- `60f79c77994acb940f881add5dca9c13cf49dddf` - add canonical runtime benchmark specification.
- continuity update commit: this file's commit.
