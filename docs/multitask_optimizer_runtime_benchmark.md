# Experiment 2 Local Optimizer Runtime Benchmark

This document defines the report-facing runtime measurement for the four canonical local optimizer families used by Experiment 2. It is a separate measurement concern and does not change the formal assignment-quality experiment.

## Purpose

Measure the computation time of the receiver-local optimizer path for:

```text
Voting Greedy
Voting Hungarian
Voting Min-Cost Flow
Voting Sinkhorn + Rounding
```

The benchmark answers: how long does one robot need, on average, to compute one local proposal from its incomplete cost view?

## Owner

```text
run_multitask_optimizer_runtime_benchmark.py
```

The benchmark reuses the existing Experiment 2 routing boundary:

```text
run_multitask_peer_cost_all_optimizers.py::solve_voter_batch_proposals
```

It does not duplicate Greedy, Hungarian, Min-Cost Flow, or Sinkhorn implementations.

## Timing boundary

The timer uses Python `time.perf_counter()` and surrounds only:

```text
solve_voter_batch_proposals(...)
```

Therefore the reported local optimizer time includes the representation work that belongs to that method's local route, such as Hungarian capacity-slot expansion and Min-Cost Flow graph construction.

It deliberately excludes:

- physical scenario generation;
- peer-to-peer message-loss sampling;
- receiver-local cost-view materialization;
- proposal-support accumulation;
- final Voting consensus;
- the full-information Hungarian Oracle;
- plotting and file I/O.

This separation keeps the reported number interpretable as local optimizer compute time rather than whole-experiment wall time.

## Canonical benchmark settings

```text
physical robots = 100 fixed
task batches = 100, 200, ..., 1000
peer-to-peer message loss = 30%
timing trials per task point = 5
voters = all 100 robots
receiver batch size = 4
parallel workers = none; timing benchmark runs in one process
```

The five timing trials are a dedicated runtime sample and are not a replacement for the 20-trial formal assignment-quality dataset.

## Pairing

For each timing trial, all four methods receive the same:

- generated physical cost matrix;
- selected voters;
- directed message-loss realization;
- task order;
- robot capacity.

The benchmark uses the same Experiment 2 seed formula and communication helpers. Method execution order is rotated deterministically by receiver batch so a fixed first/last position does not systematically favor one method.

## Warm-up

Before measurement, every method is called once on a bounded complete-information case. This warm-up is excluded from the reported timing and reduces one-time library/setup effects.

## Metrics

Primary runtime metric:

```text
local_optimizer_runtime_ms_per_voter
```

For one trial and method:

```text
1000 * summed timed optimizer seconds / number of voters
```

The task-specific summary reports the mean of that value across timing trials.

The overall report value is the mean across the selected task points and timing trials. Because the canonical benchmark uses the same number of voters and trials for every task point, every workload point receives equal weight.

A second field records the total local optimizer seconds per trial for diagnostic use.

## Outputs

```text
results/multitask_peer_cost_fixed100_workload/data/optimizer_runtime_raw.csv
results/multitask_peer_cost_fixed100_workload/data/optimizer_runtime_summary.csv
results/multitask_peer_cost_fixed100_workload/data/optimizer_runtime_overall.csv
```

The benchmark also prints a task-by-method timing table and one overall mean table to the terminal.

## Canonical run

After pulling the code and synchronizing dependencies:

```bash
git pull
pip install -r requirements.txt
python run_multitask_optimizer_runtime_benchmark.py
```

A quick smoke may use:

```bash
python run_multitask_optimizer_runtime_benchmark.py \
  --tasks 100 500 1000 \
  --trials 1 \
  --max-voters 5
```

Quick-smoke timing must not be quoted as the final paper timing.

## Diagnostic contract

Runtime-benchmark errors use:

```text
owner=run_multitask_optimizer_runtime_benchmark
function=<named function>
category=<data|time|state|dependency|planning|safety|runtime|contract>
code=<named code>
```

Important boundaries include:

```text
function=validate_timing_config category=contract code=INVALID_TASK_COUNT
function=validate_timing_config category=contract code=INVALID_MESSAGE_LOSS
function=rotate_method_order category=contract code=EMPTY_METHOD_SET
function=measure_optimizer_batch_runtime category=time code=INVALID_MEASURED_RUNTIME
function=warm_up_optimizer_paths category=contract code=WARMUP_OUTPUT_SHAPE_MISMATCH
```

Optimizer-owner diagnostics still propagate from their original owner modules.

## Reporting rule

When the values are added to the paper, describe them as **mean local optimizer computation time per robot/receiver** on the tested Mac. Do not call these values whole-system task-allocation latency or end-to-end distributed execution time, because communication, consensus, and robot execution are outside the timing boundary.
