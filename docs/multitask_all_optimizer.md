# Multi-Task All-Optimizer Comparison

This is the report-facing complete-information optimizer comparison for the single-cost MRTA study.

## Research question

For the same 100-robot capacity-one assignment problem and the same 100 paired scenarios at each task count, how do five optimizer families compare in solution quality and computation time as simultaneous task load increases?

## Controlled settings

- Robots: `100`
- Simultaneous task counts: `5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100`
- Trials per task count: `100`
- Cost: `C_ij = 0.05 + EuclideanDistance(robot_i, task_j)`
- Robot simultaneous capacity: one task
- Packet loss: disabled in this optimizer-family comparison
- Voting / proposal consensus: disabled in this optimizer-family comparison
- Random seed: `20260903`
- Robust / multi-objective cost models are excluded.

## Paired-scenario fairness contract

For every `(task_count, trial)` pair, the experiment generates the scenario exactly once:

```text
trial_seed = seed + task_count * 100003 + trial * 1009
```

That one scenario produces one ground-truth cost matrix and one paired task order. Every compared optimizer receives the exact same matrix and order where order is applicable.

The methods do not independently regenerate robot positions, task positions, or costs.

ACO is stochastic internally, so it receives a deterministic separate search stream derived from the same trial seed:

```text
aco_seed = trial_seed + 7000003
```

The ACO search seed does not alter any other optimizer input.

## Compared methods

### Hungarian

Specialized exact linear-assignment reference. Hungarian defines the ground-truth minimum total cost for this experiment.

### Auction

The existing epsilon-scaling Bertsekas-style Auction owner from `run_multitask_peer_cost_experiment.py` is reused for one complete-information receiver matrix. The implementation is not copied.

Auction is required to match Hungarian cost at complete information within the existing generic exact-cost tolerance.

### MILP

The existing binary mixed-integer assignment owner from `run_multitask_optimizer_screening.py` is reused without copying a second MILP implementation.

MILP is required to match Hungarian cost within the existing solver-specific numerical tolerance:

```text
MILP_NUMERICAL_TOLERANCE_PERCENT = 1e-6
```

### ACO + Local Search

The existing fixed-budget stochastic optimizer owner from `run_multitask_optimizer_screening.py` is reused. The method remains explicitly labeled `ACO + Local Search`; it is not claimed to be a pure Ant System.

Canonical ACO controls remain:

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

These controls are fixed for every task count.

### Greedy Baseline

The existing sequential Greedy owner is retained as the low-computation heuristic baseline.

## Exact optimizer gate

Before formal data generation, `validate_exact_optimizer_contract` checks task counts `1, 5, 50, 100`.

- Auction must match Hungarian.
- MILP must match Hungarian within its solver-specific numerical tolerance.
- ACO and Greedy are not required to be exact.

Any exact-solver failure aborts at:

```text
owner=run_multitask_all_optimizer_experiment
function=validate_exact_optimizer_contract
category=planning
```

The same Auction/MILP exactness check is repeated in every formal paired trial.

## Evaluation metrics

Primary quality metrics:

```text
average optimality gap (%)
optimal-cost match (%)
near-optimal within 5% (%)
```

Supporting metrics:

```text
exact optimal assignment (%)
average optimizer runtime (ms)
median optimizer runtime (ms)
```

Optimality gap is:

```text
gap (%) = 100 * (method_cost - Hungarian_cost) / Hungarian_cost
```

`0%` means the method reached the same optimal total cost as Hungarian.

## Run

Smoke test:

```bash
python run_multitask_all_optimizer_experiment.py --tasks 5 20 100 --trials 3
```

Canonical report run:

```bash
python run_multitask_all_optimizer_experiment.py
```

Only the canonical 100-trial run belongs in the report. Smoke-test values must not be mixed into the report tables.

## Outputs

```text
results/multitask_all_optimizer/data/
results/multitask_all_optimizer/figures/
```

Primary report CSVs:

```text
report_average_optimality_gap_percent.csv
report_optimal_cost_match_percent.csv
report_near_optimal_5pct_percent.csv
report_exact_optimal_assignment_percent.csv
report_average_runtime_ms.csv
```

Raw and summary data:

```text
all_optimizer_raw.csv
all_optimizer_summary.csv
```

## Scope boundary

This experiment isolates optimizer-family behavior under one complete-information scalar distance cost. It does not test communication loss or Voting robustness.

Communication robustness remains owned by the P2P multi-task experiment, and proposal-consensus benefit remains owned by the Direct-vs-Voting ablation. These experiments should be reported as separate controlled questions rather than merged into one uncontrolled table.
