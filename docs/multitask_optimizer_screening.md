# Multi-Task Optimizer Family Screening

This is a bounded optimizer-screening experiment used before integrating additional optimizer families into the lossy P2P Voting experiment.

## Research question

Before communication loss and multi-view Voting are involved, how do different optimizer families behave as simultaneous task count increases for the same 100-robot capacity-one assignment problem?

This screening compares solution quality and solver runtime only. It intentionally removes P2P packet loss and proposal consensus so optimizer behavior is not confounded with communication behavior.

## Controlled settings

- Robots: `100`
- Simultaneous task counts: `5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100`
- Trials per task-count/method point: `100`
- Robot simultaneous capacity: one task
- Cost model: the same normalized 2-D Euclidean spatial cost owner used by the canonical multi-task P2P experiment
- Packet loss: disabled in this screening
- Voting / proposal consensus: disabled in this screening
- Route planning, execution noise, retransmission, permanent failure, deadlines, and robust uncertainty modeling remain excluded.

## Shared cost and feasibility owners

This experiment imports existing canonical owners from `run_multitask_peer_cost_experiment.py` rather than copying them:

- spatial cost generation: `generate_spatial_cost_matrix`
- exact Hungarian assignment: `solve_hungarian_assignment`
- sequential Greedy baseline: `solve_sequential_greedy`
- assignment feasibility / true cost: `assignment_total_cost`
- experiment configuration validation: `validate_experiment_config`

The new experiment owner is `run_multitask_optimizer_screening.py`.

## Compared methods

### Hungarian

The exact linear assignment reference. It minimizes total robot-task travel cost under:

```text
one robot per task
one simultaneous task per robot
```

Hungarian defines `0%` optimality gap for this screening.

### MILP

The same assignment objective is written as binary variables:

```text
x_ij in {0,1}
```

with:

```text
sum_i x_ij = 1   for every task j
sum_j x_ij <= 1  for every robot i
```

The owner `solve_milp_assignment` uses `scipy.optimize.milp` / HiGHS.

Because the current problem contains only the linear capacity-one assignment constraints, MILP is expected to reach the same mathematical optimum as Hungarian. This is intentional: the screening measures the computational cost of using a more general mathematical optimizer on a problem for which Hungarian has a specialized polynomial solver.

`build_milp_assignment_model` caches the shape-only constraint model so repeated trials do not rebuild the same sparse constraint matrices.

The MILP owner explicitly sets:

```text
mip_rel_gap = 0.0
```

so HiGHS is not allowed to stop merely because a nonzero user MIP-gap target has been reached.

HiGHS and the SciPy MILP interface still operate in double precision. Therefore MILP objective equality is evaluated with a solver-specific numerical tolerance:

```text
MILP_NUMERICAL_TOLERANCE_PERCENT = 1e-6
```

This corresponds to a relative objective difference of `1e-8`. The tighter generic assignment tolerance used elsewhere in the repository remains unchanged.

The separate MILP tolerance exists only to classify numerically indistinguishable MILP/Hungarian objective values; it does not alter the assignment, objective, cost matrix, or any other optimizer's metric.

### ACO + Local Search

The swarm/metaheuristic method is explicitly labeled `ACO + Local Search`; it is not presented as a pure Ant System baseline.

ACO construction uses:

```text
probability proportional to pheromone^alpha * (1 / cost)^beta
```

with a bounded cheapest-robot candidate list. High-regret tasks are constructed first so assignments that have one especially important robot choice are less likely to be blocked by an earlier arbitrary assignment.

Default fixed ACO controls are:

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

These values are fixed across all task counts; they are not re-tuned per data point.

After the best ant of an iteration is selected, `improve_aco_assignment_locally` performs a bounded local refinement using two move families:

1. replace one assigned robot with an unused robot when this lowers cost;
2. swap the robots assigned to two tasks when this lowers cost.

This local-search responsibility is kept in a separate named function and is reported in the method label so the experiment does not hide the hybrid nature of the optimizer.

### Greedy Baseline

The existing sequential Greedy assignment is retained as a non-optimal heuristic baseline. It is not treated as an exact optimizer.

## Exact optimizer contract

Before the sweep, `validate_exact_optimizer_contract` checks task counts `1, 5, 50, 100`.

MILP must match Hungarian total cost within `MILP_NUMERICAL_TOLERANCE_PERCENT = 1e-6%`. If it does not, the experiment aborts with:

```text
owner=run_multitask_optimizer_screening
function=validate_exact_optimizer_contract
category=planning
code=MILP_NOT_HUNGARIAN_EXACT
```

The diagnostic includes both the expected maximum absolute percentage gap and the actual percentage gap.

The same numerical-exactness check is also enforced for every formal MILP trial through `run_trial / planning / MILP_NOT_EXACT`.

ACO is not required to match Hungarian because it is a stochastic metaheuristic.

## Evaluation metrics

### Average optimality gap

```text
gap (%) = 100 * (method_cost - hungarian_cost) / hungarian_cost
```

Lower is better.

### Optimal-cost match

Percentage of trials whose total cost equals Hungarian within the numerical tolerance owned by that solver family.

- Hungarian, ACO, and Greedy retain the existing generic tolerance.
- MILP uses `1e-6%` only because HiGHS double-precision integer optimization can return an alternative nearly tied integral assignment whose true objective differs at floating-point scale.

Higher is better.

### Near-optimal within 5%

Percentage of trials satisfying:

```text
gap <= 5%
```

### Exact optimal assignment

Percentage of trials whose complete task-to-robot assignment exactly matches Hungarian. This is supporting only because equal-cost or numerically indistinguishable alternative assignments are possible.

### Runtime

Each optimizer call is timed with `time.perf_counter()`.

The report saves both average and median runtime in milliseconds. Runtime is measured only around the solver call; cost generation and report code are outside the timed boundary.

## Paired trial contract

For every task-count/trial pair, all methods receive the exact same true cost matrix.

The trial seed is:

```text
seed + task_count * 100003 + trial * 1009
```

ACO receives a deterministic separate random stream derived from that same paired trial seed, so repeated runs are reproducible without changing the other methods' inputs.

## Run

Canonical full screening:

```bash
python run_multitask_optimizer_screening.py
```

Fast smoke test:

```bash
python run_multitask_optimizer_screening.py --tasks 5 20 100 --trials 3
```

ACO search-budget sensitivity can be tested explicitly, for example:

```bash
python run_multitask_optimizer_screening.py --tasks 100 --trials 20 --aco-ants 20 --aco-iterations 30
```

Changing ACO controls creates a sensitivity run and should not be mixed with the canonical default dataset.

## Outputs

```text
results/multitask_optimizer_screening/data/
results/multitask_optimizer_screening/figures/
```

Primary report CSVs:

```text
report_average_optimality_gap_percent.csv
report_optimal_cost_match_percent.csv
report_near_optimal_5pct_percent.csv
report_average_runtime_ms.csv
```

Raw and summary data:

```text
optimizer_screening_raw.csv
optimizer_screening_summary.csv
```

## Numerical-tolerance regression case

The first full user run reached task counts 5 through 90 and then stopped at:

```text
owner=run_multitask_optimizer_screening
function=run_trial
category=planning
code=MILP_NOT_EXACT
details=tasks=100 trial=7 gap_percent=5.2467347189941e-07
```

A focused reproduction with the same canonical seed showed:

- Hungarian true total cost: approximately `15.151402216370842`.
- MILP true total cost: approximately `15.15140229586623`.
- absolute cost difference: approximately `7.95e-8`.
- percentage gap: `5.2467347189941e-7%`.
- the MILP solution was integral and feasible; only two tasks exchanged robots relative to Hungarian in a nearly tied pair.
- across the complete 100-trial `tasks=100` focused regression, this was the only trial above the repository's older `1e-8%` generic equality threshold; the maximum observed MILP gap was the same `5.2467347189941e-7%`.

This case is treated as numerical equivalence under the MILP-specific `1e-6%` tolerance, not as evidence that MILP is a different approximate optimizer.

## Scope boundary

This experiment is only an optimizer-family screening stage. It does not test communication robustness and it must not be used to claim that MILP or ACO improves the existing Voting mechanism.

After the screening result is inspected, only optimizer families worth the extra computation should be integrated into a separate paired P2P/Voting experiment. Robust Optimization is deliberately excluded from this stage because it requires a separate, justified uncertainty model for missing costs.
