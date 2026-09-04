# Multi-Task Optimizer Family Screening

This is the bounded complete-information optimizer-screening experiment used to characterize optimizer quality/runtime before communication effects are introduced.

## Research question

Before communication loss and multi-view Voting are involved, how do different optimizer families behave as simultaneous task count increases for the same 100-robot capacity-one assignment problem?

The screening itself remains complete-information only. The MILP and ACO solver owners defined here are also reused by the canonical lossy P2P Voting experiment and therefore now have an explicit missing-edge contract.

## Controlled screening settings

- Robots: `100`
- Simultaneous task counts: `5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100`
- Trials per task-count/method point: `100`
- Robot simultaneous capacity: one task
- Cost model: normalized 2-D Euclidean spatial cost plus the canonical positive floor
- Packet loss: disabled in this screening
- Voting / proposal consensus: disabled in this screening
- Route planning, execution noise, retransmission, permanent failure, deadlines, Robust Optimization, and multi-objective costs are excluded.

## Shared owners

This experiment imports the canonical problem owners from `run_multitask_peer_cost_experiment.py`:

- `generate_spatial_cost_matrix`
- `solve_hungarian_assignment`
- `solve_sequential_greedy`
- `assignment_total_cost`
- `validate_experiment_config`

The optimizer-family owner is `run_multitask_optimizer_screening.py`.

## Compared methods

### Hungarian

The exact linear assignment reference under:

```text
one robot per task
one simultaneous task per robot
```

Hungarian defines `0%` optimality gap for the complete-information screening.

### MILP

Binary formulation:

```text
x_ij in {0,1}
sum_i x_ij = 1   for every task j
sum_j x_ij <= 1  for every robot i
minimize sum_ij C_ij x_ij
```

Owner:

```text
solve_milp_assignment
```

Solver:

```text
scipy.optimize.milp / HiGHS
```

The owner keeps:

```text
mip_rel_gap = 0.0
```

and the existing numerical objective equality tolerance:

```text
MILP_NUMERICAL_TOLERANCE_PERCENT = 1e-6
```

The complete-information screening still passes only finite cost matrices, so the formal screening behavior is unchanged.

### ACO + Local Search

The method remains explicitly labeled `ACO + Local Search` rather than pure ACO.

Fixed canonical controls:

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

ACO construction uses:

```text
pheromone^alpha * (1 / cost)^beta
```

and a bounded cheapest-robot candidate list.

`improve_aco_assignment_locally` remains the separate local-refinement owner and applies:

1. unused-robot replacements that reduce cost;
2. pairwise assigned-robot swaps that reduce cost.

The complete-information screening behavior and fixed search budget are unchanged.

### Greedy Baseline

The existing sequential Greedy assignment remains a non-optimal heuristic baseline.

## Reusable missing-edge contract

The canonical lossy P2P Voting experiment represents missing receiver-local cost entries as:

```text
+inf
```

The existing MILP and ACO owners now accept matrices containing finite costs plus `+inf` unavailable edges when reused by that experiment.

This is a solver-capability extension only; the complete-information screening still supplies fully finite matrices.

### MILP unavailable edges

`solve_milp_assignment` rejects `NaN` and `-inf`, but permits `+inf`.

For every `+inf` edge, the corresponding binary decision variable is disabled directly:

```text
upper_bound(x_ij) = 0
```

The objective coefficient stored for that disabled variable is irrelevant because the variable cannot become one.

This deliberately avoids a large artificial penalty / Big-M objective approximation.

If any task has no finite candidate, or HiGHS reports no feasible complete assignment, the owner returns `None`.

A returned assignment is also checked to ensure that every selected edge was finite in the supplied matrix.

### ACO unavailable edges

`solve_aco_assignment` likewise rejects `NaN` and `-inf`, while allowing `+inf` unavailable edges.

`select_aco_candidates` has always filtered candidates through `np.isfinite`, so unavailable edges cannot be selected during ant construction.

The previous complete-information path initialized ACO from a sequential Greedy seed. Under incomplete P2P information, that Greedy seed can fail even when another feasible matching still exists. Therefore:

- when Greedy returns a valid seed, ACO uses it exactly as before;
- when Greedy returns `None`, ACO now starts without a best seed and still runs its normal ant construction;
- if no ant ever produces a complete finite-edge assignment, ACO returns `None`.

This change is necessary so ACO feasibility is determined by ACO rather than silently inheriting Greedy's local failure boundary.

## Diagnostics for reusable solver inputs

MILP invalid-data boundary:

```text
owner=run_multitask_optimizer_screening
function=solve_milp_assignment
category=data
code=INVALID_MILP_COST
```

ACO invalid-data boundary:

```text
owner=run_multitask_optimizer_screening
function=solve_aco_assignment
category=data
code=INVALID_ACO_COST
```

`+inf` alone is not an invalid-data error; it means an unavailable edge.

A missing-edge feasibility failure is returned as `None` to the calling experiment, which owns whether that receiver proposal is counted invalid.

## Exact optimizer contract

Before the complete-information screening sweep, `validate_exact_optimizer_contract` checks task counts `1, 5, 50, 100`.

MILP must match Hungarian total cost within:

```text
MILP_NUMERICAL_TOLERANCE_PERCENT = 1e-6%
```

Failure:

```text
owner=run_multitask_optimizer_screening
function=validate_exact_optimizer_contract
category=planning
code=MILP_NOT_HUNGARIAN_EXACT
```

The same exactness check is applied to every formal complete-information MILP trial through:

```text
run_trial / planning / MILP_NOT_EXACT
```

ACO is not required to match Hungarian because it is a stochastic metaheuristic.

## Evaluation metrics

- `average_optimality_gap_percent`
- `optimal_cost_match_percent`
- `near_optimal_5pct_percent`
- `exact_optimal_assignment_percent`
- `average_runtime_ms`
- `median_runtime_ms`

### Optimality gap

```text
gap (%) = 100 * (method_cost - hungarian_cost) / hungarian_cost
```

Lower is better.

### Optimal-cost match

MILP uses its solver-specific numerical tolerance. Hungarian, ACO, and Greedy retain the generic assignment equality tolerance.

### Near-optimal within 5%

```text
gap <= 5%
```

### Runtime

Only each solver call is timed with `time.perf_counter`; cost generation and reporting are outside that timer.

## Paired complete-information trial contract

For every task-count/trial pair, all screening methods receive the same true cost matrix.

```text
trial_seed = seed + task_count * 100003 + trial * 1009
```

ACO receives its own deterministic search stream:

```text
trial_seed + 7000003
```

This deterministic ACO stream remains unchanged for the complete-information screening.

## Run

Canonical complete-information screening:

```bash
python run_multitask_optimizer_screening.py
```

Smoke:

```bash
python run_multitask_optimizer_screening.py --tasks 5 20 100 --trials 3
```

Changing ACO controls creates a sensitivity run and must not be merged into canonical data.

## Outputs

```text
results/multitask_optimizer_screening/data/
results/multitask_optimizer_screening/figures/
```

Primary CSVs:

```text
optimizer_screening_raw.csv
optimizer_screening_summary.csv
report_average_optimality_gap_percent.csv
report_optimal_cost_match_percent.csv
report_near_optimal_5pct_percent.csv
report_average_runtime_ms.csv
```

## MILP numerical regression case

The first full user run originally stopped at `tasks=100, trial=7` because the generic `1e-8%` objective tolerance was tighter than the HiGHS floating-point near-tie observed there.

Focused reproduction showed approximately:

- Hungarian total cost: `15.151402216370842`
- MILP total cost: `15.15140229586623`
- absolute cost difference: `7.95e-8`
- percentage gap: `5.2467347189941e-7%`

The MILP assignment was integral and feasible; only two tasks exchanged two nearly tied robots.

The existing `1e-6%` MILP numerical tolerance remains unchanged by the missing-edge capability extension.

## Scope boundary

The complete-information screening is supporting optimizer characterization only; it must not be used by itself to claim communication robustness or Voting benefit.

The report-facing lossy optimizer-family comparison is now owned by:

```text
run_multitask_peer_cost_all_optimizers.py
```

Robust Optimization remains excluded until a justified uncertainty set is defined.
