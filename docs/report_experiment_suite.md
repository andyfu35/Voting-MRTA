# Canonical Report Experiment Suite

This file defines which experiments must be rerun before report writing and which outputs are report-authoritative.

## Global report rules

- Use seed `20260903` unless a canonical experiment states otherwise.
- Experiment 1 remains 100 trials. Experiment 2 uses its explicit 20-trial time-budget contract.
- Smoke/trend runs using fewer trials or a sampled voter cap are preview data only and must not be mixed into formal report tables.
- Within one comparison, all methods use paired physical scenario, voter identities, packet-loss realization, task order, tie priority, and capacity.
- Robust Optimization and multi-objective success/time/energy costs remain excluded from this report cycle.
- Scalar task cost remains `0.05 + EuclideanDistance`.
- Partial output from a run that terminates before `save_outputs` is not report data.

## Experiment 1 - Single-task P2P majority robustness

Owner:

```text
run_peer_cost_majority_experiment.py
```

Canonical run:

```bash
python run_peer_cost_majority_experiment.py
```

The completed 100-trial Experiment 1 dataset remains report-authoritative.

## Experiment 2 - Fixed 100 robots, diverse fast Voting optimizers

Canonical owner:

```text
run_multitask_peer_cost_all_optimizers.py
```

Canonical workload:

```text
physical robots = 100 fixed
task batches = 100, 200, ..., 1000
capacity_per_robot = ceil(tasks / 100)
30% independent directed scalar task-cost packet loss
20 trials per task point
all 100 physical robots vote
```

Task counts above 100 are allocation batches, not claims of simultaneous physical execution.

### Canonical methods

```text
Hungarian Oracle                full-information minimum-cost reference
Voting Greedy                   sequential capacitated Greedy baseline
Voting Hungarian                exact receiver-local assignment
Voting Min-Cost Flow            capacity-native network optimization
Voting Sinkhorn + Rounding      entropic transport approximation + explicit rounding
```

The Oracle remains in CSV output but is not plotted as a redundant zero-error curve. The primary Experiment 2 figure therefore normally has four optimizer curves.

Global Greedy and Static Regret-2 Greedy remain implemented for supporting studies but are no longer canonical report curves. Auction, MILP, and ACO also remain implemented outside this canonical lossy workload sweep.

### Optimizer ownership

Greedy:

```text
run_multitask_workload_heuristics.py::solve_sequential_greedy_capacitated
```

Hungarian:

```text
run_multitask_peer_cost_experiment.py::solve_hungarian_assignment
```

Min-Cost Flow and Sinkhorn:

```text
run_multitask_workload_optimizers.py
```

Important new boundaries:

```text
solve_min_cost_flow_capacitated
quantize_min_cost_flow_costs
compute_sinkhorn_transport_plan
round_sinkhorn_plan_to_capacity
solve_sinkhorn_capacitated
```

### Min-Cost Flow contract

The receiver-local network is:

```text
source -> robots -> tasks -> sink
```

Robot capacity is native to the network and does not use slot duplication.

OR-Tools `SimpleMinCostFlow` requires integer arc costs, so float costs are multiplied by `1_000_000` and rounded. Evaluation still uses the original float costs. Complete-information preflight must match the float Hungarian Oracle within `0.01%`.

The dependency is declared in `requirements.txt` as `ortools>=9.10,<10`.

### Sinkhorn contract

Canonical configuration:

```text
epsilon = 0.08
max_iterations = 30
tolerance = 1e-5
```

Sinkhorn produces a soft transport plan on the physical receiver-local cost matrix. A separate named rounding function converts that plan into a discrete capacity-feasible proposal. Sinkhorn is not described as an exact discrete solver.

### Primary report metric

```text
Cost error (%) = 100 * (method_cost - oracle_cost) / oracle_cost
```

Lower is better. The CSV field is:

```text
average_optimality_gap_percent
```

The `<=5%` rate remains supporting only.

Exactly identical measured curves may share one legend entry; near-overlapping curves remain separate.

### Canonical x-axis

```text
Task batch size (100 robots fixed)
```

Points:

```text
100 200 300 400 500 600 700 800 900 1000
```

### Pairing and communication contract

All four Voting methods share the same:

- physical `100 x T` true cost matrix;
- physical voter identities;
- directed packet-loss realization;
- task order;
- tie-priority matrix;
- capacity value.

Missing receiver-local edges remain `+inf`. No optimizer regenerates communication loss.

### Runtime contract

Independent `(task_count, trial)` jobs use `ProcessPoolExecutor`.

Canonical default:

```text
workers = min(4, available CPU count)
receiver batch size = 4
```

Worker count and receiver batch size are runtime-only controls and must not change deterministic experiment results.

### Zero-loss gates

Before the lossy sweep:

- Greedy must return a valid capacity-feasible proposal;
- Voting Hungarian must match the capacitated Hungarian Oracle;
- Min-Cost Flow must match the float Oracle within its explicit quantization tolerance;
- Sinkhorn + Rounding must return a valid capacity-feasible proposal.

These checks are bounded to at most 200 tasks per family.

### Dependencies and real-machine run sequence

After pulling:

```bash
git pull
pip install -r requirements.txt
```

Small smoke:

```bash
time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 100 500 1000 \
  --trials 1 \
  --max-voters 5 \
  --workers 1
```

All-voter timing preview:

```bash
time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 100 500 1000 \
  --trials 2 \
  --workers 4
```

Canonical Experiment 2:

```bash
time python run_multitask_peer_cost_all_optimizers.py
```

No-argument meaning:

```text
100 robots
100..1000 tasks by 100
20 trials per point
100 voters
30% packet loss
Voting Greedy
Voting Hungarian
Voting Min-Cost Flow
Voting Sinkhorn + Rounding
Hungarian Oracle reference
up to 4 workers
```

The one-hour runtime is a target to be verified on the user's Mac, not a guaranteed property.

### Authoritative outputs

```text
results/multitask_peer_cost_fixed100_workload/data/workload_comparison_raw.csv
results/multitask_peer_cost_fixed100_workload/data/workload_comparison_summary.csv
results/multitask_peer_cost_fixed100_workload/figures/average_optimality_gap_percent.png
```

## Experiment 3 - Direct vs Voting ablation

Owner:

```text
run_multitask_voting_ablation.py
```

This remains supporting causal evidence and is not required for the current one-page CACS paper.

## Report data acceptance checklist

Before calling a dataset formal report data, confirm:

1. Experiment 1 uses its completed canonical 100-trial dataset;
2. Experiment 2 uses exactly 100 physical robots at every load;
3. task batches are `100..1000` in steps of `100`;
4. Experiment 2 has 20 completed trials per task point unless a later canonical revision changes it;
5. all 100 robots vote in formal Experiment 2;
6. `capacity_per_robot = ceil(tasks/100)` is enforced by all methods and final consensus;
7. the four Voting methods share paired scenario/communication inputs;
8. Min-Cost Flow reports use original float assignment costs even though its internal network costs are integer-scaled;
9. Sinkhorn is described as an approximation with explicit discrete rounding;
10. all zero-loss integration gates pass;
11. all final CSVs come from the final code version;
12. the main figure uses direct cost error from the minimum;
13. task counts above 100 are described as workload batches;
14. the final paper cites method-appropriate references for Hungarian, Min-Cost Flow, and Sinkhorn.

## Current one-page report structure

The paper should tell two main stories:

1. **Single-task communication robustness** - how packet loss and fleet size affect majority execution success.
2. **Fixed-fleet workload scaling** - how Voting assignment cost error changes from 100 to 1000 tasks with 100 robots using four computationally distinct local optimization methods.
