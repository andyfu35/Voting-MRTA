# Route Heuristic Optimization

This experiment keeps the shared base cost, complete voting preference, and 20% route guardrail fixed, then isolates two routing decisions:

1. which task should be assigned first;
2. where a newly assigned task should be inserted into a robot's current route.

## Compared methods

### Append + Hardness Order

Current guardrail baseline. Tasks with worse minimum base cost are processed first. A selected task can only be appended to the end of a robot's current route.

### Append + Regret Order

Tasks are ordered by the difference between the best and second-best active robot costs:

```text
regret_j = second_best_cost_j - best_cost_j
```

Large-regret tasks are assigned first because they have fewer good substitute robots.

### Best Insertion + Hardness Order

Uses the original hardness task order, but tries every insertion point in every candidate robot route and keeps the insertion that produces the smallest planned route duration.

### Best Insertion + Regret Order

Combines both changes.

## Fixed guardrail

For every task, each robot first reports the best projected route duration that would result from accepting that task. Only robots satisfying

```text
projected_duration_i <= 1.20 * best_projected_duration
```

remain eligible. Complete voting preference then selects among those route-feasible candidates.

## First validated results

Averaged across robot counts 10, 20, 40, 60 and task loads 50%, 100%, 150%:

```text
Append + Hardness (current)
  task completion      99.3365%
  mission success      80.8333%
  actual makespan      116.114 s
  energy/completed     20.288
  travel distance      532.172 m
  task-count CV        0.589

Append + Regret
  task completion      99.3896%
  mission success      82.7083%
  actual makespan      112.977 s
  energy/completed     18.399
  travel distance      471.847 m
  task-count CV        0.600

Best Insertion + Hardness
  task completion      99.3319%
  mission success      80.2083%
  actual makespan      113.571 s
  energy/completed     19.035
  travel distance      488.244 m
  task-count CV        0.600

Best Insertion + Regret
  task completion      99.4054%
  mission success      83.1250%
  actual makespan      111.292 s
  energy/completed     17.590
  travel distance      441.913 m
  task-count CV        0.606
```

Relative to the current Append + Hardness baseline, Best Insertion + Regret reduces average makespan by about 4.2%, energy per completed task by about 13.3%, and route travel distance by about 17.0%, while maintaining approximately 99.4% task completion and zero duplicate execution in the experiment.

The task-count CV is slightly higher, which shows that equal task counts are not the same as equal route workload. A later refinement should add route-finish-time CV as the primary load-balance metric.

## Run

```bash
python3 run_route_heuristic_optimization.py
```

Outputs are under:

```text
results/route_heuristic/data/
results/route_heuristic/figures/
```
