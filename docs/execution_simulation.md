# One-Shot Route Execution Simulation

This experiment extends Voting-MRTA from allocation quality into execution-level simulation.

The design deliberately separates **pre-allocation cost**, **route construction**, and **post-allocation ETA** so the cost model never pretends to know how many tasks a robot will receive before the assignment is made.

## 1. Pre-allocation cost

Every active robot/task pair uses the same base cost model. The base cost only uses information that exists before allocation:

- direct Robot -> Task travel time
- direct Robot -> Task travel energy
- robot reliability risk

For each task, the three components are normalized across active robots and combined as:

```text
C_ij = 0.05
     + 0.70 * normalized direct travel time
     + 0.20 * normalized direct travel energy
     + 0.10 * normalized reliability risk
```

`C_ij` does **not** include:

- how many tasks robot `i` will eventually receive
- queue length created by the current allocation round
- future task order
- future Task -> Task travel

This avoids the circular dependency:

```text
cost -> allocation -> workload -> cost
```

## 2. Primitive route costs

The simulator separately creates route primitives:

```text
Robot i -> Task j travel time
Robot i -> Task j distance
Task j -> Task k travel time for Robot i
Task j -> Task k distance
Task j service time
```

These primitives are not hidden inside the base voting cost. They are used only after the allocation/routing stage starts constructing a route.

## 3. Complete preference voting

The current strong voting policies are reused:

- Inverse alpha = 2
- Inverse alpha = 3
- Softmax beta = 2

All voters use the same `C_ij`; only the cost-to-preference transformation differs.

Each voter contributes a complete candidate probability distribution rather than one sampled winner.

## 4. One-shot allocation and route sequencing

All tasks in a trial are allocated in one planning round.

The planner processes harder tasks first. A task is considered harder when even its cheapest active candidate has a relatively large base cost.

For each task and each active robot, the planner evaluates the robot's **current route end point**:

```text
if route is empty:
    next travel = Robot start -> Task
else:
    next travel = Previous Task -> Task
```

It then computes the projected finish time if the task is appended to that route.

The preference-aware route method combines:

```text
60% complete voting preference
40% projected route-finish quality
```

Therefore workload is handled by the routing algorithm after voting, not by pretending the original `C_ij` knew the future task count.

The first routing heuristic is intentionally simple: **hardest-first append routing**. It is not claimed to be an optimal vehicle-routing solver.

## 5. Compared methods

### Centralized Route Greedy

Uses no voting preference. For each task, it selects the robot whose current route would have the best projected finish time.

This is a route-aware heuristic baseline, not an exact optimum.

### Full-Info Preference + Route

Aggregates all complete preference distributions and uses the preference-aware route planner with no packet loss.

This isolates the effect of complete preference voting before decentralized communication is introduced.

### Decentralized Preference + Quorum

Each active robot receives only the complete preference messages that survive peer-to-peer packet loss and retransmission, builds its own route plan, broadcasts that proposal, and executes only when it locally observes a `67%+` quorum for a plan.

Different robots may therefore execute routes derived from different local views if conflicting quorums ever occur. This allows the simulator to measure actual duplicate-task risk rather than only assignment-level split brain.

## 6. Execution model

After a route is fixed, ETA is calculated from the route itself:

```text
ETA(task 1) = travel(start -> task 1) + service(task 1)
ETA(task 2) = ETA(task 1) + travel(task 1 -> task 2) + service(task 2)
...
```

The simulator then generates actual execution time using paired multiplicative uncertainty:

- travel-time variation: log-normal noise, sigma = 0.10
- service-time variation: log-normal noise, sigma = 0.15

This creates separate values for:

```text
estimated completion time
actual completion time
ETA error
estimated makespan
actual makespan
```

## 7. Task success

Each robot has a synthetic reliability in:

```text
0.992 to 0.999
```

Each task has a synthetic success factor in:

```text
0.995 to 1.000
```

A robot/task attempt succeeds with:

```text
P(success_ij) = robot_reliability_i * task_success_factor_j
```

The current experiment intentionally does **not** reallocate a task after execution failure. This makes failure recovery the next isolated extension rather than mixing it into the first execution experiment.

## 8. Task load and communication

```text
Robot counts        = 10, 20, 40, 60
Tasks                = 50%, 100%, 150% of active robot count
Trials               = 40 per configuration
Permanent pre-round robot failure = 5%
Packet loss          = 30%, 50%, 70% per attempt
Retransmission       = 3 attempts
Quorum               = 67%+
Workspace            = 100 m x 100 m
```

Task load above 100% is important because robots must receive multiple sequential tasks, which makes Task -> Task transition cost and route ordering unavoidable.

## 9. Metrics

The experiment records:

- Task execution coverage
- Task completion rate
- All-tasks mission success rate
- Attempt success rate
- Duplicate task execution rate
- Unexecuted task rate
- Estimated mission makespan
- Actual mission makespan
- Makespan percentage error
- Mean task ETA absolute error
- Mean task ETA percentage error
- Total travel distance
- Synthetic execution energy
- Route task-count coefficient of variation
- Preference-message delivery rate
- Proposal-message delivery rate
- Modal route-plan share
- Strict all-node route-plan agreement
- Safe quorum commit rate
- Split-brain rate
- Fraction of robots locally authorized to execute

## 10. Run

```bash
python3 run_execution_simulation.py
```

Outputs:

```text
results/execution/data/execution_raw_results.csv
results/execution/data/execution_summary_results.csv
results/execution/data/execution_by_method.csv

results/execution/figures/execution_task_completion_rate.png
results/execution/figures/execution_mission_success_rate.png
results/execution/figures/execution_eta_error.png
results/execution/figures/execution_duplicate_task_rate.png
results/execution/figures/execution_unexecuted_task_rate.png
results/execution/figures/execution_actual_makespan.png
results/execution/figures/execution_energy_consumption.png
results/execution/figures/execution_safe_commit_rate.png
```

## 11. Important current limitations

This is the first execution-level simulator, not the final MRTA model. It still lacks:

1. exact or high-quality VRP / multi-robot routing optimization
2. task reallocation after execution failure
3. robot failure during an already-running route
4. explicit battery depletion / charging constraints
5. deadlines and task priority
6. obstacle-aware path planning and collision avoidance
7. heterogeneous task capabilities / skill requirements
8. dynamic task arrivals after the one-shot allocation round
9. learned or calibrated real-world execution-time uncertainty
10. physical simulator validation such as ROS / Gazebo / Isaac Sim

The purpose of this experiment is first to validate the modeling separation:

```text
pre-allocation base cost
        -> complete preferences
        -> one-shot route allocation and sequencing
        -> ETA
        -> uncertain execution
```

If this separation behaves sensibly, later experiments can replace the simple route heuristic without redesigning the voting layer.
