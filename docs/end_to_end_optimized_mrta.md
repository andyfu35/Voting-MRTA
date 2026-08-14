# End-to-End Optimized Decentralized Voting-MRTA

This experiment combines the strongest communication and routing components validated in the previous experiments and evaluates them as one execution-level system.

## Fixed assumptions

- one shared pre-allocation base-cost model;
- complete preference distributions rather than single-winner ballots;
- heterogeneous strong voting policies: Inverse a=2, Inverse a=3, Softmax b=2;
- 30% packet loss per transmission attempt;
- three stop-on-success retransmission attempts;
- 5% permanent robot failure sampled before the allocation round;
- 67%+ plan quorum;
- one-shot allocation of all tasks;
- task loads of 50%, 100%, and 150% of the active robot count;
- robot counts 10, 20, 40, 60, 80, and 100.

The base cost remains independent of future route length and future task count. Route workload is handled only after voting information is available.

## Compared methods

### Centralized Route Greedy

The existing centralized route-aware heuristic. It is a baseline, not an exact global optimum.

### Full-Info Optimized Reference

Uses all complete preferences without packet loss and applies the optimized route heuristic:

- regret-first task ordering;
- best insertion into the current route;
- 20% route guardrail.

This isolates the best currently implemented decision logic from communication effects.

### Flat P2P + Current Route

Every active robot exchanges its complete preference directly with every other active robot. Each local decision maker uses the current append + hardness-order route guardrail.

### 5-ary + Current Route

Complete preference matrices are summed hierarchically in groups of five. Up to five top leaders remain decentralized and form the plan quorum. The current append + hardness route heuristic is retained.

### 5-ary Backup + Current Route

Adds backup relay to failed hierarchical preference links while retaining the current route heuristic.

### 5-ary Backup + Optimized Route

The proposed integrated method:

```text
shared base cost
  -> complete preference voting
  -> 5-ary hierarchical aggregation
  -> backup relay
  -> regret-first task ordering
  -> best route insertion
  -> 20% route guardrail
  -> 67%+ plan quorum
  -> quorum-certificate dissemination
  -> task execution
```

## Quorum certificate and execution authorization

The end-to-end simulation no longer assumes that a safely committed plan automatically reaches every robot.

Decision makers exchange compact plan proposals. A receiver that observes at least 67% support for one plan becomes a quorum-certificate witness. The committed assignment is then pushed to active robots with the same three-attempt stop-on-success retransmission model. If more than one witness exists, a second witness can retry a failed QC delivery.

A robot executes its assigned route only after it has the committed plan. This makes duplicate-task and unexecuted-task metrics execution-level outcomes rather than only proposal-level metrics.

## Communication accounting

The reported decentralized communication totals include:

1. complete-preference exchange / hierarchical aggregation;
2. plan-proposal exchange among current decision makers;
3. QC dissemination to active robots.

Metrics include logical messages, actual transmission attempts, bytes transmitted, sequential communication stages, preference coverage, QC delivery rate, and a shared-channel latency proxy.

The latency proxy uses the same synthetic model as the hierarchy experiment and is not wall-clock runtime.

## Execution metrics

The experiment records:

- task completion rate;
- all-task mission success rate;
- unexecuted-task rate;
- duplicate-task execution rate;
- actual makespan;
- energy per completed task;
- total travel distance;
- task-count CV;
- actual route finish-time CV;
- ETA error;
- safe commit rate;
- plan match against the corresponding full-information route heuristic.

Route finish-time CV is added because equal task counts do not imply equal workload when tasks have different travel and service times.

## Run

```bash
python3 run_end_to_end_optimized_experiment.py
```

Outputs are written under:

```text
results/end_to_end/data/
results/end_to_end/figures/
```

The experiment is intentionally the final integration step before preparing the presentation summary. The most useful presentation figures are expected to be task completion, makespan, energy per completed task, route finish-time CV, communication bytes/attempts, latency proxy, duplicate execution, and the makespan-energy trade-off.
