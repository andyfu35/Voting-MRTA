# Canonical Report Experiment Suite

This file defines which experiments must be rerun before report writing and which outputs are report-authoritative.

## Global report rules

- Use seed `20260903` unless a canonical experiment states otherwise.
- Use `100` trials per reported configuration.
- Smoke tests using `3`, `10`, or `20` trials are validation only and must not be mixed into report tables.
- Within a comparison table, all compared algorithms must use paired scenarios: the same robot/task cost realization and, when communication is modeled, the same packet-loss realization.
- Robust and multi-objective cost models are excluded from this report cycle.
- The scalar task cost remains the existing spatial cost.

## Experiment 1 - Single-task P2P majority robustness

Owner:

```text
run_peer_cost_majority_experiment.py
```

Canonical run:

```bash
python run_peer_cost_majority_experiment.py
```

Purpose:

Measure optimal-robot execution success as robot count and directed peer-cost packet loss vary for one task.

Authoritative output root:

```text
results/peer_cost_majority/
```

Primary report result:

```text
packet-loss x robot-count optimal execution success
```

## Experiment 2 - Complete-information all-optimizer multi-task comparison

Owner:

```text
run_multitask_all_optimizer_experiment.py
```

Canonical run:

```bash
python run_multitask_all_optimizer_experiment.py
```

Compared methods:

```text
Hungarian
Auction
MILP
ACO + Local Search
Greedy Baseline
```

Every `(task_count, trial)` scenario is generated once and shared by all five methods.

Authoritative output root:

```text
results/multitask_all_optimizer/
```

Primary report tables:

```text
average optimality gap
optimal-cost match
near-optimal within 5%
average optimizer runtime
```

This experiment replaces the older four-method optimizer-screening table as the report-facing optimizer-family table. The older screening results remain valid historical development data but are not the final report table because Auction was absent from that comparison.

## Experiment 3 - Lossy P2P multi-task optimizer comparison

Owner:

```text
run_multitask_peer_cost_experiment.py
```

Canonical run:

```bash
python run_multitask_peer_cost_experiment.py
```

Controlled condition:

```text
100 robots
30% independent directed scalar task-cost packet loss
100 paired trials per task count
5..100 simultaneous tasks
```

Compared report methods currently remain:

```text
P2P Greedy
P2P Hungarian
P2P Auction
```

All three methods receive the same cost matrix, packet-loss visibility tensor, task order, and tie-priority realization within each trial.

Authoritative output root:

```text
results/multitask_peer_cost/
```

Primary report tables:

```text
average optimality gap
optimal-cost match
near-optimal within 5%
valid local proposal rate
```

## Experiment 4 - Direct vs Voting ablation

Owner:

```text
run_multitask_voting_ablation.py
```

Canonical run:

```bash
python run_multitask_voting_ablation.py
```

Purpose:

Isolate the benefit of proposal aggregation by comparing Direct and Voting using the exact same local proposal batch.

Authoritative output root:

```text
results/multitask_voting_ablation/
```

Primary report tables:

```text
optimal-cost match
average optimality gap
near-optimal within 5%
valid final assignment
Voting uplift
```

## Required rerun order

Run in this order so failures are easier to diagnose:

```bash
python run_peer_cost_majority_experiment.py
python run_multitask_all_optimizer_experiment.py
python run_multitask_peer_cost_experiment.py
python run_multitask_voting_ablation.py
```

If one command fails, stop the report rerun at the first failing owner/function/category/code and fix that boundary before continuing. Do not combine partial pre-fix and post-fix datasets.

## Report data acceptance checklist

Before report writing begins, confirm:

1. every canonical command completed without exception;
2. every formal point has `100` trials;
3. the all-optimizer exact gate reports Auction/MILP consistency with Hungarian;
4. the lossy P2P zero-loss gate reports Hungarian/Auction consistency;
5. the Direct-vs-Voting zero-loss ablation gate passes;
6. all result CSVs were regenerated after the final code version;
7. smoke-test CSVs are not used in report figures/tables;
8. tables use readable rounded display values while raw CSV values remain unrounded.

## Report structure after rerun

The report should tell four separate controlled stories:

1. single-task communication robustness;
2. optimizer-family behavior as task load increases;
3. multi-task optimizer behavior under 30% P2P cost-message loss;
4. causal Direct-vs-Voting ablation showing the contribution of proposal consensus.

These questions should remain separated in the report so optimizer effects and communication/Voting effects are not conflated.
