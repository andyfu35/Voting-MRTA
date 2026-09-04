# Canonical Report Experiment Suite

This file defines which experiments must be rerun before report writing and which outputs are report-authoritative.

## Global report rules

- Use seed `20260903` unless a canonical experiment states otherwise.
- Use `100` trials per reported configuration.
- Smoke tests using `3`, `10`, or `20` trials are validation only and must not be mixed into report tables.
- Within a comparison table, all compared algorithms must use paired scenarios: the same robot/task realization and, when communication is modeled, the same packet-loss realization.
- Robust Optimization and multi-objective success/time/energy cost models are excluded from this report cycle.
- The scalar task cost remains the existing spatial cost.
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

Purpose:

Measure optimal-robot execution success as robot count and directed peer-cost packet loss vary for one task.

Authoritative output root:

```text
results/peer_cost_majority/
```

Primary result:

```text
packet-loss x robot-count optimal execution success
```

## Experiment 2 - Multi-task lossy P2P Voting across optimizer families

Owner:

```text
run_multitask_peer_cost_all_optimizers.py
```

Canonical run:

```bash
python run_multitask_peer_cost_all_optimizers.py
```

Controlled condition:

```text
100 robots
30% independent directed scalar task-cost packet loss
100 paired trials per task count
5,10,20,30,40,50,60,70,80,90,100 simultaneous tasks
```

Compared report methods:

```text
Hungarian Oracle
Voting Greedy
Voting Hungarian
Voting Auction
Voting MILP
Voting ACO + Local Search
```

For every trial, all five Voting optimizers share the same:

- true cost matrix;
- packet-loss visibility tensor;
- task order;
- consensus tie-priority realization.

ACO alone has an additional deterministic search stream per receiver; that stream never consumes or alters the shared scenario RNG.

Authoritative output root:

```text
results/multitask_peer_cost_all_optimizers/
```

Primary report tables:

```text
average optimality gap
optimal-cost match
near-optimal within 5%
exact optimal assignment
valid local proposal rate
```

The preceding three-method owner:

```text
run_multitask_peer_cost_experiment.py
```

remains a regression baseline. The new experiment deliberately preserves its scenario RNG schedule, so Voting Greedy/Hungarian/Auction should reproduce the previous canonical columns under the same settings.

The complete-information scripts:

```text
run_multitask_optimizer_screening.py
run_multitask_all_optimizer_experiment.py
```

are supporting optimizer-characterization experiments only. They are no longer required as a main report experiment and do not need to be rerun before the Voting report is written.

## Experiment 3 - Direct vs Voting ablation

Owner:

```text
run_multitask_voting_ablation.py
```

Canonical run:

```bash
python run_multitask_voting_ablation.py
```

Purpose:

Isolate the contribution of proposal aggregation by comparing Direct and Voting using the exact same local proposal batch.

Current compared optimizer families remain:

```text
Hungarian
Auction
```

This experiment supports the causal statement that Voting improves these optimizers relative to one fixed incomplete-information receiver. It does not yet establish Direct-vs-Voting uplift for MILP or ACO.

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

The single-task experiment has already completed on the current report cycle. After the current code update, the required reruns are:

```bash
python run_multitask_peer_cost_all_optimizers.py
python run_multitask_voting_ablation.py
```

Before the formal Experiment 2 run, validate the new integration with:

```bash
python run_multitask_peer_cost_all_optimizers.py --tasks 5 20 100 --trials 3
```

If the smoke run fails, stop at the first owner/function/category/code boundary. Do not proceed to the 100-trial formal sweep until that boundary is fixed.

## Report data acceptance checklist

Before report writing begins, confirm:

1. Experiment 1's existing canonical 100-trial dataset completed without exception;
2. Experiment 2's new five-optimizer command completes all task counts without exception;
3. Experiment 2 contains exactly 100 trials per task-count/method point;
4. Experiment 2's zero-loss gate reports Hungarian/Auction/MILP consistency with the Oracle;
5. the old Greedy/Hungarian/Auction columns reproduce the preceding three-method canonical results within their existing numerical contracts;
6. Experiment 3's Direct-vs-Voting zero-loss ablation gate passes;
7. all report CSVs were generated by the final code version;
8. smoke-test CSVs are not used in report figures/tables;
9. report tables may round display values, while calculations use raw CSV values.

## Report structure after rerun

The report should tell three controlled stories:

1. **Single-task communication robustness** - how packet loss and fleet size affect majority execution success.
2. **Multi-task optimizer-family behavior inside lossy Voting** - how Greedy, Hungarian, Auction, MILP, and ACO + Local Search behave under the same 30% P2P information loss as task load increases.
3. **Direct-vs-Voting ablation** - whether proposal aggregation itself improves assignment quality relative to one fixed incomplete-information receiver.

The complete-information optimizer screening can be cited as supporting characterization or placed in an appendix, but it is not a main report experiment.
