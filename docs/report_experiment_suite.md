# Canonical Report Experiment Suite

This file defines which experiments must be rerun before report writing and which outputs are report-authoritative.

## Global report rules

- Use seed `20260903` unless a canonical experiment states otherwise.
- Use `100` trials per reported canonical configuration unless that contract is explicitly revised.
- Smoke/trend runs using fewer trials or a sampled voter cap are preview data only and must not be mixed into formal report tables.
- Within one comparison, all enabled algorithms must use paired scenarios: the same robot/task realization, voter identities, and packet-loss realization.
- Robust Optimization and multi-objective success/time/energy cost models remain excluded from this report cycle.
- Scalar task cost remains the existing spatial cost.
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

Purpose: measure optimal-robot execution success as robot count and directed peer-cost packet loss vary for one task.

Authoritative output root:

```text
results/peer_cost_majority/
```

The current 100-trial Experiment 1 dataset is already complete and can be retained for the one-page paper.

## Experiment 2 - Matched-scale lossy P2P Voting through 1000 tasks

Owner:

```text
run_multitask_peer_cost_all_optimizers.py
```

Matched-scale contract:

```text
robots = tasks
30% independent directed scalar task-cost packet loss
one simultaneous task per robot
```

Default report methods:

```text
Hungarian Oracle
Voting Greedy
Voting Hungarian
Voting Auction
```

Optional runtime probe:

```text
Voting MILP
```

MILP is enabled only with:

```bash
--include-milp
```

Its optimizer implementation remains owned by `run_multitask_optimizer_screening.py::solve_milp_assignment`; the scaling owner does not duplicate the MILP model.

`Voting ACO + Local Search` remains excluded from the large lossy scaling sweep until the faster MILP probe establishes an acceptable runtime budget.

### Primary report metric

The one-page paper should primarily plot direct cost error relative to the full-information minimum-cost reference:

```text
Cost error (%) = 100 * (method_cost - oracle_cost) / oracle_cost
```

Lower is better.

The code/CSV field is:

```text
average_optimality_gap_percent
```

The `<=5%` rate remains a supporting metric only.

Exactly identical plotted method series are merged into one legend entry; numerically different series remain separate.

### Memory/scaling contract

Receiver-local views are streamed in bounded batches rather than materializing the full receiver x sender x task tensor.

Default voter batch size:

```text
8
```

Changing batch size changes memory/runtime only, not experiment semantics for a fixed configuration.

Authoritative output root:

```text
results/multitask_peer_cost_scaling/
```

### Dense trend preview

The user currently prefers a denser 50-step curve. Preview command:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --tasks 50 100 150 200 250 300 350 400 450 500 \
          550 600 650 700 750 800 850 900 950 1000 \
  --trials 10 \
  --max-voters 100
```

This is preview only because it uses 10 trials and caps larger systems at 100 voting receivers.

### Faster additional-optimizer probe

Before attempting a dense additional-method sweep, test MILP first:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --tasks 50 100 150 \
  --trials 2 \
  --max-voters 20 \
  --include-milp
```

If runtime is acceptable, increase the voter cap and task range gradually. Do not start an all-voter 1000-task MILP sweep before measuring the smaller probe on the actual machine.

### Formal run

The default formal command remains:

```bash
python run_multitask_peer_cost_all_optimizers.py
```

This uses the default scale points, all robots as voters, 100 trials, and the default Greedy/Hungarian/Auction method set. Enabling MILP changes the compared method set and must be explicitly recorded with the resulting data.

## Experiment 3 - Direct vs Voting ablation

Owner:

```text
run_multitask_voting_ablation.py
```

This experiment remains available as supporting causal evidence, but the user has explicitly decided not to include the ablation in the current one-page CACS paper. It is not required for the current one-page rerun cycle.

## Report data acceptance checklist

Before any dataset is called formal report data, confirm:

1. Experiment 1 uses the completed canonical 100-trial dataset;
2. Experiment 2 records `robots`, `tasks`, and `voters` in raw/summary output;
3. Experiment 2 uses `robots == tasks`;
4. formal Experiment 2 uses all robots as voters unless the canonical specification is explicitly revised;
5. preview runs with `--max-voters` are labeled preview;
6. the zero-loss gate reports Hungarian/Auction consistency with the Oracle;
7. when MILP is enabled, the bounded MILP zero-loss integration gate passes under its numerical tolerance;
8. all final report CSVs were generated by the final code version;
9. the main Experiment 2 figure uses direct cost error from the minimum, not the `<=5%` threshold rate;
10. calculations use raw CSV values even when presentation values are rounded.

## Current one-page report structure

The CACS one-page paper should tell two main stories:

1. **Single-task communication robustness** - how packet loss and fleet size affect majority execution success.
2. **Large-scale multi-task Voting** - how percentage cost error relative to the minimum changes as the matched robot/task system scales toward 1000, with additional optimizer curves added only after successful measured runs.

The Direct-vs-Voting ablation and complete-information optimizer screening remain supporting material rather than main one-page figures.
