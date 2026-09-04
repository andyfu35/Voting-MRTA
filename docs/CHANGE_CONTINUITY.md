# Change Continuity

Historical continuity through `2026-09-04 - Add Optional MILP Probe to Large-Scale Experiment 2` is preserved exactly in Git at:

```text
commit: 416a7dc5a64316f24803eb5fd8f7a767ca72d3ff
blob:   c676ebfe4723037f0223f3982d77db5363ecb98f
```

## 2026-09-04 - Add MILP-Only 1000-Task Probe Mode

### Purpose
The user already has Greedy/Hungarian/Auction scaling data and wants to avoid rerunning those methods while measuring the previously unmeasured MILP behavior at the largest `1000 robots / 1000 tasks` matched-scale point.

The preceding `--include-milp` mode always ran the three default Voting optimizers together with MILP. This change adds an explicit MILP-only method-selection mode so the expensive probe can execute only `Voting MILP`, while still computing the full-information Hungarian Oracle once per trial as the minimum-cost evaluation reference required for the cost-error metric.

Before modification, repository-root `AGENTS.md` and `docs/AI_CHANGE_PROTOCOL.md` were checked in order and remain absent. Then `docs/CHANGE_CONTINUITY.md`, canonical `docs/multitask_voting_mrta.md`, the actual Experiment 2 owner `run_multitask_peer_cost_all_optimizers.py`, exact functions `resolve_voting_methods`, `validate_zero_loss_optimizer_contract`, `run_trial`, `run_experiment`, `parse_args`, and `main`, plus the true MILP owner `run_multitask_optimizer_screening.py::solve_milp_assignment`, were read before writing.

### Files
- `run_multitask_peer_cost_all_optimizers.py`
- `docs/multitask_voting_mrta.md`
- `docs/CHANGE_CONTINUITY.md`

### Owner and named functions
Experiment 2 remains owned by:

```text
run_multitask_peer_cost_all_optimizers.py
```

Updated named boundaries:

- active optimizer selection: `resolve_voting_methods`
- enabled-method integration gate: `validate_zero_loss_optimizer_contract`
- sweep selection/wiring: `run_experiment`
- CLI contract: `parse_args`
- CLI-to-owner wiring: `main`

The MILP optimizer itself remains owned by:

```text
run_multitask_optimizer_screening.py::solve_milp_assignment
```

No MILP formulation, solver state machine, unavailable-edge handling, or fallback was copied.

### Responsibility movement
No optimization responsibility moved.

`resolve_voting_methods` now owns the distinction among three execution modes:

```text
default       -> Greedy + Hungarian + Auction
--include-milp -> Greedy + Hungarian + Auction + MILP
--only-milp    -> MILP only
```

The Oracle remains outside that Voting method set and is still calculated in `run_trial` solely as the evaluation reference.

`validate_zero_loss_optimizer_contract` now checks only optimizer integrations that are actually enabled. This prevents a MILP-only probe from spending time on Greedy/Hungarian/Auction preflight solves while preserving the existing gates in default and include-MILP modes.

### Preserved behavior
The following remains unchanged when `--only-milp` is not supplied:

- default methods remain Voting Greedy, Voting Hungarian, and Voting Auction;
- `--include-milp` still adds Voting MILP to those methods;
- `RANDOM_SEED = 20260903`;
- 30% independent directed P2P scalar cost-message loss;
- matched scale `robot_count == task_count`;
- one simultaneous task per robot;
- scalar cost `0.05 + EuclideanDistance`;
- trial/scenario/voter/visibility seed formulas;
- receiver-local missing edges remain `+inf`;
- voter batching and support accumulation remain unchanged;
- support-maximizing consensus remains unchanged;
- full-information Hungarian Oracle remains the cost reference;
- report metric remains direct percentage cost error from the minimum;
- existing supporting metrics remain recorded;
- Experiment 1 and Experiment 3 remain unchanged.

A MILP-only run uses the same scenario/voter/visibility streams as a preceding run with the same task count, trial numbers, seed, packet-loss rate, and voter cap, so its result can be added to the existing paired preview without regenerating method-specific communication data.

### Deliberately changed behavior
New CLI flag:

```text
--only-milp
```

In this mode:

- `Voting MILP` is the only Voting optimizer executed;
- Greedy, Voting Hungarian, and Voting Auction are not executed;
- Hungarian Oracle is still solved once per trial because `Cost error (%)` requires the minimum full-information cost;
- the bounded MILP zero-loss integration gate remains active up to 50 tasks;
- Greedy/Hungarian/Auction zero-loss gates are skipped because those methods are not enabled.

`--only-milp` and `--include-milp` are intentionally mutually exclusive rather than silently choosing one behavior.

### Diagnostic contract
Conflicting method-selection flags fail at the first method-selection owner:

```text
owner=run_multitask_peer_cost_all_optimizers
function=resolve_voting_methods
category=contract
code=CONFLICTING_METHOD_FLAGS
```

MILP solver/model failures continue to propagate from the true owner, including:

```text
owner=run_multitask_optimizer_screening
function=solve_milp_assignment
category=data
code=INVALID_MILP_COST
```

Existing Experiment 2 state/planning/runtime boundaries are unchanged.

### Validation performed
- The updated owner was re-fetched after the GitHub write and inspected at `resolve_voting_methods` and `validate_zero_loss_optimizer_contract`.
- The canonical Experiment 2 document was synchronized with `--only-milp`, same-seed comparability, bounded MILP preflight, and the recommended 1000-task timing commands.
- No GitHub CI status checks are configured for this repository change.
- A real SciPy/HiGHS 1000-task MILP runtime smoke has not yet been executed; the user's target machine is the required runtime-validation boundary.

### Known limitations / unfinished risks
- A single receiver-local `1000 x 1000` MILP contains approximately one million binary assignment variables before solver presolve. Runtime and memory may therefore be large even though the other Voting optimizers are skipped.
- `--only-milp` removes redundant optimizer work but does not make MILP intrinsically cheap.
- A `10 trials x 100 voters` 1000-task preview requires 1000 receiver-local MILP solves and may take a long time.
- `--voter-batch-size 1` is recommended for the 1000-task MILP probe to reduce peak receiver-view memory; it does not change the selected voters or solver result.
- The output files still use the shared `results/multitask_peer_cost_scaling/` root, so a probe can overwrite a preceding preview unless that directory is backed up first.
- `--max-voters` remains preview-only and must not be described as all-voter canonical scaling.

### Next step
Preserve the existing scaling data if it is still needed locally:

```bash
cp -R \
  results/multitask_peer_cost_scaling \
  results/multitask_peer_cost_scaling_before_milp_1000
```

Pull the new method-selection mode:

```bash
git pull
```

First measure the actual cost of one 1000-task trial with five selected voters:

```bash
time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 1000 \
  --trials 1 \
  --max-voters 5 \
  --voter-batch-size 1 \
  --only-milp
```

If this is acceptable, run the same 100-voter/10-trial preview condition used for the existing scaling data:

```bash
time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 1000 \
  --trials 10 \
  --max-voters 100 \
  --voter-batch-size 1 \
  --only-milp
```

Use `scaling_comparison_summary.csv` or `report_average_optimality_gap_percent.csv` to add the MILP cost-error point to the existing figure.

### Commit SHA
- `ecbd3f7bfb3a9c310739087bd448c23569279ec2` - added `--only-milp`, enabled-method-only preflight checks, and CLI/sweep wiring.
- `f89e615c985ca07578199cf06d9c6f41ff424099` - updated canonical Experiment 2 documentation for MILP-only probes.
- continuity update commit: this file's commit.
