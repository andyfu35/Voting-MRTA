# Change Continuity

Historical continuity through the matched-scale 50-to-1000 Experiment 2 redesign remains preserved in Git at:

```text
commit: 041788ad39d8a8c0423c7560d16d12f475a38fd9
blob:   3b30ec661bd4f4c5843ba949f5ffb790b73ecd70
```

## 2026-09-04 - Add Optional MILP Probe to Large-Scale Experiment 2

### Purpose
The user wants more optimizer families in the Experiment 2 figure, but does not want to immediately return to the prohibitively slow five-optimizer sweep. The requested next step is to try one of the previously slow methods that is still substantially faster than ACO + Local Search.

This change adds `Voting MILP` back to the matched-scale lossy P2P experiment as an **explicit optional probe** while preserving the fast default Greedy/Hungarian/Auction sweep.

The report-facing primary metric is also synchronized with the user's latest decision: plot direct percentage cost error relative to the full-information minimum rather than using the `<=5%` threshold rate as the main figure.

Before modification, repository-root `AGENTS.md` and `docs/AI_CHANGE_PROTOCOL.md` were checked in order and remain absent. Then `docs/CHANGE_CONTINUITY.md`, canonical `docs/multitask_voting_mrta.md`, the actual Experiment 2 owner `run_multitask_peer_cost_all_optimizers.py`, exact routing/trial/report functions, and the true MILP owner `run_multitask_optimizer_screening.py::solve_milp_assignment` were read before writing.

### Files
- `run_multitask_peer_cost_all_optimizers.py`
- `docs/multitask_voting_mrta.md`
- `docs/report_experiment_suite.md`
- `docs/CHANGE_CONTINUITY.md`

### Owner and named functions
Experiment 2 remains owned by:

```text
run_multitask_peer_cost_all_optimizers.py
```

New/updated named boundaries:

- optimizer tolerance selection: `cost_match_tolerance_percent`
- active method selection: `resolve_voting_methods`
- MILP receiver-batch iteration: `solve_milp_batch_proposals`
- optimizer-family routing: `solve_voter_batch_proposals`
- streamed support collection: `collect_voting_support`
- final consensus: `finalize_voting_assignments`
- complete-information integration gate: `validate_zero_loss_optimizer_contract`
- paired scenario execution: `run_trial`
- sweep execution: `run_experiment`
- identical-curve legend merging: `combine_plot_label`, `build_plot_series_groups`
- report figure output: `save_metric_plot`, `save_outputs`

The MILP algorithm itself remains owned by:

```text
run_multitask_optimizer_screening.py::solve_milp_assignment
```

No MILP formulation, state machine, missing-edge policy, or HiGHS implementation was copied into the Experiment 2 owner.

### Responsibility movement
Previously, the large-scale owner knew only Greedy/Hungarian/Auction. MILP is now routed through a narrow owner boundary:

```text
solve_voter_batch_proposals
    -> solve_milp_batch_proposals
    -> run_multitask_optimizer_screening.solve_milp_assignment
```

`solve_milp_batch_proposals` owns only receiver iteration and proposal-validity bookkeeping. The optimization responsibility stays in `solve_milp_assignment`.

Plot-series deduplication is separated into `build_plot_series_groups`; it does not alter raw/summary data and only merges legend/line presentation when curves are numerically identical at every plotted scale point.

### Preserved behavior
The following behavior is unchanged when `--include-milp` is not supplied:

- default methods remain Voting Greedy, Voting Hungarian, and Voting Auction;
- `RANDOM_SEED = 20260903`;
- 30% independent directed scalar task-cost packet loss;
- matched scale `robot_count == task_count`;
- one simultaneous task per robot;
- scalar cost `0.05 + EuclideanDistance`;
- deterministic paired scenario/voter/visibility streams;
- every receiver always knows its own row;
- missing local entries remain `+inf`;
- receiver-local views remain streamed in bounded batches;
- support counting and support-maximizing consensus are unchanged;
- true cost is not used as a hidden consensus tie-break;
- `--max-voters` remains preview-only;
- Experiment 1 and Experiment 3 code are unchanged.

The existing `<=5%`, optimal-cost-match, exact-assignment, and valid-proposal metrics are still recorded in CSV output as supporting metrics.

### Deliberately changed behavior

#### Optional MILP
New CLI flag:

```text
--include-milp
```

When enabled, the exact same cost matrix, voter identities, packet-loss views, task order, and tie priority used by the three default methods are also passed through Voting MILP.

MILP uses the existing owner and accepts finite costs plus `+inf` unavailable edges exactly as before. No repair or Hungarian fallback is introduced.

#### Bounded MILP zero-loss gate
A 1000-task MILP preflight could dominate the runtime before the requested experiment even starts. Therefore the MILP integration gate checks complete-information consistency only up to:

```text
MILP_ZERO_LOSS_CHECK_MAX_SIZE = 50
```

Hungarian/Auction keep their preceding representative-scale zero-loss checks. The bounded MILP gate verifies integration and numerical consistency without pretending to benchmark 1000-task runtime during preflight.

#### Primary figure metric
The one-page Experiment 2 primary figure is now:

```text
Cost error (%) = 100 * (method_cost - oracle_cost) / oracle_cost
```

The existing data field `average_optimality_gap_percent` is retained, but the report y-axis is now `Cost error from minimum (%)`. The `<=5%` plot is no longer generated as a primary figure.

#### Identical plotted curves
If two or more plotted method series are equal at every task point within `1e-12`, they share one line/legend label, e.g.:

```text
Voting Hungarian / Auction
```

Near-overlapping but non-identical series remain separate.

### Diagnostic contract
Experiment-owner routing/configuration failures use the existing format:

```text
owner=run_multitask_peer_cost_all_optimizers
function=<named function>
category=<data|time|state|dependency|planning|safety|runtime|contract>
code=<named code>
```

Representative new routing failures:

```text
owner=run_multitask_peer_cost_all_optimizers
function=solve_milp_batch_proposals
category=contract
code=INVALID_BATCH_COST_SHAPE
```

```text
owner=run_multitask_peer_cost_all_optimizers
function=solve_voter_batch_proposals
category=contract
code=UNKNOWN_METHOD
```

MILP data/model diagnostics raised by the true optimizer owner are deliberately propagated unchanged, including:

```text
owner=run_multitask_optimizer_screening
function=solve_milp_assignment
category=data
code=INVALID_MILP_COST
```

No wrapper replaces that first-failure diagnostic.

### Validation performed
- The updated owner was re-fetched from GitHub after the write and inspected across imports, MILP routing, zero-loss gate, paired trial wiring, CLI wiring, reporting, and final `main()` call.
- The canonical and report-suite documents were synchronized with the optional MILP flag and direct cost-error primary metric.
- No GitHub CI status checks are configured for the code commit.
- A real SciPy/HiGHS runtime smoke has not yet been executed in the user's macOS environment; that is the required next validation boundary.

### Known limitations / unfinished risks
- MILP remains receiver-local. Runtime therefore scales with both selected voter count and problem size; a dense 50-to-1000 sweep may still be too slow even though MILP was much faster than ACO in the preceding complete-information screening.
- The current optional MILP path is serial inside each streamed voter batch. No new parallel MILP runtime layer was added in this bounded change.
- A smoke run writes to the same `results/multitask_peer_cost_scaling/` output root. Back up any preview CSVs you want to preserve before a small probe.
- `--max-voters` data remains a capped-voter preview and cannot be silently described as full-voter scaling.
- ACO + Local Search is still excluded; it should be considered only after measured MILP runtime is known.

### Next step
First preserve the current preview outputs if desired:

```bash
cp -R results/multitask_peer_cost_scaling results/multitask_peer_cost_scaling_before_milp_probe
```

Then pull and run a small timed MILP probe:

```bash
git pull

time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 50 100 150 \
  --trials 2 \
  --max-voters 20 \
  --include-milp
```

If that completes quickly, the next step is a larger 10-trial preview with 50 sampled voters before attempting 100 voters or the full dense 50-to-1000 curve.

### Commit SHA
- `307cc70130083b57c4ac540cfc4d5c205ca20504` - added optional Voting MILP routing, bounded MILP gate, direct cost-error primary plot, and identical-curve legend merging.
- `f8bee67e6b16ff663d15b7d4fb2d88f25a1eaf6c` - updated the canonical Experiment 2 specification.
- `dfeeb7709bec5d3da768b649de5f7a5bf8422572` - synchronized the report experiment suite.
- continuity update commit: this file's commit.
