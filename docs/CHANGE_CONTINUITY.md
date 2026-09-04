# Change Continuity

Historical continuity through the preceding Experiment 2 five-optimizer integration and receiver-parallel runtime work remains preserved in Git. The immediately preceding active continuity state is:

```text
commit: b72fbe0e4c964b9e899cceef5008d7b3ef02777f
blob:   a9f9f5d0b7efccf15b5efe0334763105cfdae06a
```

## 2026-09-04 - Redesign Experiment 2 as a 50-to-1000 Matched-Scale Voting Sweep

### Purpose
The user asked to extend Experiment 2 to `1000` tasks to make the trend easier to see, while removing optimizer families whose receiver-local execution made the previous run prohibitively slow.

The old fixed-100-robot design cannot legally assign 1000 simultaneous tasks under the existing capacity-one contract. This change therefore scales the fleet together with the task set:

```text
robot_count = task_count
```

The new main scale points are:

```text
50, 100, 200, 400, 600, 800, 1000 robots/tasks
```

This preserves one simultaneous task per robot and changes the research question from "load within a fixed 100-robot fleet" to "matched system-scale growth under fixed 30% P2P loss."

MILP and ACO + Local Search are removed from the main lossy scaling experiment. Greedy, Hungarian, and Auction remain. The earlier MILP/ACO implementations and complete-information screening data are preserved as supporting evidence.

Before modification, repository-root `AGENTS.md` and `docs/AI_CHANGE_PROTOCOL.md` were checked in order and remain absent. `docs/CHANGE_CONTINUITY.md`, canonical `docs/multitask_voting_mrta.md`, report suite `docs/report_experiment_suite.md`, and the actual owner `run_multitask_peer_cost_all_optimizers.py` were read before implementation. Exact owner functions inspected included the method list, local-proposal routing, zero-loss contract, `run_trial`, `run_experiment`, reporting, and CLI boundaries.

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

The filename is retained for command compatibility even though the main sweep no longer includes all previously screened optimizer families.

Current named boundaries:

- experiment diagnostics: `fail`
- scale/runtime configuration: `validate_scaling_config`
- preview voter count resolution: `resolve_voter_count`
- voter identity selection: `select_voter_indices`
- directed communication sampling for one batch: `sample_voter_batch_visibility`
- receiver-view materialization: `build_voter_batch_cost_views`
- proposal-support accumulation: `accumulate_proposal_support`
- progress-only output: `report_voter_progress`
- streamed multi-method support collection: `collect_voting_support`
- final support-consensus execution: `finalize_voting_assignments`
- zero-loss one-view consensus path: `solve_zero_loss_consensus`
- exact optimizer gate: `validate_zero_loss_optimizer_contract`
- result evaluation: `evaluate_assignment`
- paired trial owner: `run_trial`
- aggregation: `summarize_results`
- sweep owner: `run_experiment`
- report/persistence: `report_table`, `save_report_tables`, `save_metric_plot`, `save_outputs`

Existing cost, Greedy, Hungarian, Auction, proposal-support, and consensus algorithms remain owned by `run_multitask_peer_cost_experiment.py` and are reused directly rather than copied.

### Responsibility movement
The previous Experiment 2 owner contained MILP/ACO-specific process-pool runtime machinery. Those optimizer paths are no longer part of the main scale sweep, so that runtime concern is removed from this owner.

A new scalability concern is now explicit: receiver-local P2P views are streamed in bounded batches rather than materializing the full `receiver x sender x task` tensor.

Communication batching is owned by `sample_voter_batch_visibility` and `build_voter_batch_cost_views`; support accumulation is owned separately by `accumulate_proposal_support`. No second Voting state machine, second assignment solver, or hidden optimizer fallback is introduced.

### Preserved behavior
- `RANDOM_SEED = 20260903`.
- 30% independent directed scalar task-cost packet loss.
- every receiver always knows its own sender row.
- missing local costs remain `+inf`.
- scalar cost remains `0.05 + EuclideanDistance`.
- one simultaneous task per robot remains mandatory.
- task delivery remains reliable.
- final proposal collection remains reliable/in-window in this controlled stage.
- Greedy, Hungarian, and Auction implementations are unchanged.
- Auction epsilon-scaling implementation is unchanged.
- proposal support and support-maximizing consensus are unchanged.
- true cost is not used as a hidden consensus tie-break.
- evaluation still records average gap, optimal-cost match, near-optimal <=5%, exact assignment, and valid local proposal rate.
- Experiment 1 is unchanged.
- Direct-vs-Voting ablation code is unchanged and remains supporting evidence only for the current one-page paper.

### Deliberately changed behavior

#### Matched-scale system size
Experiment 2 now uses:

```text
robot_count = task_count
```

at:

```text
50, 100, 200, 400, 600, 800, 1000
```

This avoids violating the capacity-one contract at 1000 tasks and makes the x-axis a system-scale variable.

#### Optimizer set
The main lossy scaling sweep now contains:

```text
Voting Greedy
Voting Hungarian
Voting Auction
```

MILP and ACO + Local Search are intentionally excluded because their receiver-local solve cost dominated the previous Experiment 2 runtime. Their owner modules are not deleted or changed by this commit.

#### Streamed receiver batches
The experiment no longer attempts to allocate the full 1000x1000x1000 visibility tensor or the corresponding full float receiver-cost tensor.

Default:

```text
voter_batch_size = 8
```

For each batch, one visibility realization is generated and reused by all three methods. Proposal support is accumulated, then the large local views are discarded before the next batch.

A regression harness verified that changing receiver batch size changes completion/memory scheduling only and produces identical final method cost/gap/validity results for fixed seed/configuration.

#### Preview voter cap
Canonical mode still uses all robots as voting receivers.

For rapid trend inspection, the CLI now supports:

```text
--max-voters N
```

When supplied, the experiment uses all robots for fleets smaller than `N` and a deterministic random sample of `N` receivers for larger fleets. All methods share the same selected receivers and packet-loss views.

This preview mode is explicitly non-canonical and its output must not be silently mixed with all-voter formal data.

#### Primary one-page metric
The primary Experiment 2 figure is now:

```text
Trials within 5% of optimum (%)
```

Generated report plots are line-only without point markers. The Oracle remains available in CSV tables but is not plotted as a trivial quality curve.

#### Output separation
The new scale data is stored separately from the historical fixed-100-robot data:

```text
results/multitask_peer_cost_scaling/
```

Raw/summary files:

```text
scaling_comparison_raw.csv
scaling_comparison_summary.csv
```

### Paired-scenario contract
For each `(task_count, trial)`:

```text
trial_seed = seed + task_count * 100003 + trial * 1009
```

Separate deterministic streams derive:

- scenario geometry, task order, and tie priority;
- optional voter selection;
- packet-loss visibility.

The three Voting methods receive the same cost matrix, same voter identities, same receiver-local packet-loss views, same task order, and same tie priority.

No optimizer consumes or regenerates the communication RNG.

### Diagnostic contract
Representative new failures:

```text
owner=run_multitask_peer_cost_all_optimizers
function=validate_scaling_config
category=contract
code=INVALID_MAX_VOTERS
```

```text
owner=run_multitask_peer_cost_all_optimizers
function=sample_voter_batch_visibility
category=state
code=RECEIVER_INDEX_OUT_OF_RANGE
```

```text
owner=run_multitask_peer_cost_all_optimizers
function=build_voter_batch_cost_views
category=contract
code=VISIBILITY_SHAPE_MISMATCH
```

```text
owner=run_multitask_peer_cost_all_optimizers
function=finalize_voting_assignments
category=planning
code=NO_VALID_PROPOSALS
```

The zero-loss contract continues to fail at `validate_zero_loss_optimizer_contract` if Hungarian or Auction does not match the Oracle within the existing generic tolerance.

### Validation performed
- The replacement owner passed Python bytecode compilation.
- An interface-compatible end-to-end harness ran matched robot/task scales, sampled voters, batched visibility, all three Voting methods, support accumulation, consensus, aggregation, and metrics successfully.
- A batch-invariance harness ran the same full-voter scenario with voter batch sizes `1` and `3`; the resulting method total costs, optimality gaps, and valid proposal rates were exactly equal.
- These harnesses use interface-compatible stub solver owners and are not formal repository result data.
- The real repository smoke on the user's macOS/SciPy environment remains required, especially at the 1000-task Auction boundary.

### Known limitations / unfinished risks
- Full canonical mode with 1000 robots, 1000 tasks, all 1000 voters, and 100 trials remains computationally expensive even after MILP/ACO removal; batching fixes memory, not total optimization work.
- `--max-voters` preview results approximate the all-voter support distribution and are not automatically report-authoritative.
- The largest zero-loss Auction check at the configured maximum scale may expose a convergence or numerical-tolerance issue not visible at 100 tasks. If this occurs, stop at that named boundary rather than bypassing the check.
- The matched-scale curve answers a different question from the previous fixed-100-robot load sweep; old and new x-axis interpretations must not be conflated.

### Next step
Pull the new owner and run a very small real smoke first:

```bash
git pull
python run_multitask_peer_cost_all_optimizers.py \
  --tasks 50 100 200 \
  --trials 2 \
  --max-voters 50
```

If that completes, run the requested trend preview:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --tasks 50 100 200 400 600 800 1000 \
  --trials 10 \
  --max-voters 100
```

Inspect the `Trials within 5% of optimum (%)` table and curve before deciding whether the full all-voter 100-trial scale sweep is necessary for the one-page paper.

### Commit SHA
- `b0de17b0f73dd881c90812822a6149edf139684a` - replaced Experiment 2 with matched-scale 50-to-1000 Greedy/Hungarian/Auction Voting sweep and streamed voter batching.
- `726538f998b5072543bcb33d5dea996fb8567430` - updated the canonical Experiment 2 specification.
- `5be84038c0b686d5931906e6c794baba75aac263` - updated the report suite and current one-page rerun plan.
- continuity update commit: this file's commit.
