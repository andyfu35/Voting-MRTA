# Change Continuity

Historical continuity through `2026-09-04 - Add MILP-Only 1000-Task Probe Mode` is preserved exactly in Git at:

```text
commit: 08e75ee2f6b15ce6d963f3fa656726ed006fdb47
blob:   c72f3feda8f960f8a6b5f43d9f1c1803da4177b4
```

## 2026-09-04 - Redesign Experiment 2 as Fixed-100-Robot Workload Scaling

### Purpose
The user clarified that Experiment 2 must keep the physical fleet fixed at `100` robots while increasing the task workload from `50` to `1000` in steps of `50`. The preceding matched-scale design (`robot_count == task_count`) therefore answered the wrong research question.

The capacity-one model could not legally allocate more than 100 tasks to 100 physical robots. This change introduces a single explicit fixed-fleet workload contract:

```text
physical robots = 100
T = 50, 100, 150, ..., 1000
capacity_per_robot = ceil(T / 100)
```

Task counts above 100 are now defined as an allocation batch/workload. They are not described as one robot physically executing multiple tasks simultaneously.

The user also wants to rerun Experiment 2 as a multi-optimizer comparison. The owner now supports Greedy, Hungarian, Auction, MILP, and ACO + Local Search under the same capacity, scenario, communication, and Voting contracts.

Before modification, repository-root `AGENTS.md` and `docs/AI_CHANGE_PROTOCOL.md` were checked in that order and remain absent. Then `docs/CHANGE_CONTINUITY.md`, canonical `docs/multitask_voting_mrta.md`, `docs/report_experiment_suite.md`, the actual owner `run_multitask_peer_cost_all_optimizers.py`, the exact Greedy/Hungarian/Auction/support owners in `run_multitask_peer_cost_experiment.py`, and the exact MILP/ACO owners in `run_multitask_optimizer_screening.py` were read before writing.

### Files
- `run_multitask_peer_cost_all_optimizers.py`
- `docs/multitask_voting_mrta.md`
- `docs/report_experiment_suite.md`
- `docs/CHANGE_CONTINUITY.md`

The one-page English CACS draft was also updated outside the repository before this code block, with Experiment 2 result space deliberately reserved rather than fabricated.

### Owner and named functions
Experiment 2 remains owned by:

```text
run_multitask_peer_cost_all_optimizers.py
```

New/updated named boundaries:

- method selection: `resolve_voting_methods`
- fixed-workload contract validation: `validate_workload_config`
- uniform capacity resolution: `resolve_robot_capacity`
- physical-cost to slot-cost transformation: `build_capacity_slot_cost_matrix`
- receiver-view to slot-view transformation: `build_capacity_slot_cost_views`
- slot-to-physical mapping: `map_slot_assignments_to_robots`
- physical capacity validation: `validate_capacity_assignment`
- physical capacity cost evaluation: `assignment_total_cost_with_capacity`
- full-information capacitated reference: `solve_capacity_oracle`
- physical voter selection: `resolve_voter_count`, `select_voter_indices`
- physical directed visibility: `sample_voter_batch_visibility`
- physical incomplete-view materialization: `build_voter_batch_cost_views`
- MILP batch iteration: `solve_milp_batch_proposals`
- ACO batch iteration / deterministic receiver RNG: `solve_aco_batch_proposals`
- optimizer-family routing on capacity slots: `solve_slot_voter_batch_proposals`
- physical proposal support: `accumulate_proposal_support`
- streamed receiver execution: `collect_voting_support`
- support/tie capacity transformation: `build_capacity_slot_consensus_inputs`
- capacitated support consensus: `solve_capacity_support_consensus`
- final assignment: `finalize_voting_assignments`
- complete-information integration path: `solve_zero_loss_consensus`
- bounded optimizer integration gates: `validate_zero_loss_optimizer_contract`
- evaluation: `evaluate_assignment`
- paired trial owner: `run_trial`
- aggregation: `summarize_results`
- sweep owner: `run_experiment`
- plotting/report output: `report_table`, `build_plot_series_groups`, `save_metric_plot`, `save_outputs`
- CLI: `parse_args`, `main`

True optimizer ownership remains unchanged:

```text
Greedy / Hungarian / Auction:
run_multitask_peer_cost_experiment.py

MILP / ACO + Local Search:
run_multitask_optimizer_screening.py
```

No second optimizer state machine or formulation is copied into Experiment 2.

### Responsibility movement
The previous Experiment 2 owner treated one optimizer row as one physical robot and therefore required `robot_count >= task_count`.

The new capacity concern is owned explicitly by Experiment 2 representation functions. Physical robot rows are repeated into capacity-one assignment slots before calling the existing optimizer owners:

```text
100 physical rows
x K identical capacity slots per physical robot
K = ceil(tasks / 100)
```

After optimization, slot assignments are mapped back to physical robot IDs.

This is a representation/modeling responsibility, not a wrapper that changes another owner's algorithm. The existing Greedy, Hungarian, Auction, MILP, ACO, and support-consensus implementations continue to solve their original capacity-one inputs unchanged.

Final Voting support remains accumulated on physical robot/task pairs, then physical support/tie rows are expanded to the same slot capacity only at the consensus solve boundary.

ACO's stochastic search now has an explicit Experiment 2 RNG boundary. Each physical receiver gets a deterministic algorithm-internal seed derived from the trial seed and receiver identity. This keeps ACO randomness separate from communication randomness and keeps results invariant to voter batch partitioning.

### Preserved behavior
- `RANDOM_SEED = 20260903`.
- Directed P2P scalar task-cost packet loss remains `30%`.
- Scalar cost remains `0.05 + EuclideanDistance(robot_i, task_j)`.
- Every physical receiver always knows its own physical sender row.
- Missing physical cost messages remain `+inf` unavailable edges.
- Task delivery remains reliable.
- Final proposal collection remains reliable/in-window in this controlled stage.
- Voters still solve local incomplete-information assignments and final execution still maximizes proposal support.
- True cost is still not used as a hidden consensus tie-break.
- Scenario, voter, and visibility RNG streams remain method-independent.
- Default fast method set remains Voting Greedy, Voting Hungarian, and Voting Auction.
- `--include-milp` still adds MILP.
- `--only-milp` remains available.
- The cost-error primary metric remains:

```text
100 * (method_cost - oracle_cost) / oracle_cost
```

- supporting optimal-cost-match, <=5%, exact-assignment, and valid-proposal metrics remain recorded.
- exact-overlap plot legend merging remains presentation-only and never merges raw result rows.
- Experiment 1 code/data are unchanged.
- Experiment 3 code/data are unchanged.
- Superseded matched-scale outputs are not deleted.

### Deliberately changed behavior

#### Fixed physical fleet
Experiment 2 now always uses:

```text
FIXED_ROBOT_COUNT = 100
```

The task x-axis is workload, not robot scale.

#### Dense canonical workload points
Default task batches are now:

```text
50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
550, 600, 650, 700, 750, 800, 850, 900, 950, 1000
```

#### Uniform batch capacity
For each task batch:

```text
capacity_per_robot = ceil(tasks / 100)
```

This capacity is used identically by the Oracle, every local optimizer, evaluation, and final consensus.

#### Capacity-slot representation
Existing capacity-one optimizer owners now receive a slot matrix with:

```text
slot_count = 100 * capacity_per_robot
```

Slots are identical copies of the physical robot's receiver-visible task-cost row. A missing physical robot/task edge therefore remains missing in all slots for that robot.

#### ACO integration
New CLI flag:

```text
--include-aco
```

It adds:

```text
Voting ACO + Local Search
```

using the existing `ACOConfig` and `solve_aco_assignment` owner. ACO receives its own deterministic per-receiver RNG stream and does not consume the communication RNG.

A complete five-family Voting comparison is requested with:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --include-milp \
  --include-aco
```

#### Bounded integration gates
- exact Voting Hungarian/Auction integration is checked only at bounded loads up to 200 tasks, including capacity > 1;
- MILP exact integration is checked up to 100 tasks under its existing numerical tolerance;
- ACO is checked for valid capacity-feasible complete-information output up to 150 tasks, but is not incorrectly required to equal the exact Oracle.

#### Output separation
New authoritative root:

```text
results/multitask_peer_cost_fixed100_workload/
```

Primary files:

```text
workload_comparison_raw.csv
workload_comparison_summary.csv
```

The previous matched-scale root remains historical and must not be mixed with the new workload results.

#### Report axis
The primary figure x-axis is now:

```text
Task batch size (100 robots fixed)
```

### Diagnostic contract
Representative new first-failure boundaries:

```text
owner=run_multitask_peer_cost_all_optimizers
function=validate_workload_config
category=contract
code=TASK_COUNT_ABOVE_REPORT_BOUNDARY
```

```text
owner=run_multitask_peer_cost_all_optimizers
function=resolve_robot_capacity
category=contract
code=INVALID_CAPACITY_INPUT
```

```text
owner=run_multitask_peer_cost_all_optimizers
function=map_slot_assignments_to_robots
category=state
code=SLOT_INDEX_OUT_OF_RANGE
```

```text
owner=run_multitask_peer_cost_all_optimizers
function=validate_capacity_assignment
category=state
code=CAPACITY_VIOLATION
```

```text
owner=run_multitask_peer_cost_all_optimizers
function=build_capacity_slot_consensus_inputs
category=contract
code=TIE_PRIORITY_SHAPE_MISMATCH
```

```text
owner=run_multitask_peer_cost_all_optimizers
function=solve_aco_batch_proposals
category=contract
code=RECEIVER_BATCH_SHAPE_MISMATCH
```

True optimizer diagnostics continue to propagate unchanged from their owners, including MILP `INVALID_MILP_COST` and ACO `INVALID_ACO_COST` data failures.

No wrapper replaces those first-failure diagnostics.

### Validation performed
- The new Experiment 2 owner passed Python bytecode compilation in an isolated container copy.
- An interface-compatible end-to-end harness exercised 100 fixed robots at 50 and 150 tasks, including capacity `1` and `2`, Greedy/Hungarian/Auction/MILP/ACO routing, physical proposal support, capacitated consensus, evaluation, and aggregation.
- The harness confirmed that a 150-task Oracle assignment never exceeds two tasks per physical robot.
- A batch-invariance harness ran the same 150-task/5-voter scenario with voter batch sizes `1` and `3`; method total costs, cost gaps, and valid proposal rates were identical.
- A 1000-task/1-voter shape harness exercised capacity `10` and `1000` assignment slots across all five Voting method routes.
- These validation harnesses used interface-compatible solver stubs and are not formal repository result data.
- The container cannot resolve `github.com`, so a real repository checkout/runtime test could not be performed there.
- The user's macOS/SciPy/HiGHS environment remains the required real runtime boundary, especially for Auction, MILP, and ACO at large task loads.

### Known limitations / unfinished risks
- At 1000 tasks, capacity expansion creates a `1000 x 1000` slot assignment matrix for each receiver view.
- Receiver-local MILP at 1000 tasks can involve roughly one million binary variables before presolve and may be very expensive.
- ACO + Local Search is also receiver-local and may dominate the complete five-family runtime.
- The intended complete run is `20 task points x 100 trials x 100 voters`, so all-family execution may be prohibitively long on a laptop.
- `DEFAULT_VOTER_BATCH_SIZE=8` controls receiver-view memory but does not reduce the number of local solves.
- A reduced `--max-voters` or reduced-trial run is preview data unless the canonical formal contract is explicitly revised.
- The uniform `ceil(T/100)` capacity is a deliberate workload-balancing contract; it does not model heterogeneous robot service rates or task durations.
- The current scalar cost does not include queue delay for a robot receiving multiple tasks. Adding execution order or service-time cost would be a separate research change and is intentionally not introduced here.

### Next step
Pull the fixed-fleet workload owner and first run a real all-family smoke:

```bash
git pull

python run_multitask_peer_cost_all_optimizers.py \
  --tasks 50 150 300 \
  --trials 1 \
  --max-voters 5 \
  --voter-batch-size 1 \
  --include-milp \
  --include-aco
```

If that completes, run a full-range low-cost timing preview:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --trials 3 \
  --max-voters 10 \
  --voter-batch-size 1 \
  --include-milp \
  --include-aco
```

If runtime is acceptable, the intended complete Experiment 2 rerun is:

```bash
python run_multitask_peer_cost_all_optimizers.py \
  --include-milp \
  --include-aco
```

After completion, use `workload_comparison_summary.csv` and `report_average_optimality_gap_percent.csv` to replace the reserved Experiment 2 figure in the one-page paper.

### Commit SHA
- `22ca8243d5f92a459d5d4f04bb4ea9b3abca8a2b` - redesigned Experiment 2 for 100 fixed robots, capacity-slot workload scaling, ACO integration, and new output root.
- `8e50f9804873e47278ed6d25abfd1f24295f4700` - updated canonical Experiment 2 specification.
- `90fe9b333068bdca41f19545fa22c5d84a1db359` - updated the canonical report experiment suite and rerun commands.
- continuity update commit: this file's commit.
