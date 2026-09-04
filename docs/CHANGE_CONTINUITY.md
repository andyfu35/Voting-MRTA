# Change Continuity

Historical continuity through `2026-09-04 - Replace Voting Auction with Voting Hungarian for the One-Hour Runtime Target` is preserved exactly in Git at:

```text
commit: 713e348340715ed19e13909c90a47b2bde598151
blob:   39420b5cce5da41286c25df76f63879c5481cbb7
```

## 2026-09-04 - Replace Multiple Greedy Curves with Four Distinct Fast Optimizer Families

### Purpose
The user rejected a report figure dominated by multiple Greedy variants and requested only one Greedy baseline plus other computationally fast optimization families. The approximately one-hour macOS runtime target remains important.

Canonical Experiment 2 now compares:

```text
Voting Greedy
Voting Hungarian
Voting Min-Cost Flow
Voting Sinkhorn + Rounding
```

`Hungarian Oracle` remains the full-information minimum-cost reference and is not plotted as a fifth quality curve.

Before modification, repository-root `AGENTS.md` and `docs/AI_CHANGE_PROTOCOL.md` were checked in that order and remain absent. Then `docs/CHANGE_CONTINUITY.md`, canonical `docs/multitask_voting_mrta.md`, actual Experiment 2 owner `run_multitask_peer_cost_all_optimizers.py`, exact routing/preflight functions, existing Greedy owner `run_multitask_workload_heuristics.py`, existing Hungarian owner `run_multitask_peer_cost_experiment.py::solve_hungarian_assignment`, and `requirements.txt` were read before writing.

### Files
- `run_multitask_workload_optimizers.py` - new true owner for Min-Cost Flow and Sinkhorn-related optimization boundaries.
- `run_multitask_peer_cost_all_optimizers.py` - canonical method selection, routing, preflight, Voting, evaluation, and parallel runtime.
- `requirements.txt` - adds OR-Tools for capacity-native Min-Cost Flow.
- `docs/multitask_voting_mrta.md` - canonical Experiment 2 specification.
- `docs/report_experiment_suite.md` - report-authoritative rerun contract.
- `docs/CHANGE_CONTINUITY.md` - this continuity record.

The old Global Greedy, Static Regret-2 Greedy, Auction, MILP, and ACO implementations were not deleted.

### Owner and named functions

#### Single Greedy baseline owner

```text
run_multitask_workload_heuristics.py::solve_sequential_greedy_capacitated
```

#### New fast optimizer owner

```text
run_multitask_workload_optimizers.py
```

Named boundaries:

- common input validation: `validate_optimizer_problem`
- OR-Tools dependency: `require_min_cost_flow_dependency`
- Min-Cost Flow cost representation: `quantize_min_cost_flow_costs`
- Min-Cost Flow solver: `solve_min_cost_flow_capacitated`
- Min-Cost Flow batch execution: `solve_min_cost_flow_batch`
- Sinkhorn configuration validation: `validate_sinkhorn_config`
- Sinkhorn continuous optimization: `compute_sinkhorn_transport_plan`
- Sinkhorn discrete capacity rounding: `round_sinkhorn_plan_to_capacity`
- Sinkhorn composed method: `solve_sinkhorn_capacitated`
- Sinkhorn batch execution: `solve_sinkhorn_batch`

#### Experiment 2 owner

```text
run_multitask_peer_cost_all_optimizers.py
```

Changed named boundaries:

- method selection: `resolve_voting_methods`
- local owner routing: `solve_voter_batch_proposals`
- shared bounded preflight scenario: `build_zero_loss_case`
- Greedy preflight: `validate_zero_loss_greedy_contract`
- Hungarian preflight: `validate_zero_loss_hungarian_contract`
- Min-Cost Flow preflight: `validate_zero_loss_min_cost_flow_contract`
- Sinkhorn preflight: `validate_zero_loss_sinkhorn_contract`
- combined preflight: `validate_zero_loss_optimizer_contract`
- report tolerance ownership: `cost_match_tolerance_percent`

The true Hungarian implementation remains:

```text
run_multitask_peer_cost_experiment.py::solve_hungarian_assignment
```

### Responsibility movement
The canonical method set no longer imports all three capacitated Greedy variants. Experiment 2 routes only `p2p_sequential_greedy` to the existing heuristic owner.

Min-Cost Flow and Sinkhorn are not implemented as wrappers around Hungarian or another optimizer. They have a dedicated owner module and operate on the physical receiver-local `100 x T` matrix.

Min-Cost Flow owns its capacity-native network representation. Sinkhorn owns its soft transport computation, while discrete rounding is intentionally separated into its own named function because continuous optimization and discrete feasibility are different concerns.

Experiment 2 continues to own communication sampling, capacity-slot representation only where required by Hungarian/consensus, method routing, support accumulation, final Voting consensus, evaluation, and process-level trial scheduling.

### Preserved behavior
- physical robots remain fixed at `100`;
- canonical task batches remain `100, 200, ..., 1000`;
- `capacity_per_robot = ceil(tasks/100)`;
- directed P2P scalar cost-message loss remains `30%`;
- scalar cost remains `0.05 + EuclideanDistance`;
- formal Experiment 2 remains `20` trials per task point;
- formal Experiment 2 still uses all `100` physical voters;
- paired scenario, voter identities, packet-loss realization, task order, tie priority, and capacity remain shared by all methods;
- independent `(task_count, trial)` jobs remain process-parallel with default up to four workers;
- receiver batching remains a runtime/memory control only;
- final proposal support remains physical robot/task support under the same capacitated consensus;
- Hungarian Oracle remains the full-information minimum-cost reference;
- primary metric remains direct cost error from the minimum;
- output root and CSV schema remain unchanged;
- Experiment 1 and Experiment 3 are unchanged.

### Deliberately changed behavior
Previous canonical curves:

```text
Voting Sequential Greedy
Voting Global Greedy
Voting Static Regret-2 Greedy
Voting Hungarian
```

New canonical curves:

```text
Voting Greedy
Voting Hungarian
Voting Min-Cost Flow
Voting Sinkhorn + Rounding
```

Only one Greedy baseline remains report-facing. The other Greedy implementations remain available for supporting experiments.

Min-Cost Flow uses OR-Tools `SimpleMinCostFlow` with the physical capacity encoded on source-to-robot arcs. Receiver-visible float costs are multiplied by:

```text
MIN_COST_FLOW_COST_SCALE = 1_000_000
```

and rounded to integer arc costs because the OR-Tools owner requires integer costs. Final evaluation always uses the original float cost matrix. Complete-information Min-Cost Flow must be within `0.01%` of the float Hungarian Oracle in bounded preflight.

Sinkhorn uses:

```text
epsilon = 0.08
max_iterations = 30
tolerance = 1e-5
```

and then calls the separate `round_sinkhorn_plan_to_capacity` boundary. Sinkhorn is explicitly an approximation and is not required to equal the Oracle.

`requirements.txt` now includes:

```text
ortools>=9.10,<10
```

### Diagnostic contract
Experiment-level failures keep:

```text
owner=run_multitask_peer_cost_all_optimizers
function=<named function>
category=<data|time|state|dependency|planning|safety|runtime|contract>
code=<named code>
```

New optimizer-owner failures use:

```text
owner=run_multitask_workload_optimizers
```

Representative first-failure boundaries:

```text
function=require_min_cost_flow_dependency category=dependency code=ORTOOLS_NOT_AVAILABLE
function=validate_optimizer_problem category=data code=INVALID_OPTIMIZER_COST
function=validate_optimizer_problem category=planning code=TASK_WITHOUT_VISIBLE_EDGE
function=quantize_min_cost_flow_costs category=data code=MIN_COST_FLOW_COST_OVERFLOW
function=validate_sinkhorn_config category=contract code=INVALID_SINKHORN_EPSILON
function=compute_sinkhorn_transport_plan category=state code=SINKHORN_ROW_MASS_EXCEEDS_CAPACITY
function=round_sinkhorn_plan_to_capacity category=contract code=SINKHORN_PLAN_SHAPE_MISMATCH
```

Exact/quality preflight boundaries include:

```text
function=validate_zero_loss_hungarian_contract category=planning code=ZERO_LOSS_NOT_ORACLE_CONSISTENT
function=validate_zero_loss_min_cost_flow_contract category=planning code=ZERO_LOSS_NOT_ORACLE_CONSISTENT
function=validate_zero_loss_sinkhorn_contract category=planning code=ZERO_LOSS_PROPOSAL_FAILURE
```

### Validation performed
- The new optimizer owner was compiled successfully with Python bytecode syntax validation in the execution container.
- The rewritten Experiment 2 owner was also compiled successfully in the execution container.
- A standalone Sinkhorn test on a `5 x 10` matrix reached the requested row/column masses and rounded to exactly two tasks per robot.
- A standalone `100 x 1000` Sinkhorn solve in the container completed in about `0.009 s` for that synthetic matrix; this is implementation-level timing only and is not a Mac Experiment 2 runtime claim.
- OR-Tools is not installed in the execution container, so the Min-Cost Flow solver could not be executed there. Its missing-dependency diagnostic was exercised successfully.
- The OR-Tools bulk Python API used by the implementation (`add_arcs_with_capacity_and_unit_cost`, `set_nodes_supplies`, `flows`) was checked against current official OR-Tools examples/API documentation. Node arrays were adjusted to the documented `int32` dtype, with capacities/costs/supplies using `int64`.
- The modified Experiment 2 owner was re-fetched from GitHub and inspected after the write.
- Canonical and report-suite documents were synchronized with the four-method contract.

### Unfinished risks
- The one-hour target is not yet validated for the new four-method set on the user's Mac.
- OR-Tools must be installed in the user's existing virtual environment before the new canonical run can start.
- Min-Cost Flow runtime at `T=1000` with 100 receiver-local solves per trial is still unknown on the user's Mac and may become the new bottleneck.
- Sinkhorn rounding can legitimately fail under an incomplete view if its greedy discretization reaches a task with no remaining finite-capacity candidate. Such failures are exposed through `valid_proposal_rate_percent`; there is no hidden Hungarian fallback.
- Formal CSVs are still written only after `run_experiment` completes; checkpoint/resume remains a separate future concern.
- The existing one-page placeholder paper must be synchronized with Min-Cost Flow and Sinkhorn references after final Experiment 2 results are accepted.

### Next step
Update the Mac environment and run the small real-machine smoke:

```bash
git pull
pip install -r requirements.txt

time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 100 500 1000 \
  --trials 1 \
  --max-voters 5 \
  --workers 1
```

If that passes, immediately run the all-voter timing preview:

```bash
time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 100 500 1000 \
  --trials 2 \
  --workers 4
```

If the measured runtime fits the budget, start canonical Experiment 2:

```bash
time python run_multitask_peer_cost_all_optimizers.py
```

### Commit SHA
- `501487ca2366d713a4a391cd83da18a4754b6f06` - created the dedicated Min-Cost Flow/Sinkhorn optimizer owner.
- `bd629da9f46e33419c74b7599ce6aa3b58692a04` - added OR-Tools to project requirements.
- `5cad5fa234e7e15fb55da2ca9dbb4d02990ce997` - routed Min-Cost Flow cost conversion through its named quantization boundary.
- `a504dce16eac32f44992bb4f7cafadd9f139c80a` - replaced the canonical multiple-Greedy method set with Greedy/Hungarian/Min-Cost-Flow/Sinkhorn routing and preflights.
- `dbdf63abee0d357060f2c233e34886fa2f42dc80` - aligned OR-Tools node arrays with the documented Python API dtypes.
- `294c60e53f2848bbd2b94620479b738985433fe3` - updated the canonical Experiment 2 specification.
- `19ddac895de0f77cab78af40e05d6cc866c54e3a` - updated the report experiment suite.
- continuity update commit: this file's commit.
