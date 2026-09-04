# Change Continuity

Historical continuity through `2026-09-04 - Redesign Experiment 2 as Fixed-100-Robot Workload Scaling` is preserved exactly in Git at:

```text
commit: 3269adf810ea4132b216fb39b3dee842255d53cb
blob:   a142dc4e9c1c1ac9fb9bd60c48d525c6de1dccdd
```

## 2026-09-04 - Narrow Experiment 2 to 100-Task Steps and Hand Off to Windows Runtime

### Purpose
The user kept the fixed physical fleet at `100` robots but decided that the report-facing Experiment 2 curve does not need 20 points at 50-task spacing. The canonical workload sweep is now reduced to ten points:

```text
100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
```

This preserves the same workload/capacity/communication model while cutting the number of formal task-load configurations in half.

The user also completed a real largest-load timing probe on macOS before moving the run to a Windows machine. That probe used `1000` tasks, one trial, one voter, all five optimizer families, and completed in approximately `85.44 s` wall time. It is a runtime measurement only, not report data.

Before modification, repository-root `AGENTS.md` and `docs/AI_CHANGE_PROTOCOL.md` were checked in order and remain absent. Then `docs/CHANGE_CONTINUITY.md`, canonical `docs/multitask_voting_mrta.md`, the actual Experiment 2 owner `run_multitask_peer_cost_all_optimizers.py`, exact functions `parse_task_counts`, `run_experiment`, and `parse_args`, and `docs/report_experiment_suite.md` were read before writing.

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

The changed configuration concern is exposed through:

- canonical default task set: `WORKLOAD_TASK_COUNTS`
- CLI/default task resolution: `parse_task_counts`
- experiment sweep: `run_experiment`
- CLI documentation/contract: `parse_args`

No optimizer owner changed.

True optimizer ownership remains:

```text
Greedy / Hungarian / Auction:
run_multitask_peer_cost_experiment.py

MILP / ACO + Local Search:
run_multitask_optimizer_screening.py
```

### Responsibility movement
No responsibility moved between modules.

The fixed-100 capacity-slot model, receiver-local communication model, optimizer routing, support accumulation, and final capacity consensus are unchanged. Only the canonical list of workload samples changed from 50-task spacing to 100-task spacing.

### Preserved behavior
- physical robot count remains exactly `100`;
- directed P2P scalar cost-message loss remains `30%`;
- scalar cost remains `0.05 + EuclideanDistance`;
- task counts represent allocation batch workload, not simultaneous physical execution;
- `capacity_per_robot = ceil(tasks/100)` remains unchanged;
- all enabled methods share the same scenario, voter identities, communication realization, task order, tie priority, and capacity;
- ACO keeps its separate deterministic per-receiver algorithm RNG;
- default Voting methods remain Greedy, Hungarian, and Auction;
- `--include-milp`, `--include-aco`, and `--only-milp` keep their preceding meanings;
- formal mode still means `100` trials and all `100` robots voting;
- receiver batching remains a runtime/memory control only;
- output root remains `results/multitask_peer_cost_fixed100_workload/`;
- the primary metric remains direct cost error from the full-information capacitated minimum;
- exact-overlap curve merging remains presentation-only;
- Experiment 1 and Experiment 3 are unchanged.

### Deliberately changed behavior
The canonical/default workload points changed from:

```text
50, 100, 150, ..., 1000
```

to:

```text
100, 200, 300, ..., 1000
```

Therefore the no-`--tasks` full Experiment 2 command now runs ten configurations rather than twenty.

The canonical/report documentation now records the real macOS timing boundary:

```text
T=1000
1 trial
1 voter
all five optimizer families
85.44 s wall time
```

This timing datum is explicitly diagnostic/runtime evidence and must not be plotted as Experiment 2 result data.

### Diagnostic contract
No new diagnostic category or code was introduced.

Existing first-failure diagnostics remain unchanged, including:

```text
owner=run_multitask_peer_cost_all_optimizers
function=validate_workload_config
category=contract
code=TASK_COUNT_ABOVE_REPORT_BOUNDARY
```

and true optimizer failures continue to propagate from their owners, including MILP `INVALID_MILP_COST` and ACO `INVALID_ACO_COST`.

### Validation performed
- The updated owner was re-fetched from GitHub after the write.
- `WORKLOAD_TASK_COUNTS` was confirmed as `tuple(range(100, 1001, 100))`.
- `run_experiment` and `parse_args` were re-read and confirmed to describe the 100-to-1000 default sweep.
- The preceding real macOS smoke already demonstrated that the unchanged 1000-task capacity-10 path can execute all five methods successfully under SciPy/HiGHS and the ACO implementation.
- No GitHub CI status checks are configured for this code block.

### Known limitations / unfinished risks
- The full formal run is still computationally expensive because every task load uses 100 trials x 100 receiver-local voters.
- The measured macOS largest-load boundary of 85.44 s for one voter means the 1000-task formal point alone can be very expensive if runtime scales approximately with voter/trial count.
- Moving to a faster Windows desktop may reduce wall time, but the current Experiment 2 owner is still primarily serial across receiver-local optimizer calls; GPU hardware does not accelerate the current NumPy/SciPy/Python implementation automatically.
- ACO and MILP remain the likely runtime bottlenecks at large task loads.
- The current script writes the report CSVs only after `run_experiment` completes. An interrupted long formal run therefore does not yet have a resumable checkpoint contract. Adding checkpoint/resume would be a separate runtime-resilience change.

### Next step
On the Windows machine, pull the new ten-point default and first reproduce the largest-point timing boundary:

```powershell
git pull
python run_multitask_peer_cost_all_optimizers.py --tasks 1000 --trials 1 --max-voters 1 --voter-batch-size 1 --include-milp --include-aco
```

Then run the ten-point low-cost machine benchmark:

```powershell
python run_multitask_peer_cost_all_optimizers.py --trials 1 --max-voters 5 --voter-batch-size 1 --include-milp --include-aco
```

If the measured Windows runtime is acceptable, the intended complete formal Experiment 2 command is:

```powershell
python run_multitask_peer_cost_all_optimizers.py --include-milp --include-aco
```

### Commit SHA
- `a5261455fc64fae029f2a42a06c9baeb221d3eb0` - changed the Experiment 2 owner default workload to `100..1000` by `100` and synchronized CLI text.
- `a34c4364f4c740f389fae6778ca79085b11e6803` - updated canonical Experiment 2 specification and Windows runtime handoff.
- `08669a1411b9ba2bf591da0b66cce6a27f0c5e66` - synchronized the report experiment suite.
- continuity update commit: this file's commit.
