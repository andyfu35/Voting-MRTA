# Change Continuity

Historical continuity through `2026-09-03 - Fix 100-Task Auction Convergence` remains preserved verbatim in `docs/CHANGE_CONTINUITY_ARCHIVE_20260903.md`.

The entries for Direct-vs-Voting ablation, optimizer-family screening, the MILP numerical boundary, and the superseded complete-information all-optimizer report-table plan are preserved exactly in Git. The immediately preceding active continuity state is:

```text
commit: aa0689d32f158eda6982ad84b5b7250ef2492e40
blob:   c425c8773611a01d46001593c56e7a23c54987d1
```

This compaction changes documentation layout only; no experiment behavior is changed by moving older active continuity into Git history.

## 2026-09-04 - Expand Experiment 2 to Five Optimizers under Lossy P2P Voting

### Purpose
Replace the report-facing complete-information optimizer table with the experiment that directly matches the research question: multiple optimizer families operating inside the same 30% lossy P2P Voting pipeline.

The user identified that the existing multi-task P2P experiment contained only Greedy, Hungarian, and Auction. This change adds MILP and ACO + Local Search while preserving the exact same paired robot/task and packet-loss scenarios used by the previous three methods.

The canonical report-facing Experiment 2 is now:

```text
Hungarian Oracle
Voting Greedy
Voting Hungarian
Voting Auction
Voting MILP
Voting ACO + Local Search
```

Robust Optimization and success/time/energy multi-objective costs remain deliberately excluded. The scalar spatial cost is unchanged.

Before code modification, repository-root `AGENTS.md` and `docs/AI_CHANGE_PROTOCOL.md` were checked in order and remain absent. `docs/CHANGE_CONTINUITY.md`, `docs/multitask_voting_mrta.md`, the actual P2P owner `run_multitask_peer_cost_experiment.py`, and the MILP/ACO owner functions in `run_multitask_optimizer_screening.py` were read before the change. The optimizer-screening canonical document was also re-read when its reusable solver contract was updated.

### Files
- `run_multitask_optimizer_screening.py`
- `run_multitask_peer_cost_all_optimizers.py`
- `docs/multitask_voting_mrta.md`
- `docs/multitask_optimizer_screening.md`
- `docs/report_experiment_suite.md`
- `docs/CHANGE_CONTINUITY.md`

### Owner and named functions

#### Existing optimizer-family owner: `run_multitask_optimizer_screening.py`

- MILP solve owner: `solve_milp_assignment`
- ACO candidate owner: `select_aco_candidates`
- one-ant construction owner: `construct_aco_assignment`
- ACO local-search owner: `improve_aco_assignment_locally`
- ACO solve owner: `solve_aco_assignment`
- ACO configuration contract: `validate_aco_config`

The MILP/ACO algorithms were not copied into the new experiment. Their true existing owner was extended so the same solver implementations can accept receiver-local incomplete cost matrices.

#### New Experiment 2 orchestration owner: `run_multitask_peer_cost_all_optimizers.py`

- experiment diagnostic boundary: `fail`
- method-specific numerical equality boundary: `cost_match_tolerance_percent`
- deterministic ACO receiver RNG owner: `aco_receiver_seed`
- local proposal output contract: `validate_local_proposal`
- extended local proposal orchestration: `solve_extended_local_optimizer_proposals`
- shared Voting path: `optimizer_consensus_assignment`
- zero-loss exact gate: `validate_zero_loss_optimizer_contract`
- final result evaluation: `evaluate_assignment`
- paired scenario owner: `run_trial`
- summary owner: `summarize_results`
- sweep owner: `run_experiment`
- report/persistence owners: `report_table`, `save_report_tables`, `save_metric_plot`, `save_outputs`

Existing communication, cost, Hungarian, Greedy, Auction, support, and consensus owners remain in `run_multitask_peer_cost_experiment.py` and are imported rather than copied.

### Responsibility movement
No communication, consensus, or existing P2P optimizer responsibility moved.

MILP and ACO remain owned by `run_multitask_optimizer_screening.py`; only their accepted cost-domain contract changed from "all finite only" to "finite values plus `+inf` unavailable edges" so they can be reused by a P2P receiver-local experiment.

The new Experiment 2 file owns only orchestration across optimizer families, deterministic ACO per-receiver RNG derivation, local proposal validation, paired trials, aggregation, and report outputs.

No second MILP implementation, second ACO state machine, second packet-loss generator, second Voting consensus implementation, or fallback-to-Hungarian repair was introduced.

### Preserved behavior
- `RANDOM_SEED = 20260903`.
- 100 robots.
- 30% independent directed sender->receiver scalar task-cost packet loss.
- 100 trials per task-count/method point.
- task counts `5,10,20,30,40,50,60,70,80,90,100`.
- scalar cost `0.05 + EuclideanDistance`.
- one simultaneous task per robot.
- task delivery remains reliable.
- final proposal collection remains reliable/in-window in this controlled stage.
- missing receiver-local costs remain represented as `+inf`.
- existing Greedy, Hungarian, Auction, support, and consensus implementations are unchanged.
- existing Auction epsilon-scaling behavior is unchanged.
- existing MILP `mip_rel_gap=0.0` and `MILP_NUMERICAL_TOLERANCE_PERCENT=1e-6` are unchanged.
- existing ACO parameters/search budget are unchanged.
- previous complete-information screening remains reproducible because its matrices are fully finite, so the new missing-edge branches are inactive there.
- Direct-vs-Voting ablation remains unchanged and still covers Hungarian/Auction only.

### Paired-scenario contract
For each task count, the new Experiment 2 intentionally keeps the exact previous P2P RNG schedule:

```text
rng = default_rng(seed + task_count * 100003)
```

Each trial consumes this shared stream in the same order as the previous three-optimizer experiment:

```text
generate cost matrix
generate packet-loss visibility tensor
generate task order
generate tie priority
```

No optimizer consumes that shared RNG afterward.

Therefore Voting Greedy, Voting Hungarian, and Voting Auction should reproduce the previous canonical three-method results when the same seed/configuration is used.

ACO receives a separate deterministic per-receiver stream:

```text
seed
+ 7000003
+ task_count * 100003
+ trial * 1009
+ receiver * 10000019
```

This makes ACO reproducible without changing any other method's scenario.

### Deliberately changed behavior

#### MILP missing-edge support
`solve_milp_assignment` now accepts finite values plus `+inf`.

For every unavailable edge:

```text
upper_bound(x_ij) = 0
```

The missing edge is therefore mathematically forbidden. No large artificial objective penalty is used.

`NaN` and `-inf` fail at:

```text
owner=run_multitask_optimizer_screening
function=solve_milp_assignment
category=data
code=INVALID_MILP_COST
```

If the incomplete graph has no complete feasible assignment, the function returns `None` and the calling experiment marks that receiver proposal invalid.

#### ACO missing-edge support
`solve_aco_assignment` now accepts finite values plus `+inf`.

`select_aco_candidates` already filters on finite cost, so missing edges cannot enter ant construction.

On complete information, the Greedy seed path is unchanged. On incomplete information, when the Greedy seed fails, ACO no longer immediately returns `None`; it continues the normal ant search and can still recover a feasible assignment.

`NaN` and `-inf` fail at:

```text
owner=run_multitask_optimizer_screening
function=solve_aco_assignment
category=data
code=INVALID_ACO_COST
```

#### New report-facing Experiment 2
`run_multitask_peer_cost_all_optimizers.py` now generates one receiver-cost tensor per trial and runs all five optimizer families on it before the shared support/Voting consensus boundary.

MILP/ACO proposals are validated before Voting. A selected unavailable edge, invalid robot index, capacity violation, or wrong proposal shape fails at the first `validate_local_proposal` diagnostic boundary rather than being repaired.

### Zero-loss diagnostic contract
Before the formal 30% sweep, the new owner checks task counts `1,5,50,100`.

Required exact behavior:

- Voting Hungarian matches Oracle cost within the generic exact tolerance.
- Voting Auction matches Oracle cost within the generic exact tolerance.
- Voting MILP matches Oracle cost within `1e-6%` MILP numerical tolerance.
- single-task Voting Greedy matches Oracle cost.
- exact-method receiver proposals are 100% valid at zero loss.

Representative failures:

```text
owner=run_multitask_peer_cost_all_optimizers
function=validate_zero_loss_optimizer_contract
category=planning
code=ZERO_LOSS_PROPOSAL_FAILURE
```

or:

```text
owner=run_multitask_peer_cost_all_optimizers
function=validate_zero_loss_optimizer_contract
category=planning
code=ZERO_LOSS_NOT_ORACLE_CONSISTENT
```

ACO is intentionally not exact-gated because it is a stochastic metaheuristic.

### Report outputs
Authoritative Experiment 2 output root:

```text
results/multitask_peer_cost_all_optimizers/
```

Primary metrics:

- average optimality gap percent;
- optimal-cost match percent;
- near-optimal within 5% percent;
- exact optimal assignment percent;
- average valid local proposal rate percent.

Terminal display values are rounded for readability; CSV values remain unrounded.

### Report-suite change
The complete-information `run_multitask_all_optimizer_experiment.py` is no longer a required main report experiment. It remains supporting/development evidence only.

The report now has three main controlled stories:

1. single-task communication robustness;
2. five optimizer families inside the same lossy P2P Voting experiment;
3. Direct-vs-Voting causal ablation.

### Validation performed
- `run_multitask_peer_cost_all_optimizers.py` passed Python bytecode compilation.
- A focused standalone MILP/ACO regression verified:
  - complete finite MILP still matches Hungarian;
  - incomplete MILP never selects a `+inf` edge;
  - incomplete ACO can construct a valid finite-edge assignment;
  - clearly infeasible missing-edge cases return `None` for MILP and ACO.
- An interface-compatible end-to-end smoke verified:
  - zero-loss exact contract path;
  - one shared P2P scenario feeding all five optimizer methods;
  - shared support/Voting consensus;
  - summary generation;
  - CSV/figure persistence;
  - readable six-column terminal tables including the Oracle.
- The interface-compatible smoke used a simplified Auction stub and is not formal repository result data. The user's real repository smoke remains required before the 100-trial canonical run.

### Known limitations / unfinished risks
- The new five-optimizer lossy experiment has not yet been executed on the user's real repository environment.
- MILP and especially ACO now solve one local optimization problem per receiver; the formal run is computationally much heavier than the previous three-method P2P experiment.
- ACO is stochastic approximate optimization, so it is not expected to match Hungarian/Auction/MILP exactly even at complete information.
- Valid proposal rate is especially important for ACO at high task load because incomplete local views can make construction difficult.
- The current Direct-vs-Voting ablation still tests Hungarian and Auction only. This change must not be used to claim Direct-vs-Voting uplift for MILP or ACO.
- The shared proposal-support consensus remains a controlled centralized boundary; do not describe it as fully asynchronous decentralized consensus.

### Next step
On the user's machine:

```bash
git pull
python run_multitask_peer_cost_all_optimizers.py --tasks 5 20 100 --trials 3
```

Only if that smoke reaches all three task counts, saves outputs, and prints all five Voting optimizer columns should the formal run begin:

```bash
python run_multitask_peer_cost_all_optimizers.py
```

After the formal 100-trial run, verify that the Greedy/Hungarian/Auction columns reproduce the preceding canonical P2P results, then use the new MILP/ACO columns for the report's optimizer-family comparison.

### Commit SHA
- `0bc8aaa81576f77fc1a85800d63bcc3426b63907` - extended existing MILP/ACO owners for `+inf` unavailable edges.
- `c65d2c40c6d254073b9a4a795b52128eebbf35f1` - added the five-optimizer lossy P2P Voting Experiment 2 owner.
- `beeeb25437021f0621b49dde56a5660220ac4a2b` - updated the canonical multi-task Voting specification.
- `b69441e26318cf47fa47b629a40457af9ba61561` - updated the optimizer-owner reusable missing-edge contract.
- `f3022c0033d0883f0d6b56cc3c972d0456a2dae7` - updated the canonical report experiment suite.
- `e714f27e5cd5b2b993208aeeb704cb9405a375a3` - recorded this continuity entry.
