# Change Continuity

Historical continuity through `2026-09-04 - Replace Slow Lossy Optimizers with Fast Heuristics and Parallel Trials` is preserved exactly in Git at:

```text
commit: 838a715e340791f500db2880f0010d9d7af6069c
blob:   e1f098316ed33bf5a40e0fdd00975c683c6d21ea
```

## 2026-09-04 - Replace Voting Auction with Voting Hungarian for the One-Hour Runtime Target

### Purpose
The user completed the all-voter parallel timing preview for the first fast Experiment 2 redesign:

```text
task points = 100, 500, 1000
trials per point = 2
voters = all 100 robots
workers = 4
methods = Sequential Greedy, Global Greedy, Static Regret-2 Greedy, Auction
wall time = 193.18 s
```

That measured preview extrapolates to roughly `1 h 47 min` for the canonical ten task points x 20 trials if runtime scales similarly, so it still misses the approximately one-hour target.

The user approved replacing receiver-local Voting Auction with receiver-local Voting Hungarian while keeping the three fast heuristics, all 100 voters, 20 trials per task point, the fixed 100-robot workload contract, and four-process trial parallelism.

Before modification, repository-root `AGENTS.md` and `docs/AI_CHANGE_PROTOCOL.md` were checked in that order and remain absent. Then `docs/CHANGE_CONTINUITY.md`, canonical `docs/multitask_voting_mrta.md`, actual owner `run_multitask_peer_cost_all_optimizers.py`, exact owner function `solve_voter_batch_proposals`, exact preflight functions, and true Hungarian owner `run_multitask_peer_cost_experiment.py::solve_hungarian_assignment` were read before writing.

### Files
- `run_multitask_peer_cost_all_optimizers.py`
- `docs/multitask_voting_mrta.md`
- `docs/report_experiment_suite.md`
- `docs/CHANGE_CONTINUITY.md`

No heuristic-owner implementation changed.

### Owner and named functions
Experiment 2 remains owned by:

```text
run_multitask_peer_cost_all_optimizers.py
```

Changed named boundaries:

- method-set ownership: `resolve_voting_methods`
- receiver-local exact routing: `solve_voter_batch_proposals`
- exact preflight: `validate_zero_loss_hungarian_contract`
- combined preflight: `validate_zero_loss_optimizer_contract`
- experiment reporting: `run_experiment`

The actual Hungarian algorithm remains owned by:

```text
run_multitask_peer_cost_experiment.py::solve_hungarian_assignment
```

Experiment 2 does not duplicate Hungarian or create a second exact solver state machine.

### Responsibility movement
No responsibility moved to a wrapper.

The Experiment 2 adapter already owns capacity-slot representation and receiver-local routing. It now routes the exact local branch to existing `p2p_hungarian` instead of existing `p2p_auction`.

The true optimization algorithm remains in the existing Hungarian owner. The full-information Oracle also uses that same owner, but under a different information condition.

### Preserved behavior
- fixed physical fleet: `100` robots;
- task batches: `100, 200, ..., 1000`;
- `capacity_per_robot = ceil(tasks/100)`;
- directed scalar cost-message loss: `30%`;
- scalar cost: `0.05 + EuclideanDistance`;
- formal Experiment 2 trials: `20` per task point;
- formal voters: all `100` robots;
- default process workers: up to `4`;
- default receiver batch size: `4`;
- paired scenario, voter, packet-loss, task-order, tie-priority, and capacity inputs;
- three fast heuristics and their dedicated owner;
- physical proposal support and capacitated final Voting consensus;
- Hungarian Oracle as the full-information minimum-cost reference;
- primary metric: direct cost error from the minimum;
- output root and CSV schema;
- Experiment 1 and Experiment 3.

### Deliberately changed behavior
Canonical report-facing Voting methods are now:

```text
Voting Sequential Greedy
Voting Global Greedy
Voting Static Regret-2 Greedy
Voting Hungarian
```

`Voting Auction` is removed from the canonical lossy workload sweep. Its implementation is not deleted.

The exact local branch now performs:

```text
receiver-local physical incomplete cost view
-> capacity-slot expansion
-> existing p2p_hungarian / solve_hungarian_assignment owner
-> slot-to-physical mapping
-> Voting support
```

The figure therefore normally has four optimizer curves. `Hungarian Oracle` remains in the CSV and supplies the `0%` reference, but it is not plotted as a fifth quality curve.

### Why Oracle and Voting Hungarian are not duplicates

```text
Hungarian Oracle
  full true cost matrix
  one minimum-cost solve per trial
  defines C_min

Voting Hungarian
  one lossy receiver-local cost matrix per voter
  100 local proposals per formal trial
  proposal-support Voting consensus
```

They use the same underlying assignment algorithm but test different information/aggregation conditions.

### Diagnostic contract
The exact preflight boundary is now:

```text
owner=run_multitask_peer_cost_all_optimizers
function=validate_zero_loss_hungarian_contract
category=planning
code=ZERO_LOSS_NOT_ORACLE_CONSISTENT
```

Unknown method routing still fails at:

```text
owner=run_multitask_peer_cost_all_optimizers
function=solve_voter_batch_proposals
category=contract
code=UNKNOWN_METHOD
```

True Hungarian owner diagnostics continue to propagate unchanged from `run_multitask_peer_cost_experiment.py`.

### Validation performed
- Re-fetched the modified owner after the code commit.
- Confirmed `DEFAULT_VOTING_METHODS` contains the three heuristic methods plus `p2p_hungarian`.
- Confirmed `METHOD_LABELS` exposes `Voting Hungarian` and no longer exposes `Voting Auction` in the canonical owner.
- Confirmed `solve_voter_batch_proposals` routes the exact branch through existing `solve_local_optimizer_proposals("p2p_hungarian", ...)` after capacity-slot expansion.
- Confirmed the exact preflight is renamed to `validate_zero_loss_hungarian_contract` and compares complete-information Voting Hungarian against the capacitated Hungarian Oracle.
- Canonical and report-suite documents were synchronized.
- No repository CI checks are available here; the user's Mac remains the real runtime boundary.

### Unfinished risks
- The one-hour target is still a target, not a guarantee. The same `100/500/1000 x 2 trials x 100 voters x 4 workers` timing preview must be rerun after this exact-solver swap.
- Hungarian still uses a capacity-slot matrix; at 1000 tasks this is a `1000 x 1000` local assignment problem per voter, although SciPy's Hungarian path is expected to be substantially faster than the prior Python Auction loop.
- The three heuristic valid-proposal rates can legitimately fall under severe incomplete information and tight capacity; no hidden fallback is added.
- Formal outputs are still written only after the complete experiment returns; checkpoint/resume remains separate future work.

### Next step
Pull and rerun the same measured preview:

```bash
git pull

time python run_multitask_peer_cost_all_optimizers.py \
  --tasks 100 500 1000 \
  --trials 2 \
  --workers 4
```

If the measured wall time is sufficiently reduced, run the canonical Experiment 2:

```bash
time python run_multitask_peer_cost_all_optimizers.py
```

### Commit SHA
- `50b7c7d0bd9e071740888a6ade98600dc3eebc67` - replace canonical Voting Auction routing/preflight/label with Voting Hungarian.
- `d5c768a0fc2461d430fa363682dc98d0e566dc17` - update canonical Experiment 2 specification.
- `8740595f00d66f07e81470951255b8ea09f027db` - update report experiment suite and four-curve contract.
- continuity update commit: this file's commit.
