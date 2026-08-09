# Voting-MRTA

A simulation project for a voting-based Multi-Robot Task Allocation (MRTA) method under communication packet loss.

The experiments intentionally use a simple, controlled single-task setting so communication robustness and voting behavior can be studied separately before extending the method to multi-task allocation.

## Common experiment settings

- Single task
- Robot counts: `5, 10, 15, ..., 100`
- Trials per configuration: `100`
- Fixed robot costs
- One vote per active robot
- Packet loss is applied to the vote -> terminal communication stage
- Reproducible random seeds

### Fixed cost model

For Robot `i` (1-based index):

```text
C_i = 10 + 5(i - 1)
```

Without robot failure, Robot 1 always has the minimum cost. In experiments with permanent failure, the optimal robot is the minimum-cost robot among the robots that are still active in that trial.

### Baseline cost-to-vote probability

The original Voting-MRTA rule converts lower cost into larger voting probability:

```text
w_i = (1 / C_i)^alpha
p_i = w_i / sum(w_j)
```

The original baseline uses `alpha = 1.0`.

## Experiment 1: packet-loss sweep

This experiment compares six vote-message packet-loss rates:

- 5%
- 10%
- 15%
- 20%
- 25%
- 30%

Every robot first casts one vote. The same complete set of ballots is reused for all packet-loss rates within a trial, so the curves are paired comparisons of the same underlying voting decision.

Run:

```bash
python run_experiment.py
```

Generated figures:

```text
results/figures/winner_preservation_rate.png
results/figures/optimal_win_rate.png
results/figures/average_regret.png
results/figures/tie_rate.png
results/figures/packet_loss_validation.png
```

Generated data:

```text
results/data/raw_results.csv
results/data/summary_results.csv
```

## Experiment 2: retransmission at 30% packet loss + 5% permanent robot failure

This experiment fixes the packet-loss probability of each individual transmission attempt at `30%` and independently gives every robot a `5%` probability of being permanently failed in each trial.

A permanently failed robot:

- does not cast a vote
- does not transmit or retransmit
- cannot be selected as the task winner

The minimum-cost **active** robot is used as the optimal reference for that trial.

The experiment compares maximum transmission attempts:

```text
1, 2, 3, ..., 10
```

The model uses **stop-on-success retransmission**. An active robot retries the same vote until one attempt succeeds or the configured maximum attempt count is reached. The terminal counts at most one vote from each active robot.

For an active robot, if the per-attempt loss probability is `p = 0.30`, the theoretical probability that its vote is still lost after `k` attempts is:

```text
p_effective = 0.30^k
```

With permanent failure included, the theoretical total unavailable fraction is:

```text
p_total_unavailable = 0.05 + 0.95 * (0.30^k)
```

Therefore retransmission can drive temporary packet loss close to zero, but it cannot remove the approximately 5% floor caused by permanently failed robots.

Run:

```bash
python run_retransmission_experiment.py
```

Generated retransmission figures:

```text
results/retransmission/figures/retransmission_winner_preservation_rate.png
results/retransmission/figures/retransmission_optimal_win_rate.png
results/retransmission/figures/retransmission_average_regret.png
results/retransmission/figures/retransmission_tie_rate.png
results/retransmission/figures/retransmission_effective_loss_rate.png
results/retransmission/figures/retransmission_overhead.png
results/retransmission/figures/retransmission_robot_failure_validation.png
```

Generated retransmission data:

```text
results/retransmission/data/retransmission_raw_results.csv
results/retransmission/data/retransmission_summary_results.csv
results/retransmission/data/retransmission_by_attempt.csv
```

The current results show that approximately 3-4 maximum transmission attempts already remove most of the temporary packet-loss effect. This motivates fixing the communication settings and studying the voting decision rule separately.

## Experiment 3: voting algorithm comparison

This experiment keeps the communication environment fixed so differences are mainly caused by the voting algorithm:

```text
Packet loss per attempt = 30%
Permanent robot failure = 5%
Maximum transmission attempts = 3
Trials per robot count = 100
```

The following single-algorithm voting strategies are compared:

1. `Inverse alpha=1` - original Voting-MRTA baseline
2. `Inverse alpha=2`
3. `Inverse alpha=3`
4. `Inverse alpha=4`
5. `Softmax beta=2`
6. `Softmax beta=4`
7. `Greedy`

For inverse-cost voting:

```text
p_i proportional to (1 / C_i)^alpha
```

For Softmax voting, the cost difference from the minimum active cost is divided by a fixed cost scale equal to the experiment cost step (`5` cost units):

```text
scaled_cost_i = (C_i - C_min) / 5
p_i proportional to exp(-beta * scaled_cost_i)
```

The fixed scale prevents Softmax from becoming artificially flatter when more robots are added.

`Greedy` is included as an upper-bound baseline for the current single-task setting. Because all active costs are known and only one task is assigned, directly choosing the minimum-cost active robot is mathematically optimal here.

Run:

```bash
python run_algorithm_experiment.py
```

Generated algorithm-comparison figures:

```text
results/algorithms/figures/algorithm_optimal_win_rate.png
results/algorithms/figures/algorithm_full_vote_optimal_win_rate.png
results/algorithms/figures/algorithm_average_regret.png
results/algorithms/figures/algorithm_winner_preservation_rate.png
results/algorithms/figures/algorithm_tie_rate.png
results/algorithms/figures/algorithm_method_summary.png
```

Generated algorithm-comparison data:

```text
results/algorithms/data/algorithm_raw_results.csv
results/algorithms/data/algorithm_summary_results.csv
results/algorithms/data/algorithm_by_method.csv
```

## Experiment 4: heterogeneous multi-algorithm voting

This experiment directly tests whether combining different local decision rules through final voting can improve the result compared with using one rule everywhere.

The communication environment remains fixed:

```text
Packet loss per attempt = 30%
Permanent robot failure = 5%
Maximum transmission attempts = 3
Trials per robot count = 100
```

Each active robot is assigned one decision rule. For a multi-algorithm strategy, active voters are randomly ordered and distributed as evenly as possible across the component rules. The assignment is recreated in every trial so algorithm assignment is not tied to Robot ID or cost rank.

All compared strategies in the same trial share:

- the same permanent-failure mask
- the same random vote draws
- the same tie priority
- the same packet-loss/retransmission outcomes
- the same randomized voter-assignment order

This keeps the comparison paired and isolates the effect of the decision-rule mixture.

### Single-rule baselines

```text
Single: Inverse alpha=1
Single: Inverse alpha=2
Single: Inverse alpha=3
Single: Softmax beta=0.25
Single: Softmax beta=0.5
Single: Greedy
```

The softer Softmax parameters are used here intentionally. Experiment 3 showed that large beta values are already very close to Greedy, so smaller beta values preserve enough decision diversity to make heterogeneous voting meaningful.

### Multi-algorithm strategies

`Multi Balanced`:

```text
Inverse alpha=1
+ Inverse alpha=2
+ Softmax beta=0.25
```

`Multi Strong`:

```text
Inverse alpha=2
+ Inverse alpha=3
+ Softmax beta=0.5
```

`Multi Diverse`:

```text
Inverse alpha=1
+ Softmax beta=0.25
+ Greedy
```

Each active robot still casts exactly one final vote. The terminal does not know or care which local rule generated that vote; it only aggregates the received votes using the same Voting-MRTA terminal logic.

Run:

```bash
python run_multi_algorithm_experiment.py
```

Generated figures:

```text
results/multi_algorithm/figures/multi_algorithm_optimal_win_rate.png
results/multi_algorithm/figures/multi_algorithm_average_regret.png
results/multi_algorithm/figures/multi_algorithm_tie_rate.png
results/multi_algorithm/figures/multi_algorithm_winner_preservation_rate.png
results/multi_algorithm/figures/multi_algorithm_strategy_summary.png
results/multi_algorithm/figures/multi_algorithm_gain_vs_components.png
```

Generated data:

```text
results/multi_algorithm/data/multi_algorithm_raw_results.csv
results/multi_algorithm/data/multi_algorithm_summary_results.csv
results/multi_algorithm/data/multi_algorithm_by_strategy.csv
```

The most direct figure for the research question is:

```text
multi_algorithm_gain_vs_components.png
```

For every heterogeneous strategy it compares:

1. the mean Optimal Win Rate of its single-rule components
2. the best Optimal Win Rate among those components
3. the final Optimal Win Rate after the component rules are mixed across robots and aggregated by voting

The CSV also reports:

```text
gain_vs_component_mean
gain_vs_best_component
```

A positive `gain_vs_component_mean` means heterogeneous voting performs better than the average of its constituent single algorithms. A positive `gain_vs_best_component` is a stronger result: it means the combined vote outperforms even the strongest individual component under the same experiment conditions.

## Main metrics

1. **Winner Preservation Rate** - probability that communication loss does not change the complete-vote winner.
2. **Optimal Win Rate** - probability that the final winner is the minimum-cost active robot.
3. **Complete-Vote Optimal Win Rate** - intrinsic decision quality before temporary packet loss.
4. **Average Regret** - `selected_robot_cost - minimum_active_robot_cost`.
5. **Tie Rate** - fraction of trials with a top-vote tie before tie-breaking.
6. **Effective Packet Loss Rate** - fraction of active votes still undelivered after all allowed attempts.
7. **Total Unavailable Rate** - permanent failures plus active votes still lost after retries.
8. **Average Actual Transmissions per Active Vote** - communication overhead under stop-on-success retransmission.
9. **Gain vs Component Mean** - multi-algorithm Optimal Win Rate minus the mean of its constituent single-rule rates.
10. **Gain vs Best Component** - multi-algorithm Optimal Win Rate minus the strongest constituent single-rule rate.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the experiments independently with:

```bash
python run_experiment.py
python run_retransmission_experiment.py
python run_algorithm_experiment.py
python run_multi_algorithm_experiment.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Inspect one voting round

To see the original single-transmission voting process vote-by-vote:

```bash
python demo_single_vote.py --robots 20 --loss 0.20
```

## GitHub Actions

The workflow under `.github/workflows/run-experiment.yml` runs all four experiments and uploads all generated CSV files and PNG figures as the `voting-mrta-results` workflow artifact.

## Current scope

The current code studies **vote-message packet loss, retransmission, permanent robot communication failure, single-algorithm voting quality, and heterogeneous multi-algorithm voting**. Cost-sharing packet loss, dynamic cost, multiple simultaneous tasks, robot capacity constraints, task reassignment, and multi-task allocation remain future extensions.
