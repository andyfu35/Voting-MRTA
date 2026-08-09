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

### Single-rule baselines

```text
Single: Inverse alpha=1
Single: Inverse alpha=2
Single: Inverse alpha=3
Single: Softmax beta=0.25
Single: Softmax beta=0.5
Single: Greedy
```

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

The key finding motivating the next experiment is that heterogeneous voting can outperform the average quality of its component algorithms, but a fixed mixture does not necessarily outperform the best component in every operating condition.

## Experiment 5: adaptive context-aware multi-algorithm voting

Experiment 5 changes the research question from "does a fixed mixture help?" to:

> Can Voting-MRTA use different experts when the operating context changes, preserving each expert's strengths while reducing the effect of experts that are weak in that context?

The communication model remains:

```text
Packet loss per attempt = 30%
Permanent robot failure = 5%
Maximum transmission attempts = 3
Evaluation trials per robot count and scenario = 100
Calibration trials per robot count and scenario = 40
```

### Robot state used by the adaptive experiment

Every active robot candidate has four normalized attributes:

```text
Task cost
Energy risk
Communication risk
Load risk
```

The fixed task-cost ordering is still derived from:

```text
C_i = 10 + 5(i - 1)
```

Energy, communication, and load risk are independently regenerated in every trial.

### Operating contexts

The true task objective changes between five scenarios. The values below are weights for:

```text
[task cost, energy risk, communication risk, load risk]
```

```text
Cost-dominant          = [0.70, 0.10, 0.10, 0.10]
Energy-critical        = [0.20, 0.60, 0.10, 0.10]
Communication-critical = [0.20, 0.10, 0.60, 0.10]
Load-critical          = [0.20, 0.10, 0.10, 0.60]
Balanced               = [0.25, 0.25, 0.25, 0.25]
```

The optimal active robot in a trial is the robot with minimum weighted objective score for that scenario.

### Expert decision algorithms

Five expert rules generate candidate voting distributions:

1. `Cost Expert` - focuses only on task cost
2. `Energy-Aware Expert` - combines 40% task cost and 60% energy risk
3. `Communication-Aware Expert` - combines 40% task cost and 60% communication risk
4. `Load-Aware Expert` - combines 40% task cost and 60% load risk
5. `Balanced Expert` - equally considers all four attributes

Each expert converts its score into a probabilistic candidate preference using a fixed-strength Softmax. This preserves voting diversity while letting each expert specialize in a different type of operating condition.

### Compared fusion strategies

`Equal Expert Fusion`:

```text
All expert probability vectors receive equal weight.
```

`Context-Aware Fusion`:

```text
The current scenario determines which experts receive more weight.
```

`Context + Reliability Fusion`:

```text
final expert weight
    proportional to
context relevance * calibrated historical reliability
```

Reliability is estimated using 40 independent calibration rounds for the same robot count and context. Laplace smoothing prevents a short calibration run from completely removing an expert. Calibration and evaluation use separate random streams, so the adaptive strategy does not use the current evaluation trial's answer to select its weights.

The reliability multiplier is deliberately bounded so a context-relevant expert is down-weighted when its history is poor but is not instantly eliminated after a small sample.

`Oracle Objective` is included only as an upper-bound reference. It directly knows the true weighted objective and should not be interpreted as an implementable distributed method.

### Research comparison

The most important comparison is no longer only against the best expert in one scenario. It is whether one fixed expert can remain good across all five changing contexts compared with an adaptive fusion strategy.

The experiment therefore compares:

```text
Best fixed expert across all contexts
Equal expert fusion
Context-aware fusion
Context + reliability fusion
Oracle upper bound
```

Run:

```bash
python run_adaptive_voting_experiment.py
```

Generated figures:

```text
results/adaptive/figures/adaptive_overall_optimal_win_rate.png
results/adaptive/figures/adaptive_average_regret.png
results/adaptive/figures/adaptive_by_scenario.png
results/adaptive/figures/adaptive_strategy_summary.png
results/adaptive/figures/adaptive_expert_weights.png
results/adaptive/figures/adaptive_gain_vs_fixed_expert.png
```

Generated data:

```text
results/adaptive/data/adaptive_raw_results.csv
results/adaptive/data/adaptive_summary_results.csv
results/adaptive/data/adaptive_by_strategy.csv
results/adaptive/data/adaptive_expert_weights.csv
```

The most important figures are:

- `adaptive_by_scenario.png` - checks whether adaptive voting follows the best expert as the context changes.
- `adaptive_gain_vs_fixed_expert.png` - checks whether adaptation beats committing to one fixed expert across all contexts.
- `adaptive_expert_weights.png` - shows which expert types the context + reliability mechanism emphasizes or suppresses.

## Main metrics

1. **Winner Preservation Rate** - probability that communication loss does not change the complete-vote winner.
2. **Optimal Win Rate** - probability that the final winner is the minimum-objective active robot.
3. **Complete-Vote Optimal Win Rate** - intrinsic decision quality before temporary packet loss.
4. **Average Regret** - selected objective score minus the minimum active objective score.
5. **Tie Rate** - fraction of trials with a top-vote tie before tie-breaking.
6. **Effective Packet Loss Rate** - fraction of active votes still undelivered after all allowed attempts.
7. **Total Unavailable Rate** - permanent failures plus active votes still lost after retries.
8. **Average Actual Transmissions per Active Vote** - communication overhead under stop-on-success retransmission.
9. **Gain vs Component Mean** - heterogeneous Optimal Win Rate minus the mean of its constituent single-rule rates.
10. **Gain vs Best Component** - heterogeneous Optimal Win Rate minus the strongest constituent single-rule rate.
11. **Adaptive Expert Weight** - final context- and reliability-adjusted contribution assigned to each expert.
12. **Cross-Context Gain** - adaptive performance compared with the strongest single fixed expert averaged over all contexts.

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
python run_adaptive_voting_experiment.py
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

The workflow under `.github/workflows/run-experiment.yml` runs all five experiments and uploads all generated CSV files and PNG figures as the `voting-mrta-results` workflow artifact.

## Current scope

The current code studies **vote-message packet loss, retransmission, permanent robot communication failure, single-algorithm voting quality, fixed heterogeneous multi-algorithm voting, and adaptive context-aware multi-algorithm voting**. The adaptive experiment introduces changing robot/task-state priorities while keeping a single task. Cost-sharing packet loss, multiple simultaneous tasks, robot capacity constraints, task reassignment, and full multi-task allocation remain future extensions.
