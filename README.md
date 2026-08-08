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

Each figure uses robot count on the x-axis and contains six curves, one for each communication packet-loss rate.

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

The model uses **stop-on-success retransmission**:

1. An active robot sends its vote.
2. If the packet is delivered, transmission stops for that robot.
3. If the packet is lost, the same vote is retried until success or the configured maximum attempt count is reached.
4. The terminal counts at most one vote from each active robot, so retransmissions never create duplicate votes.
5. A permanently failed robot never responds, regardless of the configured retransmission count.

For an active robot, if the per-attempt loss probability is `p = 0.30`, the theoretical probability that its vote is still lost after `k` maximum attempts is:

```text
p_effective = 0.30^k
```

When permanent robot failure is also included, the theoretical total unavailable fraction is:

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

### Important retransmission interpretation

Retransmission improves **temporary communication reliability**, but it cannot recover a permanently failed robot and it does not automatically make the voting rule mathematically optimal.

The current results show that approximately 3-4 maximum transmission attempts already remove most of the temporary packet-loss effect. This motivates fixing the communication settings and studying the voting decision rule separately.

## Experiment 3: voting algorithm comparison

This experiment keeps the communication environment fixed so differences are mainly caused by the voting algorithm:

```text
Packet loss per attempt = 30%
Permanent robot failure = 5%
Maximum transmission attempts = 3
Trials per robot count = 100
```

The same permanent-failure mask, random vote draws, tie priorities, and transmission outcomes are shared across methods within each trial. This gives a paired comparison under the same underlying conditions.

The following single-algorithm voting strategies are compared:

1. `Inverse alpha=1` - original Voting-MRTA baseline
2. `Inverse alpha=2`
3. `Inverse alpha=3`
4. `Inverse alpha=4`
5. `Softmax beta=2`
6. `Softmax beta=4`
7. `Greedy` - every active voter selects the minimum-cost active robot

For inverse-cost voting:

```text
p_i proportional to (1 / C_i)^alpha
```

Larger `alpha` gives the lower-cost candidates stronger preference.

For softmax voting, active-robot costs are first normalized within the active candidate set and then converted using:

```text
p_i proportional to exp(-beta * normalized_cost_i)
```

Larger `beta` gives lower-cost candidates stronger preference.

`Greedy` is included as an upper-bound baseline for the current single-task setting. Because all active costs are known and only one task is assigned, directly choosing the minimum-cost active robot is mathematically optimal here. This does not imply Greedy will remain globally optimal after the project is extended to multiple tasks and assignment constraints.

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

The most important comparison is between:

- **Complete-vote optimal rate**: intrinsic quality of the voting rule before temporary packet loss
- **Post-communication optimal rate**: final quality after 30% packet loss, 3-attempt retransmission, and 5% permanent robot failure
- **Average regret**: how costly a non-optimal selection is
- **Winner preservation**: whether communication changes the complete-vote result

This experiment is intended to identify strong single voting rules before building a future **heterogeneous multi-algorithm voting** experiment.

## Main metrics

1. **Winner Preservation Rate**  
   Probability that the winner after communication loss is the same as the winner with all active votes delivered.

2. **Optimal Win Rate**  
   Probability that the final winner is the minimum-cost active robot.

3. **Complete-Vote Optimal Win Rate**  
   Probability that the voting algorithm chooses the minimum-cost active robot before temporary packet loss is applied.

4. **Average Regret**  
   `selected_robot_cost - minimum_active_robot_cost`.

5. **Tie Rate**  
   Fraction of trials in which the highest received vote count is shared by multiple robots before tie-breaking.

6. **Effective Packet Loss Rate**  
   Fraction of active robot votes that remain undelivered after all allowed transmission attempts.

7. **Total Unavailable Rate**  
   Fraction of all robots that are unavailable because they are permanently failed or because an active robot's vote is still lost after all retries.

8. **Average Actual Transmissions per Active Vote**  
   Communication overhead when retransmission stops after the first successful delivery.

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

The workflow under `.github/workflows/run-experiment.yml` runs all three experiments and uploads all generated CSV files and PNG figures as the `voting-mrta-results` workflow artifact.

## Current scope

The current code studies **vote-message packet loss, retransmission, permanent robot communication failure, and single voting-algorithm selection quality**. Cost-sharing packet loss, dynamic cost, multiple simultaneous tasks, robot capacity constraints, task reassignment, and heterogeneous multi-algorithm voting are intentionally left for later experiments.
