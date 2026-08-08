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

Without robot failure, Robot 1 always has the minimum cost. In the permanent-failure experiment, the optimal robot is instead defined as the minimum-cost robot among the robots that are still active in that trial.

### Cost-to-vote probability

Lower cost should receive higher voting probability, so the weight is

```text
w_i = (1 / C_i)^alpha
```

and the normalized voting probability is

```text
p_i = w_i / sum(w_j)
```

The default experiments use `alpha = 1.0`.

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

This experiment fixes the packet-loss probability of each individual transmission attempt at:

```text
30%
```

and independently gives every robot a:

```text
5%
```

probability of being permanently failed in each trial.

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

Examples:

| Max attempts | Active-robot effective packet loss |
|---:|---:|
| 1 | 30% |
| 2 | 9% |
| 3 | 2.7% |
| 4 | 0.81% |
| 5 | 0.243% |
| 6 | 0.0729% |
| 7 | 0.02187% |
| 8 | 0.006561% |
| 9 | 0.0019683% |
| 10 | 0.00059049% |

When permanent robot failure is also included, the theoretical total unavailable fraction is:

```text
p_total_unavailable = 0.05 + 0.95 * (0.30^k)
```

Therefore retransmission can drive temporary packet loss close to zero, but it cannot remove the approximately 5% floor caused by permanently failed robots.

Within each trial, one shared transmission-attempt random matrix is generated. The 1-attempt through 10-attempt cases are therefore nested versions of exactly the same communication realization, which makes the retry comparison fair.

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

The first four figures keep robot count on the x-axis and contain ten curves, one for each maximum transmission-attempt setting from 1 to 10.

`retransmission_effective_loss_rate.png` compares temporary packet loss among active robots with the total unavailable-robot rate. The total-unavailable curve approaches the 5% permanent-failure floor as retransmission increases.

`retransmission_overhead.png` shows the average number of actual transmissions required per active vote when transmission stops immediately after success.

`retransmission_robot_failure_validation.png` checks that the simulated permanent robot-failure rate remains close to the configured 5% level.

Generated retransmission data:

```text
results/retransmission/data/retransmission_raw_results.csv
results/retransmission/data/retransmission_summary_results.csv
results/retransmission/data/retransmission_by_attempt.csv
```

### Important interpretation

Retransmission improves **temporary communication reliability**, but it cannot recover a permanently failed robot and it does not automatically make the voting rule mathematically optimal.

If temporary packet loss is nearly eliminated, Winner Preservation Rate should approach the full-communication result for the same active robot set. However, the approximately 5% permanently failed robots remain unavailable, and the complete voting process can still choose a non-minimum-cost active robot.

This separates three research questions:

1. temporary packet-loss robustness
2. permanent robot-failure robustness
3. voting-solution optimality

## Main metrics

1. **Winner Preservation Rate**  
   Probability that the winner after communication loss is the same as the winner with all active votes delivered.

2. **Optimal Win Rate**  
   Probability that the final winner is the minimum-cost active robot.

3. **Average Regret**  
   `selected_robot_cost - minimum_active_robot_cost`.

4. **Tie Rate**  
   Fraction of trials in which the highest received vote count is shared by multiple robots before tie-breaking.

5. **Effective Packet Loss Rate**  
   Fraction of active robot votes that remain undelivered after all allowed transmission attempts.

6. **Total Unavailable Rate**  
   Fraction of all robots that are unavailable because they are permanently failed or because an active robot's vote is still lost after all retries.

7. **Average Actual Transmissions per Active Vote**  
   Communication overhead when retransmission stops after the first successful delivery.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run either experiment:

```bash
python run_experiment.py
python run_retransmission_experiment.py
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

The workflow under `.github/workflows/run-experiment.yml` runs both experiments and uploads all generated CSV files and PNG figures as the `voting-mrta-results` workflow artifact.

## Current scope

The current code studies **vote-message packet loss, retransmission, and permanent robot communication failure**. Cost-sharing packet loss, dynamic cost, multiple simultaneous tasks, robot capacity constraints, task reassignment, and heterogeneous multi-algorithm voting are intentionally left for later experiments.
