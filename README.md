# Voting-MRTA

A simulation project for a voting-based Multi-Robot Task Allocation (MRTA) method under communication packet loss.

The experiments intentionally use a simple, controlled single-task setting so communication robustness and voting behavior can be studied separately before extending the method to multi-task allocation.

## Common experiment settings

- Single task
- Robot counts: `5, 10, 15, ..., 100`
- Trials per configuration: `100`
- Fixed robot costs
- One vote per robot
- Packet loss is applied to the vote -> terminal communication stage
- Reproducible random seeds

### Fixed cost model

For Robot `i` (1-based index):

```text
C_i = 10 + 5(i - 1)
```

Therefore Robot 1 always has the minimum cost.

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

## Experiment 2: retransmission at 30% packet loss

This experiment fixes the packet-loss probability of each individual transmission attempt at:

```text
30%
```

and compares maximum transmission attempts:

```text
1, 2, 3, ..., 10
```

The model uses **stop-on-success retransmission**:

1. A robot sends its vote.
2. If the packet is delivered, transmission stops for that robot.
3. If the packet is lost, the same vote is retried until success or the configured maximum attempt count is reached.
4. The terminal counts at most one vote from each robot, so retransmissions never create duplicate votes.

Each transmission attempt is modeled as an independent Bernoulli packet-loss event. Therefore, if the per-attempt loss probability is `p = 0.30`, the theoretical probability that a vote is still lost after `k` maximum attempts is:

```text
p_effective = 0.30^k
```

Examples:

| Max attempts | Theoretical effective loss |
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
```

The first four figures keep robot count on the x-axis and contain ten curves, one for each maximum transmission-attempt setting from 1 to 10.

`retransmission_effective_loss_rate.png` directly compares the observed effective loss rate with the theoretical `0.30^k` curve.

`retransmission_overhead.png` shows the average number of actual transmissions required per vote when transmission stops immediately after success. This makes it possible to study the reliability-versus-communication-overhead trade-off.

Generated retransmission data:

```text
results/retransmission/data/retransmission_raw_results.csv
results/retransmission/data/retransmission_summary_results.csv
results/retransmission/data/retransmission_by_attempt.csv
```

### Important interpretation

Retransmission improves **communication reliability**, but it does not automatically make the voting rule mathematically optimal.

If packet loss is eliminated, the lossy winner approaches the complete-vote winner, so Winner Preservation Rate should approach 100%. However, if the complete voting process itself chooses a non-minimum-cost robot, retransmitting the same votes cannot correct that decision. This distinction lets the project study communication robustness separately from voting optimality.

## Main metrics

1. **Winner Preservation Rate**  
   Probability that the winner after communication loss is the same as the winner with all votes delivered.

2. **Optimal Win Rate**  
   Probability that the final winner is the minimum-cost robot.

3. **Average Regret**  
   `selected_robot_cost - minimum_cost`.

4. **Tie Rate**  
   Fraction of trials in which the highest received vote count is shared by multiple robots before tie-breaking.

5. **Effective Packet Loss Rate**  
   Fraction of robot votes that remain undelivered after all allowed transmission attempts.

6. **Average Actual Transmissions per Vote**  
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

The demo prints robot costs, voting probabilities, every ballot, whether that ballot was delivered or dropped, full vote counts, received vote counts, and both winners.

## GitHub Actions

The workflow under `.github/workflows/run-experiment.yml` runs both experiments and uploads all generated CSV files and PNG figures as the `voting-mrta-results` workflow artifact.

## Current scope

The current code isolates **vote-message packet loss and retransmission**. Cost-sharing packet loss, dynamic cost, multiple simultaneous tasks, robot capacity constraints, task reassignment, and heterogeneous multi-algorithm voting are intentionally left for later experiments.
