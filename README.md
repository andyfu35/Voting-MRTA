# Voting-MRTA

A simulation project for a voting-based Multi-Robot Task Allocation (MRTA) method under communication packet loss.

The first experiment intentionally uses a simple, controlled setting so the effect of robot-team size and vote-message loss can be observed clearly.

## Experiment design

- Single task
- Robot counts: `5, 10, 15, ..., 100`
- Trials per configuration: `100`
- Vote packet-loss rates: `5%, 10%, 15%, 20%, 25%, 30%`
- Fixed robot costs
- One vote per robot
- Packet loss is applied only to the vote -> terminal communication stage in this first version
- Reproducible random seed: `42`

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

The default experiment uses `alpha = 1.0`.

### Communication-loss model

Every robot first casts one vote. The same complete set of ballots is then reused for all packet-loss rates in that trial. This makes comparisons fair: the only difference between loss settings is which vote messages reach the terminal.

For a configured packet-loss probability `q`, each vote is independently delivered with probability `1 - q`.

## Main metrics

1. **Winner Preservation Rate**  
   Probability that the winner after packet loss is the same as the winner with all votes delivered.

2. **Optimal Win Rate**  
   Probability that the final winner is the minimum-cost robot.

3. **Average Regret**  
   `selected_robot_cost - minimum_cost`.

4. **Tie Rate**  
   Fraction of trials in which the highest received vote count is shared by multiple robots before tie-breaking.

5. **Packet Loss Validation**  
   The observed dropped-vote fraction, used to verify the simulator matches the configured loss rate.

## Generated figures

Running the experiment generates five main figures:

```text
results/figures/winner_preservation_rate.png
results/figures/optimal_win_rate.png
results/figures/average_regret.png
results/figures/tie_rate.png
results/figures/packet_loss_validation.png
```

Each figure uses robot count on the x-axis and contains **six curves**, one for each communication packet-loss rate:

- 5%
- 10%
- 15%
- 20%
- 25%
- 30%

This layout makes it easy to compare the effect of packet loss directly for the same performance metric.

The experiment also generates:

```text
results/data/raw_results.csv
results/data/summary_results.csv
```

Old PNG files in `results/figures/` are removed automatically before the new figures are generated.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Inspect one voting round

To see the voting process vote-by-vote:

```bash
python demo_single_vote.py --robots 20 --loss 0.20
```

The demo prints robot costs, voting probabilities, every ballot, whether that ballot was delivered or dropped, full vote counts, received vote counts, and both winners.

## GitHub Actions

A workflow is included under `.github/workflows/run-experiment.yml`. It runs the simulation and uploads the generated CSV files and PNG figures as a workflow artifact.

## Current scope

This first version isolates **vote-message packet loss**. Cost-sharing packet loss, dynamic cost, multiple simultaneous tasks, robot capacity constraints, and task reassignment are intentionally left for later experiments so the first results remain easy to interpret.
