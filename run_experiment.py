from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from voting_mrta import apply_vote_loss, generate_costs, generate_full_round


ROBOT_COUNTS = list(range(5, 101, 5))
TRIALS = 100
LOSS_RATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

COST_START = 10.0
COST_STEP = 5.0
ALPHA = 1.0
RANDOM_SEED = 42

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "results" / "data"
FIGURE_DIR = ROOT / "results" / "figures"


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_figures() -> None:
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    records: list[dict[str, object]] = []

    total_configurations = len(ROBOT_COUNTS) * len(LOSS_RATES)
    print(f"Running {total_configurations} configurations, {TRIALS} trials each...")

    for n in ROBOT_COUNTS:
        costs = generate_costs(n, cost_start=COST_START, cost_step=COST_STEP)
        optimal_robot = int(np.argmin(costs))
        optimal_cost = float(costs[optimal_robot])

        for trial in range(TRIALS):
            round_data = generate_full_round(
                n,
                rng,
                cost_start=COST_START,
                cost_step=COST_STEP,
                alpha=ALPHA,
            )

            for loss_rate in LOSS_RATES:
                loss_result = apply_vote_loss(round_data, loss_rate)
                winner = loss_result.winner
                no_decision = winner is None

                if no_decision:
                    lossy_is_optimal = False
                    winner_preserved = False
                    regret = np.nan
                    winner_cost = np.nan
                else:
                    winner_cost = float(costs[winner])
                    lossy_is_optimal = winner == optimal_robot
                    winner_preserved = winner == round_data.full_winner
                    regret = winner_cost - optimal_cost

                delivered_votes = int(loss_result.delivered.sum())
                dropped_votes = n - delivered_votes

                records.append(
                    {
                        "robots": n,
                        "trial": trial + 1,
                        "loss_rate": loss_rate,
                        "alpha": ALPHA,
                        "optimal_robot": optimal_robot + 1,
                        "optimal_cost": optimal_cost,
                        "full_winner": round_data.full_winner + 1,
                        "lossy_winner": np.nan if winner is None else winner + 1,
                        "winner_cost": winner_cost,
                        "full_tie": round_data.full_tie,
                        "lossy_tie": loss_result.tie,
                        "winner_preserved": winner_preserved,
                        "lossy_is_optimal": lossy_is_optimal,
                        "delivered_votes": delivered_votes,
                        "dropped_votes": dropped_votes,
                        "observed_loss_rate": dropped_votes / n,
                        "no_decision": no_decision,
                        "regret": regret,
                    }
                )

    raw = pd.DataFrame.from_records(records)

    summary = (
        raw.groupby(["robots", "loss_rate"], as_index=False)
        .agg(
            winner_preservation_rate=("winner_preserved", "mean"),
            optimal_win_rate=("lossy_is_optimal", "mean"),
            average_regret=("regret", "mean"),
            tie_rate=("lossy_tie", "mean"),
            no_decision_rate=("no_decision", "mean"),
            average_observed_loss_rate=("observed_loss_rate", "mean"),
            average_delivered_votes=("delivered_votes", "mean"),
        )
        .sort_values(["loss_rate", "robots"])
        .reset_index(drop=True)
    )

    return raw, summary


def plot_metric(
    summary: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    percent_y: bool = False,
    reference_lines: bool = False,
) -> None:
    plt.figure(figsize=(10, 6))

    for loss_rate in LOSS_RATES:
        data = summary[summary["loss_rate"] == loss_rate]
        plt.plot(
            data["robots"],
            data[metric],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=f"Loss = {loss_rate:.0%}",
        )

        if reference_lines:
            plt.axhline(
                loss_rate,
                linewidth=0.8,
                linestyle="--",
                alpha=0.25,
            )

    plt.xlabel("Number of Robots")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(ROBOT_COUNTS)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)

    if percent_y:
        from matplotlib.ticker import PercentFormatter

        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.ylim(0.0, 1.02)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def generate_figures(summary: pd.DataFrame) -> None:
    plot_metric(
        summary,
        metric="winner_preservation_rate",
        ylabel="Winner Preservation Rate",
        title="Winner Preservation vs. Robot Team Size",
        filename="winner_preservation_rate.png",
        percent_y=True,
    )

    plot_metric(
        summary,
        metric="optimal_win_rate",
        ylabel="Optimal Win Rate",
        title="Minimum-Cost Robot Selection vs. Robot Team Size",
        filename="optimal_win_rate.png",
        percent_y=True,
    )

    plot_metric(
        summary,
        metric="average_regret",
        ylabel="Average Regret (Cost Units)",
        title="Average Cost Regret vs. Robot Team Size",
        filename="average_regret.png",
    )

    plot_metric(
        summary,
        metric="tie_rate",
        ylabel="Tie Rate",
        title="Voting Tie Rate vs. Robot Team Size",
        filename="tie_rate.png",
        percent_y=True,
    )

    plot_metric(
        summary,
        metric="average_observed_loss_rate",
        ylabel="Observed Packet Loss Rate",
        title="Packet Loss Simulation Validation",
        filename="packet_loss_validation.png",
        percent_y=True,
        reference_lines=True,
    )


def print_summary(summary: pd.DataFrame) -> None:
    display_columns = [
        "robots",
        "loss_rate",
        "winner_preservation_rate",
        "optimal_win_rate",
        "average_regret",
        "tie_rate",
        "average_observed_loss_rate",
    ]
    print("\nExperiment summary (first 20 rows):")
    print(summary[display_columns].head(20).to_string(index=False))


def main() -> None:
    ensure_output_dirs()
    clear_old_figures()
    raw, summary = run_experiment()

    raw_path = DATA_DIR / "raw_results.csv"
    summary_path = DATA_DIR / "summary_results.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    generate_figures(summary)
    print_summary(summary)

    print("\nGenerated files:")
    print(f"  {raw_path.relative_to(ROOT)}")
    print(f"  {summary_path.relative_to(ROOT)}")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
