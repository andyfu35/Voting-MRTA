from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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


def clean_figure_output() -> None:
    """Remove old PNG figures so each run leaves only the current six reports."""
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


def configure_x_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Number of Robots")
    ax.set_xticks(ROBOT_COUNTS)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.grid(True, alpha=0.25)


def plot_rate_metric(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    ax.plot(
        data["robots"],
        data[metric],
        marker="o",
        linewidth=2.0,
        markersize=4.5,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    configure_x_axis(ax)


def plot_regret_metric(ax: plt.Axes, data: pd.DataFrame) -> None:
    ax.plot(
        data["robots"],
        data["average_regret"],
        marker="o",
        linewidth=2.0,
        markersize=4.5,
    )
    ax.set_title("Average Regret")
    ax.set_ylabel("Cost Units")
    ax.set_ylim(bottom=0.0)
    configure_x_axis(ax)


def build_summary_text(data: pd.DataFrame, loss_rate: float) -> str:
    observed_loss = float(data["average_observed_loss_rate"].mean())
    mean_wpr = float(data["winner_preservation_rate"].mean())
    mean_optimal = float(data["optimal_win_rate"].mean())
    mean_regret = float(data["average_regret"].mean())
    mean_tie = float(data["tie_rate"].mean())
    mean_no_decision = float(data["no_decision_rate"].mean())

    return (
        f"Configured packet loss: {loss_rate:.0%}    |    "
        f"Observed mean loss: {observed_loss:.2%}    |    "
        f"Trials per team size: {TRIALS}    |    "
        f"Robots: {ROBOT_COUNTS[0]}-{ROBOT_COUNTS[-1]} (step 5)\n"
        f"Single task    |    Cost: C_i = {COST_START:g} + {COST_STEP:g}(i-1)    |    "
        f"Voting weight: w_i = (1/C_i)^alpha    |    alpha = {ALPHA:g}    |    seed = {RANDOM_SEED}\n"
        f"Mean across all team sizes -> "
        f"Winner preservation: {mean_wpr:.1%}    |    "
        f"Optimal win: {mean_optimal:.1%}    |    "
        f"Regret: {mean_regret:.2f}    |    "
        f"Tie: {mean_tie:.1%}    |    "
        f"No decision: {mean_no_decision:.1%}"
    )


def generate_loss_summary_figure(summary: pd.DataFrame, loss_rate: float) -> Path:
    data = (
        summary[summary["loss_rate"] == loss_rate]
        .sort_values("robots")
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))

    plot_rate_metric(
        axes[0, 0],
        data,
        metric="winner_preservation_rate",
        title="Winner Preservation Rate",
        ylabel="Preservation Rate",
    )

    plot_rate_metric(
        axes[0, 1],
        data,
        metric="optimal_win_rate",
        title="Optimal Win Rate",
        ylabel="Optimal Selection Rate",
    )

    plot_regret_metric(axes[1, 0], data)

    plot_rate_metric(
        axes[1, 1],
        data,
        metric="tie_rate",
        title="Tie Rate",
        ylabel="Tie Rate",
    )

    fig.suptitle(
        f"Voting-MRTA Performance under {loss_rate:.0%} Packet Loss",
        fontsize=17,
        y=0.985,
    )

    summary_text = build_summary_text(data, loss_rate)
    fig.text(
        0.5,
        0.018,
        summary_text,
        ha="center",
        va="bottom",
        fontsize=9.5,
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.95,
        },
    )

    fig.tight_layout(rect=[0.025, 0.13, 0.975, 0.95], h_pad=2.0, w_pad=1.6)

    loss_percent = int(round(loss_rate * 100))
    output_path = FIGURE_DIR / f"loss_{loss_percent:02d}_summary.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output_path


def generate_figures(summary: pd.DataFrame) -> list[Path]:
    clean_figure_output()
    return [
        generate_loss_summary_figure(summary, loss_rate)
        for loss_rate in LOSS_RATES
    ]


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
    raw, summary = run_experiment()

    raw_path = DATA_DIR / "raw_results.csv"
    summary_path = DATA_DIR / "summary_results.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    figure_paths = generate_figures(summary)
    print_summary(summary)

    print("\nGenerated files:")
    print(f"  {raw_path.relative_to(ROOT)}")
    print(f"  {summary_path.relative_to(ROOT)}")
    for path in figure_paths:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
