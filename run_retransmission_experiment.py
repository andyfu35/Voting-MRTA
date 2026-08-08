from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from voting_mrta import (
    apply_vote_retransmission,
    generate_costs,
    generate_full_round,
)


ROBOT_COUNTS = list(range(5, 101, 5))
MAX_ATTEMPT_VALUES = list(range(1, 11))
TRIALS = 100
LOSS_RATE = 0.30

COST_START = 10.0
COST_STEP = 5.0
ALPHA = 1.0
RANDOM_SEED = 4242

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "retransmission"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_outputs() -> None:
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    records: list[dict[str, object]] = []
    max_attempts_available = max(MAX_ATTEMPT_VALUES)

    print(
        "Running retransmission experiment: "
        f"loss={LOSS_RATE:.0%}, robots={ROBOT_COUNTS[0]}-{ROBOT_COUNTS[-1]}, "
        f"max attempts={MAX_ATTEMPT_VALUES[0]}-{MAX_ATTEMPT_VALUES[-1]}, "
        f"trials={TRIALS}"
    )

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

            # One shared attempt matrix makes max_attempts=1..10 nested versions
            # of the exact same communication realization.
            attempt_random = rng.random((max_attempts_available, n))

            for max_attempts in MAX_ATTEMPT_VALUES:
                result = apply_vote_retransmission(
                    round_data,
                    attempt_random,
                    loss_rate=LOSS_RATE,
                    max_attempts=max_attempts,
                )

                winner = result.winner
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

                delivered_votes = int(result.delivered.sum())
                dropped_votes = n - delivered_votes
                total_transmissions = int(result.attempts_used.sum())

                records.append(
                    {
                        "robots": n,
                        "trial": trial + 1,
                        "loss_rate": LOSS_RATE,
                        "max_attempts": max_attempts,
                        "theoretical_effective_loss_rate": LOSS_RATE**max_attempts,
                        "alpha": ALPHA,
                        "optimal_robot": optimal_robot + 1,
                        "optimal_cost": optimal_cost,
                        "full_winner": round_data.full_winner + 1,
                        "lossy_winner": np.nan if winner is None else winner + 1,
                        "winner_cost": winner_cost,
                        "winner_preserved": winner_preserved,
                        "lossy_is_optimal": lossy_is_optimal,
                        "lossy_tie": result.tie,
                        "no_decision": no_decision,
                        "regret": regret,
                        "delivered_votes": delivered_votes,
                        "dropped_votes": dropped_votes,
                        "observed_effective_loss_rate": dropped_votes / n,
                        "total_transmissions": total_transmissions,
                        "average_attempts_per_vote": total_transmissions / n,
                    }
                )

    raw = pd.DataFrame.from_records(records)

    summary = (
        raw.groupby(["robots", "max_attempts"], as_index=False)
        .agg(
            winner_preservation_rate=("winner_preserved", "mean"),
            optimal_win_rate=("lossy_is_optimal", "mean"),
            average_regret=("regret", "mean"),
            tie_rate=("lossy_tie", "mean"),
            no_decision_rate=("no_decision", "mean"),
            observed_effective_loss_rate=("observed_effective_loss_rate", "mean"),
            average_attempts_per_vote=("average_attempts_per_vote", "mean"),
            average_total_transmissions=("total_transmissions", "mean"),
        )
        .sort_values(["max_attempts", "robots"])
        .reset_index(drop=True)
    )
    summary["theoretical_effective_loss_rate"] = np.power(
        LOSS_RATE,
        summary["max_attempts"],
    )

    by_attempt = (
        summary.groupby("max_attempts", as_index=False)
        .agg(
            winner_preservation_rate=("winner_preservation_rate", "mean"),
            optimal_win_rate=("optimal_win_rate", "mean"),
            average_regret=("average_regret", "mean"),
            tie_rate=("tie_rate", "mean"),
            no_decision_rate=("no_decision_rate", "mean"),
            observed_effective_loss_rate=("observed_effective_loss_rate", "mean"),
            average_attempts_per_vote=("average_attempts_per_vote", "mean"),
        )
        .sort_values("max_attempts")
        .reset_index(drop=True)
    )
    by_attempt["theoretical_effective_loss_rate"] = np.power(
        LOSS_RATE,
        by_attempt["max_attempts"],
    )

    return raw, summary, by_attempt


def plot_robot_scale_metric(
    summary: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    percent_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for max_attempts in MAX_ATTEMPT_VALUES:
        data = summary[summary["max_attempts"] == max_attempts]
        ax.plot(
            data["robots"],
            data[metric],
            marker="o",
            linewidth=1.5,
            markersize=3.5,
            label=f"Max attempts = {max_attempts}",
        )

    ax.set_xlabel("Number of Robots")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} at {LOSS_RATE:.0%} Packet Loss")
    ax.set_xticks(ROBOT_COUNTS)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    if percent_y:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0.0, 1.02)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_effective_loss(by_attempt: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        by_attempt["max_attempts"],
        by_attempt["observed_effective_loss_rate"],
        marker="o",
        linewidth=2.0,
        label="Observed effective loss",
    )
    ax.plot(
        by_attempt["max_attempts"],
        by_attempt["theoretical_effective_loss_rate"],
        marker="x",
        linestyle="--",
        linewidth=1.7,
        label=r"Theory: $0.3^k$",
    )
    ax.set_xlabel("Maximum Transmission Attempts")
    ax.set_ylabel("Effective Vote Loss Rate")
    ax.set_title("Retransmission Effect on Effective Vote Loss (30% per Attempt)")
    ax.set_xticks(MAX_ATTEMPT_VALUES)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "retransmission_effective_loss_rate.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_transmission_overhead(by_attempt: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        by_attempt["max_attempts"],
        by_attempt["average_attempts_per_vote"],
        marker="o",
        linewidth=2.0,
    )
    ax.set_xlabel("Maximum Transmission Attempts")
    ax.set_ylabel("Average Actual Transmissions per Vote")
    ax.set_title("Communication Overhead of Stop-on-Success Retransmission")
    ax.set_xticks(MAX_ATTEMPT_VALUES)
    ax.set_ylim(bottom=1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "retransmission_overhead.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def generate_figures(summary: pd.DataFrame, by_attempt: pd.DataFrame) -> None:
    plot_robot_scale_metric(
        summary,
        metric="winner_preservation_rate",
        ylabel="Winner Preservation Rate",
        title="Winner Preservation vs. Robot Team Size",
        filename="retransmission_winner_preservation_rate.png",
        percent_y=True,
    )
    plot_robot_scale_metric(
        summary,
        metric="optimal_win_rate",
        ylabel="Optimal Win Rate",
        title="Minimum-Cost Robot Selection vs. Robot Team Size",
        filename="retransmission_optimal_win_rate.png",
        percent_y=True,
    )
    plot_robot_scale_metric(
        summary,
        metric="average_regret",
        ylabel="Average Regret (Cost Units)",
        title="Average Cost Regret vs. Robot Team Size",
        filename="retransmission_average_regret.png",
    )
    plot_robot_scale_metric(
        summary,
        metric="tie_rate",
        ylabel="Tie Rate",
        title="Voting Tie Rate vs. Robot Team Size",
        filename="retransmission_tie_rate.png",
        percent_y=True,
    )
    plot_effective_loss(by_attempt)
    plot_transmission_overhead(by_attempt)


def print_summary(by_attempt: pd.DataFrame) -> None:
    columns = [
        "max_attempts",
        "theoretical_effective_loss_rate",
        "observed_effective_loss_rate",
        "winner_preservation_rate",
        "optimal_win_rate",
        "average_regret",
        "tie_rate",
        "average_attempts_per_vote",
    ]
    print("\nRetransmission summary averaged across robot team sizes:")
    print(by_attempt[columns].to_string(index=False))


def main() -> None:
    ensure_output_dirs()
    clear_old_outputs()
    raw, summary, by_attempt = run_experiment()

    raw_path = DATA_DIR / "retransmission_raw_results.csv"
    summary_path = DATA_DIR / "retransmission_summary_results.csv"
    by_attempt_path = DATA_DIR / "retransmission_by_attempt.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_attempt.to_csv(by_attempt_path, index=False)

    generate_figures(summary, by_attempt)
    print_summary(by_attempt)

    print("\nGenerated retransmission files:")
    for path in [raw_path, summary_path, by_attempt_path]:
        print(f"  {path.relative_to(ROOT)}")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
