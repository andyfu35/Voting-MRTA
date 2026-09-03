from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


RANDOM_SEED = 20260903
ROBOT_MIN = 5
ROBOT_MAX = 100
LOSS_MIN_PERCENT = 0
LOSS_MAX_PERCENT = 99
DEFAULT_TRIALS = 100
COST_START = 10.0
COST_STEP = 5.0

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "peer_cost_majority"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


@dataclass(frozen=True)
class MajorityResult:
    committed: np.ndarray
    winner: np.ndarray
    vote_share: np.ndarray
    votes_in_window: np.ndarray
    required_votes: np.ndarray


def validate_experiment_config(
    robot_min: int,
    robot_max: int,
    loss_min_percent: int,
    loss_max_percent: int,
    trials: int,
) -> None:
    if robot_min < 2:
        raise ValueError("robot_min must be at least 2")
    if robot_max < robot_min:
        raise ValueError("robot_max must be >= robot_min")
    if not 0 <= loss_min_percent <= 99:
        raise ValueError("loss_min_percent must be between 0 and 99")
    if not loss_min_percent <= loss_max_percent <= 99:
        raise ValueError("loss_max_percent must be between loss_min_percent and 99")
    if trials <= 0:
        raise ValueError("trials must be positive")


def generate_costs(n: int) -> np.ndarray:
    """Deterministic strictly increasing costs; Robot 0 is the global optimum."""
    if n <= 0:
        raise ValueError("n must be positive")
    return COST_START + COST_STEP * np.arange(n, dtype=float)


def simulate_cost_exchange_round(
    costs: np.ndarray,
    loss_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build one receiver-specific cost view under directed independent P2P loss.

    ``views[receiver, sender]`` is sender's cost if the directed sender->receiver
    message arrived before the cost-exchange cutoff, otherwise NaN. Every robot
    always knows its own local cost.
    """
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 1 or len(costs) < 2:
        raise ValueError("costs must be a 1-D array with at least two robots")
    if not 0.0 <= loss_rate < 1.0:
        raise ValueError("loss_rate must be in [0, 1)")

    n = len(costs)
    delivered = rng.random((n, n)) >= loss_rate
    np.fill_diagonal(delivered, True)
    return np.where(delivered, costs[None, :], np.nan)


def choose_local_greedy_votes(cost_views: np.ndarray) -> np.ndarray:
    """Each receiver votes for the minimum-cost candidate visible at cutoff."""
    views = np.asarray(cost_views, dtype=float)
    if views.ndim != 2 or views.shape[0] != views.shape[1]:
        raise ValueError("cost_views must be a square receiver x sender matrix")
    if np.any(np.all(np.isnan(views), axis=1)):
        raise ValueError("every receiver must know at least one candidate cost")
    return np.nanargmin(views, axis=1).astype(int)


def sample_local_greedy_votes(
    n: int,
    loss_rate: float,
    trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample local-greedy votes exactly without materializing all N^2 links.

    Costs are strictly increasing by robot index. For receiver r, only delivery
    of candidates 0..r-1 can beat its always-known self cost. The first delivered
    lower-cost candidate is therefore the greedy choice. Independent directed
    packet loss makes that first success geometrically distributed.

    This is decision-equivalent to simulating every sender->receiver link while
    making the full 5..100 x 0..99 sweep substantially cheaper to run.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if not 0.0 <= loss_rate < 1.0:
        raise ValueError("loss_rate must be in [0, 1)")
    if trials <= 0:
        raise ValueError("trials must be positive")

    delivery_probability = 1.0 - loss_rate
    first_success_index = rng.geometric(
        delivery_probability,
        size=(trials, n),
    ) - 1
    receiver_index = np.arange(n, dtype=int)[None, :]
    return np.minimum(first_success_index, receiver_index).astype(int)


def collect_votes_within_window(
    votes: np.ndarray,
    n: int,
    vote_in_window: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Count only votes received before the fixed vote-window cutoff.

    Vote-message delay/loss is intentionally disabled in this experiment, so
    every locally generated vote is currently in-window. ``vote_in_window`` is
    retained as the explicit timing boundary for later experiments.
    """
    votes = np.asarray(votes, dtype=int)
    if votes.ndim != 2 or votes.shape[1] != n:
        raise ValueError("votes must have shape (trials, n)")
    if np.any((votes < 0) | (votes >= n)):
        raise ValueError("vote candidate out of range")

    if vote_in_window is None:
        in_window = np.ones_like(votes, dtype=bool)
    else:
        in_window = np.asarray(vote_in_window, dtype=bool)
        if in_window.shape != votes.shape:
            raise ValueError("vote_in_window must have the same shape as votes")

    trial_count = votes.shape[0]
    counts = np.zeros((trial_count, n), dtype=np.int16)
    rows, voter_columns = np.nonzero(in_window)
    np.add.at(counts, (rows, votes[rows, voter_columns]), 1)
    votes_in_window = in_window.sum(axis=1).astype(int)
    return counts, votes_in_window


def resolve_strict_majority(
    vote_counts: np.ndarray,
    votes_in_window: np.ndarray,
) -> MajorityResult:
    """Commit when one candidate has >50% of votes received before cutoff."""
    counts = np.asarray(vote_counts)
    received = np.asarray(votes_in_window, dtype=int)
    if counts.ndim != 2 or counts.shape[1] < 2:
        raise ValueError("vote_counts must have shape (trials, candidates>=2)")
    if received.shape != (counts.shape[0],):
        raise ValueError("votes_in_window must have one value per trial")
    if np.any(received < 0):
        raise ValueError("votes_in_window cannot be negative")
    if np.any(counts.sum(axis=1) != received):
        raise ValueError("vote_counts must sum to votes_in_window for every trial")

    required_votes = received // 2 + 1
    winner = np.argmax(counts, axis=1).astype(int)
    top_votes = counts[np.arange(len(counts)), winner]
    committed = (received > 0) & (top_votes >= required_votes)
    resolved_winner = np.where(committed, winner, -1)
    vote_share = np.divide(
        top_votes.astype(float),
        received,
        out=np.zeros_like(top_votes, dtype=float),
        where=received > 0,
    )
    return MajorityResult(
        committed=committed,
        winner=resolved_winner,
        vote_share=vote_share,
        votes_in_window=received,
        required_votes=required_votes,
    )


def summarize_execution_outcome(
    vote_counts: np.ndarray,
    majority: MajorityResult,
    optimal_robot: int,
) -> dict[str, float | int]:
    """Summarize whether the globally optimal robot actually executes the task."""
    trials = vote_counts.shape[0]
    optimal_votes = vote_counts[:, optimal_robot]
    successful_optimal_executions = int(
        np.count_nonzero(majority.winner == optimal_robot)
    )
    wrong_executions = int(
        np.count_nonzero(
            (majority.winner >= 0) & (majority.winner != optimal_robot)
        )
    )
    no_execution = trials - successful_optimal_executions - wrong_executions

    return {
        "successful_optimal_executions": successful_optimal_executions,
        "failed_optimal_executions": trials - successful_optimal_executions,
        "optimal_execution_success_rate": successful_optimal_executions / trials,
        "optimal_execution_success_percent": 100.0 * successful_optimal_executions / trials,
        "wrong_execution_count": wrong_executions,
        "no_execution_count": no_execution,
        "mean_optimal_votes": float(optimal_votes.mean()),
        "min_optimal_votes": int(optimal_votes.min()),
        "max_optimal_votes": int(optimal_votes.max()),
    }


def run_configuration(
    n: int,
    loss_percent: int,
    trials: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    loss_rate = loss_percent / 100.0
    votes = sample_local_greedy_votes(n, loss_rate, trials, rng)

    # Current controlled experiment: every robot finishes its local decision and
    # its vote is counted before the shared vote-window cutoff. Only cost-message
    # exchange is lossy in this stage of the study.
    vote_counts, votes_in_window = collect_votes_within_window(votes, n)
    majority = resolve_strict_majority(vote_counts, votes_in_window)

    optimal_robot = 0
    execution = summarize_execution_outcome(vote_counts, majority, optimal_robot)

    committed_trials = int(np.count_nonzero(majority.committed))
    wrong_commits = int(execution["wrong_execution_count"])
    no_majority = trials - committed_trials
    expected_optimal_vote_share = (
        1.0 + (n - 1) * (1.0 - loss_rate)
    ) / n

    return {
        "robots": n,
        "packet_loss_percent": loss_percent,
        "trials_per_point": trials,
        "mean_votes_in_window": float(majority.votes_in_window.mean()),
        "vote_window_participation_rate": float(majority.votes_in_window.mean() / n),
        "mean_required_majority_votes": float(majority.required_votes.mean()),
        **execution,
        "majority_commit_rate": committed_trials / trials,
        "wrong_commit_rate": wrong_commits / trials,
        "no_majority_rate": no_majority / trials,
        "conditional_commit_accuracy": (
            int(execution["successful_optimal_executions"]) / committed_trials
            if committed_trials
            else np.nan
        ),
        "mean_optimal_vote_share": float(execution["mean_optimal_votes"] / n),
        "expected_optimal_vote_share": float(expected_optimal_vote_share),
        "mean_winning_vote_share": float(majority.vote_share.mean()),
    }


def run_experiment(
    *,
    robot_min: int = ROBOT_MIN,
    robot_max: int = ROBOT_MAX,
    loss_min_percent: int = LOSS_MIN_PERCENT,
    loss_max_percent: int = LOSS_MAX_PERCENT,
    trials: int = DEFAULT_TRIALS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    validate_experiment_config(
        robot_min,
        robot_max,
        loss_min_percent,
        loss_max_percent,
        trials,
    )
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int]] = []

    for n in range(robot_min, robot_max + 1):
        for loss_percent in range(loss_min_percent, loss_max_percent + 1):
            records.append(run_configuration(n, loss_percent, trials, rng))
        print(f"robots={n:3d}/{robot_max} complete")

    return pd.DataFrame.from_records(records)


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_heatmap(
    results: pd.DataFrame,
    metric: str,
    title: str,
    filename: str,
) -> None:
    pivot = results.pivot(
        index="robots",
        columns="packet_loss_percent",
        values=metric,
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(
        pivot.to_numpy(),
        origin="lower",
        aspect="auto",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max(),
        ],
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Directed P2P cost-message packet loss (%)")
    ax.set_ylabel("Robot count")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close(fig)


def save_selected_robot_curves(results: pd.DataFrame) -> None:
    selected_counts = [5, 10, 20, 30, 50, 75, 100]
    selected_counts = [
        n
        for n in selected_counts
        if results["robots"].min() <= n <= results["robots"].max()
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for n in selected_counts:
        part = results[results["robots"] == n].sort_values("packet_loss_percent")
        ax.plot(
            part["packet_loss_percent"],
            part["optimal_execution_success_rate"],
            label=f"N={n}",
        )
    ax.set_xlabel("Directed P2P cost-message packet loss (%)")
    ax.set_ylabel("Optimal robot execution success rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "optimal_execution_selected_robot_counts.png", dpi=180)
    plt.close(fig)


def save_outputs(results: pd.DataFrame) -> None:
    ensure_output_dirs()
    results.to_csv(DATA_DIR / "peer_cost_majority_results.csv", index=False)
    save_heatmap(
        results,
        "optimal_execution_success_rate",
        "Optimal robot execution success rate (100 trials per point)",
        "optimal_execution_success_rate_heatmap.png",
    )
    save_heatmap(
        results,
        "no_majority_rate",
        "No-majority rate",
        "no_majority_rate_heatmap.png",
    )
    save_heatmap(
        results,
        "wrong_commit_rate",
        "Wrong execution rate",
        "wrong_execution_rate_heatmap.png",
    )
    save_selected_robot_curves(results)


def printable_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Return a compact terminal summary that can be pasted back for analysis."""
    preferred_robots = [5, 10, 20, 30, 50, 75, 100]
    preferred_losses = [0, 10, 20, 25, 30, 40, 45, 50, 55, 60, 70, 75, 80, 90, 99]
    robot_values = [
        n
        for n in preferred_robots
        if results["robots"].min() <= n <= results["robots"].max()
    ]
    loss_values = [
        p
        for p in preferred_losses
        if results["packet_loss_percent"].min()
        <= p
        <= results["packet_loss_percent"].max()
    ]
    summary = results[
        results["robots"].isin(robot_values)
        & results["packet_loss_percent"].isin(loss_values)
    ].copy()
    columns = [
        "robots",
        "packet_loss_percent",
        "trials_per_point",
        "successful_optimal_executions",
        "optimal_execution_success_percent",
        "mean_optimal_votes",
        "mean_required_majority_votes",
        "wrong_execution_count",
        "no_execution_count",
    ]
    return summary[columns].sort_values(["robots", "packet_loss_percent"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep robot count and directed P2P cost-message packet loss for "
            "local greedy voting. Each (robots, loss) point is summarized over "
            "100 trials by default."
        )
    )
    parser.add_argument("--robot-min", type=int, default=ROBOT_MIN)
    parser.add_argument("--robot-max", type=int, default=ROBOT_MAX)
    parser.add_argument("--loss-min", type=int, default=LOSS_MIN_PERCENT)
    parser.add_argument("--loss-max", type=int, default=LOSS_MAX_PERCENT)
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Trials summarized into each data point (default: 100).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(
        robot_min=args.robot_min,
        robot_max=args.robot_max,
        loss_min_percent=args.loss_min,
        loss_max_percent=args.loss_max,
        trials=args.trials,
        seed=args.seed,
    )
    save_outputs(results)

    print("\nSaved full 5..100 x 0..99 summary:")
    print(DATA_DIR / "peer_cost_majority_results.csv")
    print(FIGURE_DIR)
    print("\nPasteable execution summary:")
    print(printable_summary(results).to_string(index=False))


if __name__ == "__main__":
    main()
