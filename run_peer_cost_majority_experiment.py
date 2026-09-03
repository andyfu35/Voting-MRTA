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
DEFAULT_TRIALS = 200
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
    message arrived, otherwise NaN. Every robot always knows its own local cost.
    """
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 1 or len(costs) < 2:
        raise ValueError("costs must be a 1-D array with at least two robots")
    if not 0.0 <= loss_rate < 1.0:
        raise ValueError("loss_rate must be in [0, 1)")

    n = len(costs)
    delivered = rng.random((n, n)) >= loss_rate
    np.fill_diagonal(delivered, True)
    views = np.where(delivered, costs[None, :], np.nan)
    return views


def choose_local_greedy_votes(cost_views: np.ndarray) -> np.ndarray:
    """Each receiver votes for the minimum-cost candidate visible in its view."""
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


def count_votes(votes: np.ndarray, n: int) -> np.ndarray:
    votes = np.asarray(votes, dtype=int)
    if votes.ndim != 2 or votes.shape[1] != n:
        raise ValueError("votes must have shape (trials, n)")
    if np.any((votes < 0) | (votes >= n)):
        raise ValueError("vote candidate out of range")

    trials = votes.shape[0]
    counts = np.zeros((trials, n), dtype=np.int16)
    rows = np.repeat(np.arange(trials), n)
    np.add.at(counts, (rows, votes.ravel()), 1)
    return counts


def resolve_strict_majority(vote_counts: np.ndarray) -> MajorityResult:
    """Commit only when one candidate receives strictly more than 50% of votes."""
    counts = np.asarray(vote_counts)
    if counts.ndim != 2 or counts.shape[1] < 2:
        raise ValueError("vote_counts must have shape (trials, candidates>=2)")

    n = counts.shape[1]
    required_votes = n // 2 + 1
    winner = np.argmax(counts, axis=1).astype(int)
    top_votes = counts[np.arange(len(counts)), winner]
    committed = top_votes >= required_votes
    resolved_winner = np.where(committed, winner, -1)
    vote_share = top_votes.astype(float) / float(n)
    return MajorityResult(
        committed=committed,
        winner=resolved_winner,
        vote_share=vote_share,
    )


def run_configuration(
    n: int,
    loss_percent: int,
    trials: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    loss_rate = loss_percent / 100.0
    votes = sample_local_greedy_votes(n, loss_rate, trials, rng)
    vote_counts = count_votes(votes, n)
    majority = resolve_strict_majority(vote_counts)

    optimal_robot = 0
    optimal_votes = vote_counts[:, optimal_robot]
    committed_trials = int(np.count_nonzero(majority.committed))
    correct_commits = int(np.count_nonzero(majority.winner == optimal_robot))
    wrong_commits = int(
        np.count_nonzero((majority.winner >= 0) & (majority.winner != optimal_robot))
    )
    no_majority = trials - committed_trials

    expected_optimal_vote_share = (
        1.0 + (n - 1) * (1.0 - loss_rate)
    ) / n

    return {
        "robots": n,
        "packet_loss_percent": loss_percent,
        "trials": trials,
        "required_majority_votes": n // 2 + 1,
        "majority_commit_rate": committed_trials / trials,
        "optimal_commit_rate": correct_commits / trials,
        "wrong_commit_rate": wrong_commits / trials,
        "no_majority_rate": no_majority / trials,
        "conditional_commit_accuracy": (
            correct_commits / committed_trials if committed_trials else np.nan
        ),
        "mean_optimal_vote_share": float(optimal_votes.mean() / n),
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
        n for n in selected_counts if results["robots"].min() <= n <= results["robots"].max()
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for n in selected_counts:
        part = results[results["robots"] == n].sort_values("packet_loss_percent")
        ax.plot(
            part["packet_loss_percent"],
            part["optimal_commit_rate"],
            label=f"N={n}",
        )
    ax.set_xlabel("Directed P2P cost-message packet loss (%)")
    ax.set_ylabel("Optimal strict-majority commit rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "optimal_commit_selected_robot_counts.png", dpi=180)
    plt.close(fig)


def save_outputs(results: pd.DataFrame) -> None:
    ensure_output_dirs()
    results.to_csv(DATA_DIR / "peer_cost_majority_results.csv", index=False)
    save_heatmap(
        results,
        "optimal_commit_rate",
        "Optimal strict-majority commit rate",
        "optimal_commit_rate_heatmap.png",
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
        "Wrong strict-majority commit rate",
        "wrong_commit_rate_heatmap.png",
    )
    save_selected_robot_curves(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep robot count and directed P2P cost-message packet loss for "
            "local greedy voting with a strict >50% commit rule."
        )
    )
    parser.add_argument("--robot-min", type=int, default=ROBOT_MIN)
    parser.add_argument("--robot-max", type=int, default=ROBOT_MAX)
    parser.add_argument("--loss-min", type=int, default=LOSS_MIN_PERCENT)
    parser.add_argument("--loss-max", type=int, default=LOSS_MAX_PERCENT)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
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

    print("\nSaved:")
    print(DATA_DIR / "peer_cost_majority_results.csv")
    print(FIGURE_DIR)
    print("\nSanity rows:")
    sample = results[
        results["packet_loss_percent"].isin([0, 25, 50, 75, 99])
        & results["robots"].isin([5, 30, 100])
    ]
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
