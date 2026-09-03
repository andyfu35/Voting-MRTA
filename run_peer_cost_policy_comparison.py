from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from run_peer_cost_majority_experiment import (
    collect_votes_within_window,
    generate_costs,
    resolve_strict_majority,
    summarize_execution_outcome,
)

RANDOM_SEED = 20260903
DEFAULT_ROBOT_COUNTS = (5, 10, 20, 30, 50, 75, 100)
LOSS_MIN_PERCENT = 0
LOSS_MAX_PERCENT = 99
DEFAULT_TRIALS = 100

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "peer_cost_policy_comparison"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


@dataclass(frozen=True)
class PolicySpec:
    key: str
    label: str
    family: str
    parameter: float | None = None


POLICIES = (
    PolicySpec("greedy", "Greedy", "greedy"),
    PolicySpec("inverse_a1", "Inverse a=1", "inverse", 1.0),
    PolicySpec("inverse_a2", "Inverse a=2", "inverse", 2.0),
    PolicySpec("inverse_a3", "Inverse a=3", "inverse", 3.0),
    PolicySpec("softmax_b1", "Softmax b=1", "softmax", 1.0),
    PolicySpec("softmax_b2", "Softmax b=2", "softmax", 2.0),
    PolicySpec("softmax_b5", "Softmax b=5", "softmax", 5.0),
    PolicySpec("rank_g1", "Rank g=1", "rank", 1.0),
    PolicySpec("rank_g2", "Rank g=2", "rank", 2.0),
    PolicySpec("rank_g3", "Rank g=3", "rank", 3.0),
)

REPORT_LOSSES = (0, 10, 20, 25, 30, 40, 45, 50, 55, 60, 70, 75, 80, 90, 99)


def validate_experiment_config(
    robot_counts: tuple[int, ...],
    loss_min_percent: int,
    loss_max_percent: int,
    trials: int,
) -> None:
    if not robot_counts:
        raise ValueError("robot_counts cannot be empty")
    if any(n < 2 for n in robot_counts):
        raise ValueError("every robot count must be at least 2")
    if not 0 <= loss_min_percent <= 99:
        raise ValueError("loss_min_percent must be between 0 and 99")
    if not loss_min_percent <= loss_max_percent <= 99:
        raise ValueError("loss_max_percent must be between loss_min_percent and 99")
    if trials <= 0:
        raise ValueError("trials must be positive")


def sample_paired_cost_visibility(
    n: int,
    loss_rate: float,
    trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample receiver-specific directed P2P cost visibility for paired policies."""
    if n < 2:
        raise ValueError("n must be at least 2")
    if not 0.0 <= loss_rate < 1.0:
        raise ValueError("loss_rate must be in [0, 1)")
    if trials <= 0:
        raise ValueError("trials must be positive")

    visible = rng.random((trials, n, n)) >= loss_rate
    indices = np.arange(n)
    visible[:, indices, indices] = True
    return visible


def sample_from_weight_tensor(
    weights: np.ndarray,
    voter_uniforms: np.ndarray,
) -> np.ndarray:
    """Sample one candidate per trial/voter from non-negative candidate weights."""
    weights = np.asarray(weights, dtype=float)
    voter_uniforms = np.asarray(voter_uniforms, dtype=float)
    if weights.ndim != 3 or weights.shape[0:2] != voter_uniforms.shape:
        raise ValueError("weights must have shape (trials, voters, candidates)")
    if np.any(weights < 0.0):
        raise ValueError("weights cannot be negative")

    totals = weights.sum(axis=2, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("every voter must have at least one positive candidate weight")
    probabilities = weights / totals
    cdf = np.cumsum(probabilities, axis=2)
    sampled = np.sum(cdf < voter_uniforms[:, :, None], axis=2)
    return np.minimum(sampled, weights.shape[2] - 1).astype(int)


def greedy_votes(costs: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Vote for the minimum-cost candidate visible to each voter."""
    masked_costs = np.where(visible, costs[None, None, :], np.inf)
    return np.argmin(masked_costs, axis=2).astype(int)


def inverse_cost_votes(
    costs: np.ndarray,
    visible: np.ndarray,
    voter_uniforms: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Sample one vote using weights proportional to 1 / cost**alpha."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    base_weights = np.power(1.0 / costs, alpha)
    weights = visible * base_weights[None, None, :]
    return sample_from_weight_tensor(weights, voter_uniforms)


def softmax_votes(
    costs: np.ndarray,
    visible: np.ndarray,
    voter_uniforms: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Sample one vote from row-normalized visible costs using Boltzmann weights."""
    if beta <= 0:
        raise ValueError("beta must be positive")

    broadcast_costs = np.broadcast_to(costs, visible.shape)
    visible_costs = np.where(visible, broadcast_costs, np.nan)
    row_min = np.nanmin(visible_costs, axis=2, keepdims=True)
    row_max = np.nanmax(visible_costs, axis=2, keepdims=True)
    span = row_max - row_min
    scaled = np.divide(
        visible_costs - row_min,
        span,
        out=np.zeros_like(visible_costs, dtype=float),
        where=span > 0.0,
    )
    weights = np.where(visible, np.exp(-beta * scaled), 0.0)
    return sample_from_weight_tensor(weights, voter_uniforms)


def rank_votes(
    visible: np.ndarray,
    voter_uniforms: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Sample one vote from visible cost rank using weights 1 / rank**gamma.

    The controlled cost model is strictly increasing by robot index, so the
    visible rank is the cumulative number of visible candidates by candidate ID.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    ranks = np.cumsum(visible, axis=2)
    safe_ranks = np.where(visible, ranks, 1)
    weights = np.where(visible, 1.0 / np.power(safe_ranks, gamma), 0.0)
    return sample_from_weight_tensor(weights, voter_uniforms)


def generate_policy_votes(
    costs: np.ndarray,
    visible: np.ndarray,
    voter_uniforms: np.ndarray,
    policy: PolicySpec,
) -> np.ndarray:
    """Dispatch one named local voting policy without changing communication data."""
    if policy.family == "greedy":
        return greedy_votes(costs, visible)
    if policy.family == "inverse":
        return inverse_cost_votes(costs, visible, voter_uniforms, float(policy.parameter))
    if policy.family == "softmax":
        return softmax_votes(costs, visible, voter_uniforms, float(policy.parameter))
    if policy.family == "rank":
        return rank_votes(visible, voter_uniforms, float(policy.parameter))
    raise ValueError(f"unknown policy family: {policy.family}")


def summarize_policy(
    *,
    n: int,
    loss_percent: int,
    trials: int,
    policy: PolicySpec,
    votes: np.ndarray,
    optimal_robot: int,
) -> dict[str, float | int | str]:
    vote_counts, votes_in_window = collect_votes_within_window(votes, n)
    majority = resolve_strict_majority(vote_counts, votes_in_window)
    execution = summarize_execution_outcome(vote_counts, majority, optimal_robot)
    return {
        "robots": n,
        "packet_loss_percent": loss_percent,
        "trials_per_point": trials,
        "method": policy.key,
        "method_label": policy.label,
        **execution,
    }


def run_configuration(
    n: int,
    loss_percent: int,
    trials: int,
    rng: np.random.Generator,
) -> list[dict[str, float | int | str]]:
    costs = generate_costs(n)
    optimal_robot = int(np.argmin(costs))
    visible = sample_paired_cost_visibility(n, loss_percent / 100.0, trials, rng)

    # Common random numbers keep stochastic policy comparisons paired.
    voter_uniforms = rng.random((trials, n))
    records: list[dict[str, float | int | str]] = []
    for policy in POLICIES:
        votes = generate_policy_votes(costs, visible, voter_uniforms, policy)
        records.append(
            summarize_policy(
                n=n,
                loss_percent=loss_percent,
                trials=trials,
                policy=policy,
                votes=votes,
                optimal_robot=optimal_robot,
            )
        )
    return records


def run_experiment(
    *,
    robot_counts: tuple[int, ...] = DEFAULT_ROBOT_COUNTS,
    loss_min_percent: int = LOSS_MIN_PERCENT,
    loss_max_percent: int = LOSS_MAX_PERCENT,
    trials: int = DEFAULT_TRIALS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    validate_experiment_config(robot_counts, loss_min_percent, loss_max_percent, trials)
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int | str]] = []

    for n in robot_counts:
        for loss_percent in range(loss_min_percent, loss_max_percent + 1):
            records.extend(run_configuration(n, loss_percent, trials, rng))
        print(f"robots={n:3d} complete")

    return pd.DataFrame.from_records(records)


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_robot_success_tables(results: pd.DataFrame) -> None:
    for n in sorted(results["robots"].unique()):
        subset = results[results["robots"] == n]
        table = subset.pivot(
            index="packet_loss_percent",
            columns="method_label",
            values="optimal_execution_success_percent",
        )
        ordered_labels = [policy.label for policy in POLICIES]
        table = table.reindex(columns=ordered_labels)
        table.to_csv(DATA_DIR / f"success_percent_N{int(n):03d}.csv")


def save_success_plot(results: pd.DataFrame, n: int) -> None:
    subset = results[results["robots"] == n]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    for policy in POLICIES:
        part = subset[subset["method"] == policy.key].sort_values("packet_loss_percent")
        ax.plot(
            part["packet_loss_percent"],
            part["optimal_execution_success_rate"],
            label=policy.label,
        )
    ax.set_xlabel("Directed P2P cost-message packet loss (%)")
    ax.set_ylabel("Optimal robot execution success rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"policy_success_N{n:03d}.png", dpi=180)
    plt.close(fig)


def save_outputs(results: pd.DataFrame) -> None:
    ensure_output_dirs()
    results.to_csv(DATA_DIR / "peer_cost_policy_comparison_summary.csv", index=False)
    save_robot_success_tables(results)
    for n in sorted(results["robots"].unique()):
        save_success_plot(results, int(n))


def printable_success_table(results: pd.DataFrame, n: int) -> pd.DataFrame:
    subset = results[
        (results["robots"] == n)
        & (results["packet_loss_percent"].isin(REPORT_LOSSES))
    ]
    table = subset.pivot(
        index="packet_loss_percent",
        columns="method_label",
        values="optimal_execution_success_percent",
    )
    ordered_labels = [policy.label for policy in POLICIES]
    return table.reindex(columns=ordered_labels).reset_index()


def parse_robot_counts(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("robot counts cannot be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 10 single-task local voting policies under the same directed "
            "P2P cost-message packet-loss realizations. Each point uses 100 trials "
            "by default and reports optimal-executor success percentage."
        )
    )
    parser.add_argument(
        "--robot-counts",
        type=parse_robot_counts,
        default=DEFAULT_ROBOT_COUNTS,
        help="Comma-separated robot counts (default: 5,10,20,30,50,75,100).",
    )
    parser.add_argument("--loss-min", type=int, default=LOSS_MIN_PERCENT)
    parser.add_argument("--loss-max", type=int, default=LOSS_MAX_PERCENT)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(
        robot_counts=tuple(args.robot_counts),
        loss_min_percent=args.loss_min,
        loss_max_percent=args.loss_max,
        trials=args.trials,
        seed=args.seed,
    )
    save_outputs(results)

    print("\nSaved full comparison summary:")
    print(DATA_DIR / "peer_cost_policy_comparison_summary.csv")
    print("Saved report tables:")
    print(DATA_DIR / "success_percent_Nxxx.csv")
    print("Saved figures:")
    print(FIGURE_DIR)

    for n in args.robot_counts:
        print(f"\nPasteable success table - N={n} (percent, {args.trials} trials/point):")
        print(printable_success_table(results, int(n)).to_string(index=False))


if __name__ == "__main__":
    main()
