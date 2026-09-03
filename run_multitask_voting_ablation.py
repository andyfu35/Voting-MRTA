from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_multitask_peer_cost_experiment import (
    DEFAULT_TRIALS,
    NEAR_OPTIMAL_GAP_PERCENT,
    OPTIMAL_COST_TOLERANCE_PERCENT,
    PACKET_LOSS_RATE,
    RANDOM_SEED,
    ROBOT_COUNT,
    TASK_COUNTS,
    assignment_total_cost,
    build_assignment_support,
    build_receiver_cost_views,
    generate_spatial_cost_matrix,
    sample_p2p_cost_visibility,
    solve_hungarian_assignment,
    solve_local_optimizer_proposals,
    solve_support_consensus,
    validate_experiment_config,
)

DIRECT_RECEIVER = 0
OPTIMIZERS = ("p2p_hungarian", "p2p_auction")
METHODS = (
    "oracle",
    "direct_hungarian",
    "voting_hungarian",
    "direct_auction",
    "voting_auction",
)
METHOD_LABELS = {
    "oracle": "Hungarian Oracle",
    "direct_hungarian": "Direct Hungarian",
    "voting_hungarian": "Voting Hungarian",
    "direct_auction": "Direct Auction",
    "voting_auction": "Voting Auction",
}
OPTIMIZER_METHOD_KEYS = {
    "p2p_hungarian": ("direct_hungarian", "voting_hungarian"),
    "p2p_auction": ("direct_auction", "voting_auction"),
}

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "multitask_voting_ablation"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


def fail(function: str, category: str, code: str, details: str) -> None:
    """Raise one ablation diagnostic at the first named owner boundary."""
    raise ValueError(
        "owner=run_multitask_voting_ablation "
        f"function={function} category={category} code={code} details={details}"
    )


def validate_ablation_config(direct_receiver: int) -> None:
    """Validate the fixed single-receiver direct-control boundary."""
    if not 0 <= direct_receiver < ROBOT_COUNT:
        fail(
            "validate_ablation_config",
            "contract",
            "DIRECT_RECEIVER_OUT_OF_RANGE",
            f"expected=0..{ROBOT_COUNT - 1} actual={direct_receiver}",
        )


def select_direct_assignment(
    proposals: np.ndarray,
    valid: np.ndarray,
    direct_receiver: int,
) -> np.ndarray | None:
    """Adopt exactly one designated receiver proposal with no fallback or voting."""
    if proposals.ndim != 2:
        fail(
            "select_direct_assignment",
            "contract",
            "INVALID_PROPOSAL_SHAPE",
            f"actual={proposals.shape}",
        )
    if valid.shape != (proposals.shape[0],):
        fail(
            "select_direct_assignment",
            "contract",
            "VALID_MASK_SHAPE_MISMATCH",
            f"valid={valid.shape} proposals={proposals.shape}",
        )
    if not 0 <= direct_receiver < proposals.shape[0]:
        fail(
            "select_direct_assignment",
            "contract",
            "DIRECT_RECEIVER_OUT_OF_RANGE",
            f"receivers={proposals.shape[0]} actual={direct_receiver}",
        )
    if not bool(valid[direct_receiver]):
        return None
    return proposals[direct_receiver].copy()


def solve_voting_assignment(
    proposals: np.ndarray,
    valid: np.ndarray,
    robot_count: int,
    tie_priority: np.ndarray,
) -> np.ndarray:
    """Use the existing proposal-support consensus on the shared proposal batch."""
    support = build_assignment_support(proposals, valid, robot_count)
    return solve_support_consensus(support, tie_priority)


def evaluate_ablation_assignment(
    *,
    task_count: int,
    trial: int,
    method: str,
    packet_loss_rate: float,
    costs: np.ndarray,
    assignment: np.ndarray | None,
    optimal_assignment: np.ndarray,
    optimal_cost: float,
    local_valid_proposal_rate_percent: float,
) -> dict[str, object]:
    """Evaluate one direct/voting result while keeping invalid decisions visible."""
    valid_assignment = assignment is not None
    if not valid_assignment:
        return {
            "robots": ROBOT_COUNT,
            "packet_loss_percent": 100.0 * packet_loss_rate,
            "tasks": task_count,
            "trial": trial,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "valid_assignment": False,
            "total_cost": np.nan,
            "optimal_total_cost": optimal_cost,
            "optimality_gap_percent": np.nan,
            "optimal_cost_match": False,
            "near_optimal_5pct": False,
            "exact_optimal_assignment": False,
            "local_valid_proposal_rate_percent": local_valid_proposal_rate_percent,
        }

    total_cost = assignment_total_cost(costs, assignment)
    gap_percent = 100.0 * (total_cost - optimal_cost) / optimal_cost
    return {
        "robots": ROBOT_COUNT,
        "packet_loss_percent": 100.0 * packet_loss_rate,
        "tasks": task_count,
        "trial": trial,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "valid_assignment": True,
        "total_cost": total_cost,
        "optimal_total_cost": optimal_cost,
        "optimality_gap_percent": gap_percent,
        "optimal_cost_match": abs(gap_percent) <= OPTIMAL_COST_TOLERANCE_PERCENT,
        "near_optimal_5pct": gap_percent <= NEAR_OPTIMAL_GAP_PERCENT + 1e-12,
        "exact_optimal_assignment": bool(np.array_equal(assignment, optimal_assignment)),
        "local_valid_proposal_rate_percent": local_valid_proposal_rate_percent,
    }


def append_optimizer_ablation_records(
    *,
    records: list[dict[str, object]],
    optimizer: str,
    task_count: int,
    trial: int,
    packet_loss_rate: float,
    costs: np.ndarray,
    receiver_costs: np.ndarray,
    task_order: np.ndarray,
    tie_priority: np.ndarray,
    optimal_assignment: np.ndarray,
    optimal_cost: float,
    direct_receiver: int,
) -> None:
    """Solve local proposals once, then branch only at Direct vs Voting aggregation."""
    proposals, valid = solve_local_optimizer_proposals(
        optimizer,
        receiver_costs,
        task_order,
    )
    direct_method, voting_method = OPTIMIZER_METHOD_KEYS[optimizer]
    direct_assignment = select_direct_assignment(
        proposals,
        valid,
        direct_receiver,
    )
    voting_assignment = solve_voting_assignment(
        proposals,
        valid,
        costs.shape[0],
        tie_priority,
    )
    valid_rate_percent = 100.0 * float(valid.mean())

    records.append(
        evaluate_ablation_assignment(
            task_count=task_count,
            trial=trial,
            method=direct_method,
            packet_loss_rate=packet_loss_rate,
            costs=costs,
            assignment=direct_assignment,
            optimal_assignment=optimal_assignment,
            optimal_cost=optimal_cost,
            local_valid_proposal_rate_percent=(
                100.0 if bool(valid[direct_receiver]) else 0.0
            ),
        )
    )
    records.append(
        evaluate_ablation_assignment(
            task_count=task_count,
            trial=trial,
            method=voting_method,
            packet_loss_rate=packet_loss_rate,
            costs=costs,
            assignment=voting_assignment,
            optimal_assignment=optimal_assignment,
            optimal_cost=optimal_cost,
            local_valid_proposal_rate_percent=valid_rate_percent,
        )
    )


def validate_zero_loss_ablation_contract(
    seed: int,
    direct_receiver: int,
) -> None:
    """Require Direct and Voting exact optimizers to recover oracle with full data."""
    rng = np.random.default_rng(seed + 187631)
    for task_count in (1, 5, 50, 100):
        costs = generate_spatial_cost_matrix(ROBOT_COUNT, task_count, rng)
        visibility = np.ones(
            (ROBOT_COUNT, ROBOT_COUNT, task_count),
            dtype=bool,
        )
        task_order = rng.permutation(task_count)
        tie_priority = rng.random((ROBOT_COUNT, task_count))
        receiver_costs = build_receiver_cost_views(costs, visibility)

        oracle = solve_hungarian_assignment(costs)
        if oracle is None:
            fail(
                "validate_zero_loss_ablation_contract",
                "planning",
                "ORACLE_INFEASIBLE",
                f"tasks={task_count}",
            )
        oracle_cost = assignment_total_cost(costs, oracle)

        for optimizer in OPTIMIZERS:
            proposals, valid = solve_local_optimizer_proposals(
                optimizer,
                receiver_costs,
                task_order,
            )
            direct = select_direct_assignment(
                proposals,
                valid,
                direct_receiver,
            )
            voting = solve_voting_assignment(
                proposals,
                valid,
                ROBOT_COUNT,
                tie_priority,
            )
            if direct is None:
                fail(
                    "validate_zero_loss_ablation_contract",
                    "planning",
                    "ZERO_LOSS_DIRECT_INVALID",
                    f"optimizer={optimizer} tasks={task_count}",
                )
            for path, assignment in (("direct", direct), ("voting", voting)):
                actual_cost = assignment_total_cost(costs, assignment)
                gap_percent = 100.0 * (actual_cost - oracle_cost) / oracle_cost
                if abs(gap_percent) > OPTIMAL_COST_TOLERANCE_PERCENT:
                    fail(
                        "validate_zero_loss_ablation_contract",
                        "planning",
                        "ZERO_LOSS_ABLATION_NOT_ORACLE_CONSISTENT",
                        (
                            f"optimizer={optimizer} path={path} tasks={task_count} "
                            f"gap_percent={gap_percent}"
                        ),
                    )


def run_trial(
    *,
    task_count: int,
    trial: int,
    packet_loss_rate: float,
    rng: np.random.Generator,
    direct_receiver: int,
) -> list[dict[str, object]]:
    """Run one paired Direct-vs-Voting ablation trial."""
    costs = generate_spatial_cost_matrix(ROBOT_COUNT, task_count, rng)
    visibility = sample_p2p_cost_visibility(
        ROBOT_COUNT,
        task_count,
        packet_loss_rate,
        rng,
    )
    task_order = rng.permutation(task_count)
    tie_priority = rng.random((ROBOT_COUNT, task_count))
    receiver_costs = build_receiver_cost_views(costs, visibility)

    optimal_assignment = solve_hungarian_assignment(costs)
    if optimal_assignment is None:
        fail(
            "run_trial",
            "planning",
            "ORACLE_INFEASIBLE",
            f"tasks={task_count} trial={trial}",
        )
    optimal_cost = assignment_total_cost(costs, optimal_assignment)

    records: list[dict[str, object]] = [
        evaluate_ablation_assignment(
            task_count=task_count,
            trial=trial,
            method="oracle",
            packet_loss_rate=packet_loss_rate,
            costs=costs,
            assignment=optimal_assignment,
            optimal_assignment=optimal_assignment,
            optimal_cost=optimal_cost,
            local_valid_proposal_rate_percent=100.0,
        )
    ]

    for optimizer in OPTIMIZERS:
        append_optimizer_ablation_records(
            records=records,
            optimizer=optimizer,
            task_count=task_count,
            trial=trial,
            packet_loss_rate=packet_loss_rate,
            costs=costs,
            receiver_costs=receiver_costs,
            task_order=task_order,
            tie_priority=tie_priority,
            optimal_assignment=optimal_assignment,
            optimal_cost=optimal_cost,
            direct_receiver=direct_receiver,
        )
    return records


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize quality and validity; invalid Direct decisions count as failures."""
    return (
        raw.groupby(["tasks", "method", "method_label"], as_index=False)
        .agg(
            average_total_cost=("total_cost", "mean"),
            average_optimality_gap_percent=("optimality_gap_percent", "mean"),
            optimal_cost_match_percent=(
                "optimal_cost_match",
                lambda series: 100.0 * series.mean(),
            ),
            near_optimal_5pct_percent=(
                "near_optimal_5pct",
                lambda series: 100.0 * series.mean(),
            ),
            valid_assignment_percent=(
                "valid_assignment",
                lambda series: 100.0 * series.mean(),
            ),
            average_local_valid_proposal_rate_percent=(
                "local_valid_proposal_rate_percent",
                "mean",
            ),
        )
        .sort_values(["tasks", "method"])
        .reset_index(drop=True)
    )


def run_experiment(
    *,
    task_counts: tuple[int, ...] = TASK_COUNTS,
    trials: int = DEFAULT_TRIALS,
    packet_loss_rate: float = PACKET_LOSS_RATE,
    seed: int = RANDOM_SEED,
    direct_receiver: int = DIRECT_RECEIVER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the 100-robot paired Direct-vs-Voting ablation sweep."""
    validate_experiment_config(
        ROBOT_COUNT,
        packet_loss_rate,
        task_counts,
        trials,
    )
    validate_ablation_config(direct_receiver)
    validate_zero_loss_ablation_contract(seed, direct_receiver)
    print(
        "Zero-loss ablation contract: PASS "
        "(Direct/Voting Hungarian and Auction match oracle)"
    )

    records: list[dict[str, object]] = []
    for task_count in task_counts:
        rng = np.random.default_rng(seed + task_count * 100003)
        for trial in range(1, trials + 1):
            records.extend(
                run_trial(
                    task_count=task_count,
                    trial=trial,
                    packet_loss_rate=packet_loss_rate,
                    rng=rng,
                    direct_receiver=direct_receiver,
                )
            )
        print(f"tasks={task_count:3d}/{max(task_counts)} complete")

    raw = pd.DataFrame.from_records(records)
    return raw, summarize_results(raw)


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def report_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = summary.pivot(
        index="tasks",
        columns="method_label",
        values=metric,
    )
    ordered_labels = [METHOD_LABELS[method] for method in METHODS]
    return table.reindex(columns=ordered_labels).reset_index()


def ablation_pair_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return Direct/Voting pairs without the constant oracle column."""
    table = report_table(summary, metric)
    return table[
        [
            "tasks",
            "Direct Hungarian",
            "Voting Hungarian",
            "Direct Auction",
            "Voting Auction",
        ]
    ]


def voting_uplift_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Positive match uplift / gap reduction means Voting improved the result."""
    match = ablation_pair_table(summary, "optimal_cost_match_percent")
    gap = ablation_pair_table(summary, "average_optimality_gap_percent")
    return pd.DataFrame(
        {
            "tasks": match["tasks"],
            "hungarian_match_uplift_pp": (
                match["Voting Hungarian"] - match["Direct Hungarian"]
            ),
            "auction_match_uplift_pp": (
                match["Voting Auction"] - match["Direct Auction"]
            ),
            "hungarian_gap_reduction_pp": (
                gap["Direct Hungarian"] - gap["Voting Hungarian"]
            ),
            "auction_gap_reduction_pp": (
                gap["Direct Auction"] - gap["Voting Auction"]
            ),
        }
    )


def save_metric_plot(
    summary: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in METHODS[1:]:
        part = summary[summary["method"] == method].sort_values("tasks")
        ax.plot(
            part["tasks"],
            part[metric],
            marker="o",
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Simultaneous task count")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close(fig)


def save_outputs(raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    ensure_output_dirs()
    raw.to_csv(DATA_DIR / "voting_ablation_raw.csv", index=False)
    summary.to_csv(DATA_DIR / "voting_ablation_summary.csv", index=False)
    ablation_pair_table(summary, "optimal_cost_match_percent").to_csv(
        DATA_DIR / "report_optimal_cost_match_percent.csv",
        index=False,
    )
    ablation_pair_table(summary, "average_optimality_gap_percent").to_csv(
        DATA_DIR / "report_average_optimality_gap_percent.csv",
        index=False,
    )
    ablation_pair_table(summary, "near_optimal_5pct_percent").to_csv(
        DATA_DIR / "report_near_optimal_5pct_percent.csv",
        index=False,
    )
    ablation_pair_table(summary, "valid_assignment_percent").to_csv(
        DATA_DIR / "report_valid_assignment_percent.csv",
        index=False,
    )
    voting_uplift_table(summary).to_csv(
        DATA_DIR / "report_voting_uplift.csv",
        index=False,
    )
    save_metric_plot(
        summary,
        metric="optimal_cost_match_percent",
        ylabel="Optimal-cost match rate (%)",
        filename="optimal_cost_match_ablation.png",
    )
    save_metric_plot(
        summary,
        metric="average_optimality_gap_percent",
        ylabel="Average optimality gap among valid assignments (%)",
        filename="average_optimality_gap_ablation.png",
    )


def parse_task_counts(values: list[int] | None) -> tuple[int, ...]:
    if values is None:
        return TASK_COUNTS
    parsed = tuple(values)
    if not parsed:
        raise argparse.ArgumentTypeError("task counts cannot be empty")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablate the proposal-consensus mechanism using the same optimizer, "
            "cost matrix, and P2P loss realization. Direct adopts one fixed "
            "receiver proposal; Voting aggregates all receiver proposals."
        )
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=int,
        default=None,
        help="Task counts. Default: 5 10 20 30 40 50 60 70 80 90 100",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--packet-loss",
        type=float,
        default=PACKET_LOSS_RATE,
        help="Directed P2P loss probability (default: 0.30).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--direct-receiver",
        type=int,
        default=DIRECT_RECEIVER,
        help="Fixed receiver used by Direct baselines (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_counts = parse_task_counts(args.tasks)
    raw, summary = run_experiment(
        task_counts=task_counts,
        trials=args.trials,
        packet_loss_rate=args.packet_loss,
        seed=args.seed,
        direct_receiver=args.direct_receiver,
    )
    save_outputs(raw, summary)

    print("\nSaved:")
    print(DATA_DIR / "voting_ablation_raw.csv")
    print(DATA_DIR / "voting_ablation_summary.csv")
    print(FIGURE_DIR)

    print("\nOptimal-cost match (%) - higher is better:")
    print(
        ablation_pair_table(
            summary,
            "optimal_cost_match_percent",
        ).to_string(index=False)
    )

    print("\nAverage optimality gap (%) - lower is better:")
    print(
        ablation_pair_table(
            summary,
            "average_optimality_gap_percent",
        ).to_string(index=False)
    )

    print("\nNear-optimal within 5% (%) - higher is better:")
    print(
        ablation_pair_table(
            summary,
            "near_optimal_5pct_percent",
        ).to_string(index=False)
    )

    print("\nValid final assignment (%) - higher is better:")
    print(
        ablation_pair_table(
            summary,
            "valid_assignment_percent",
        ).to_string(index=False)
    )

    print("\nVoting uplift (positive means Voting improved):")
    print(voting_uplift_table(summary).to_string(index=False))


if __name__ == "__main__":
    main()
