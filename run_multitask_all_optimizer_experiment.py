from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_multitask_optimizer_screening import (
    ACOConfig,
    MILP_NUMERICAL_TOLERANCE_PERCENT,
    solve_aco_assignment,
    solve_milp_assignment,
    validate_aco_config,
)
from run_multitask_peer_cost_experiment import (
    DEFAULT_TRIALS,
    NEAR_OPTIMAL_GAP_PERCENT,
    OPTIMAL_COST_TOLERANCE_PERCENT,
    RANDOM_SEED,
    ROBOT_COUNT,
    TASK_COUNTS,
    assignment_total_cost,
    generate_spatial_cost_matrix,
    solve_batched_auction_assignments,
    solve_hungarian_assignment,
    solve_sequential_greedy,
    validate_experiment_config,
)

METHODS = ("hungarian", "auction", "milp", "aco_ls", "greedy")
METHOD_LABELS = {
    "hungarian": "Hungarian",
    "auction": "Auction",
    "milp": "MILP",
    "aco_ls": "ACO + Local Search",
    "greedy": "Greedy Baseline",
}

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "multitask_all_optimizer"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


def fail(function: str, category: str, code: str, details: str) -> None:
    raise ValueError(
        "owner=run_multitask_all_optimizer_experiment "
        f"function={function} category={category} code={code} details={details}"
    )


def cost_match_tolerance_percent(method: str) -> float:
    if method == "milp":
        return MILP_NUMERICAL_TOLERANCE_PERCENT
    if method in METHODS:
        return OPTIMAL_COST_TOLERANCE_PERCENT
    fail("cost_match_tolerance_percent", "contract", "UNKNOWN_METHOD", f"method={method}")


def solve_single_auction_assignment(costs: np.ndarray) -> np.ndarray | None:
    """Reuse the canonical batched Auction owner for one complete-information matrix."""
    assignments, valid = solve_batched_auction_assignments(costs[None, :, :])
    if valid.shape != (1,):
        fail(
            "solve_single_auction_assignment",
            "contract",
            "AUCTION_VALID_SHAPE_MISMATCH",
            f"expected={(1,)} actual={valid.shape}",
        )
    if not bool(valid[0]):
        return None
    return assignments[0].copy()


def evaluate_method(
    *,
    task_count: int,
    trial: int,
    method: str,
    costs: np.ndarray,
    assignment: np.ndarray,
    optimal_assignment: np.ndarray,
    optimal_cost: float,
    runtime_ms: float,
) -> dict[str, object]:
    total_cost = assignment_total_cost(costs, assignment)
    gap_percent = 100.0 * (total_cost - optimal_cost) / optimal_cost
    return {
        "robots": ROBOT_COUNT,
        "tasks": task_count,
        "trial": trial,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "total_cost": total_cost,
        "optimal_total_cost": optimal_cost,
        "optimality_gap_percent": gap_percent,
        "optimal_cost_match": abs(gap_percent) <= cost_match_tolerance_percent(method),
        "near_optimal_5pct": gap_percent <= NEAR_OPTIMAL_GAP_PERCENT + 1e-12,
        "exact_optimal_assignment": bool(np.array_equal(assignment, optimal_assignment)),
        "runtime_ms": runtime_ms,
    }


def validate_exact_optimizer_contract(seed: int) -> None:
    """Require Auction and MILP to recover the Hungarian optimum on representative cases."""
    rng = np.random.default_rng(seed + 381227)
    for task_count in (1, 5, 50, 100):
        costs = generate_spatial_cost_matrix(ROBOT_COUNT, task_count, rng)
        hungarian = solve_hungarian_assignment(costs)
        auction = solve_single_auction_assignment(costs)
        milp = solve_milp_assignment(costs)
        if hungarian is None or auction is None or milp is None:
            fail(
                "validate_exact_optimizer_contract",
                "planning",
                "EXACT_OPTIMIZER_INFEASIBLE",
                f"tasks={task_count}",
            )
        optimal_cost = assignment_total_cost(costs, hungarian)
        for method, assignment in (("auction", auction), ("milp", milp)):
            actual_cost = assignment_total_cost(costs, assignment)
            gap_percent = 100.0 * (actual_cost - optimal_cost) / optimal_cost
            tolerance = cost_match_tolerance_percent(method)
            if abs(gap_percent) > tolerance:
                fail(
                    "validate_exact_optimizer_contract",
                    "planning",
                    "EXACT_OPTIMIZER_NOT_HUNGARIAN",
                    (
                        f"method={method} tasks={task_count} "
                        f"expected_abs_gap_percent<={tolerance} actual={gap_percent}"
                    ),
                )


def solve_timed(
    method: str,
    costs: np.ndarray,
    task_order: np.ndarray,
    aco_rng: np.random.Generator,
    aco_config: ACOConfig,
) -> tuple[np.ndarray | None, float]:
    """Time exactly one optimizer call without changing its input matrix."""
    start = time.perf_counter()
    if method == "hungarian":
        assignment = solve_hungarian_assignment(costs)
    elif method == "auction":
        assignment = solve_single_auction_assignment(costs)
    elif method == "milp":
        assignment = solve_milp_assignment(costs)
    elif method == "aco_ls":
        assignment = solve_aco_assignment(costs, task_order, aco_rng, aco_config)
    elif method == "greedy":
        assignment = solve_sequential_greedy(costs, task_order)
    else:
        fail("solve_timed", "contract", "UNKNOWN_METHOD", f"method={method}")
    runtime_ms = 1000.0 * (time.perf_counter() - start)
    return assignment, runtime_ms


def run_trial(
    *,
    task_count: int,
    trial: int,
    seed: int,
    aco_config: ACOConfig,
) -> list[dict[str, object]]:
    """Generate one paired scenario, then give that exact scenario to all methods."""
    trial_seed = seed + task_count * 100003 + trial * 1009
    scenario_rng = np.random.default_rng(trial_seed)
    costs = generate_spatial_cost_matrix(ROBOT_COUNT, task_count, scenario_rng)
    task_order = scenario_rng.permutation(task_count)

    hungarian, hungarian_runtime_ms = solve_timed(
        "hungarian",
        costs,
        task_order,
        np.random.default_rng(trial_seed + 7000003),
        aco_config,
    )
    if hungarian is None:
        fail("run_trial", "planning", "HUNGARIAN_INFEASIBLE", f"tasks={task_count} trial={trial}")
    optimal_cost = assignment_total_cost(costs, hungarian)

    records = [
        evaluate_method(
            task_count=task_count,
            trial=trial,
            method="hungarian",
            costs=costs,
            assignment=hungarian,
            optimal_assignment=hungarian,
            optimal_cost=optimal_cost,
            runtime_ms=hungarian_runtime_ms,
        )
    ]

    for method in ("auction", "milp", "aco_ls", "greedy"):
        method_rng = np.random.default_rng(trial_seed + 7000003)
        assignment, runtime_ms = solve_timed(method, costs, task_order, method_rng, aco_config)
        if assignment is None:
            fail(
                "run_trial",
                "planning",
                "OPTIMIZER_INFEASIBLE",
                f"method={method} tasks={task_count} trial={trial}",
            )
        if method in ("auction", "milp"):
            actual_cost = assignment_total_cost(costs, assignment)
            gap_percent = 100.0 * (actual_cost - optimal_cost) / optimal_cost
            tolerance = cost_match_tolerance_percent(method)
            if abs(gap_percent) > tolerance:
                fail(
                    "run_trial",
                    "planning",
                    "EXACT_OPTIMIZER_NOT_EXACT",
                    (
                        f"method={method} tasks={task_count} trial={trial} "
                        f"expected_abs_gap_percent<={tolerance} actual={gap_percent}"
                    ),
                )
        records.append(
            evaluate_method(
                task_count=task_count,
                trial=trial,
                method=method,
                costs=costs,
                assignment=assignment,
                optimal_assignment=hungarian,
                optimal_cost=optimal_cost,
                runtime_ms=runtime_ms,
            )
        )
    return records


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["tasks", "method", "method_label"], as_index=False)
        .agg(
            average_total_cost=("total_cost", "mean"),
            average_optimality_gap_percent=("optimality_gap_percent", "mean"),
            optimal_cost_match_percent=("optimal_cost_match", lambda series: 100.0 * series.mean()),
            near_optimal_5pct_percent=("near_optimal_5pct", lambda series: 100.0 * series.mean()),
            exact_optimal_assignment_percent=("exact_optimal_assignment", lambda series: 100.0 * series.mean()),
            average_runtime_ms=("runtime_ms", "mean"),
            median_runtime_ms=("runtime_ms", "median"),
        )
        .sort_values(["tasks", "method"])
        .reset_index(drop=True)
    )


def run_experiment(
    *,
    task_counts: tuple[int, ...] = TASK_COUNTS,
    trials: int = DEFAULT_TRIALS,
    seed: int = RANDOM_SEED,
    aco_config: ACOConfig = ACOConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_experiment_config(ROBOT_COUNT, 0.0, task_counts, trials)
    validate_aco_config(aco_config)
    validate_exact_optimizer_contract(seed)
    print("Exact optimizer contract: PASS (Auction/MILP match Hungarian reference)")

    records: list[dict[str, object]] = []
    for task_count in task_counts:
        for trial in range(1, trials + 1):
            records.extend(
                run_trial(
                    task_count=task_count,
                    trial=trial,
                    seed=seed,
                    aco_config=aco_config,
                )
            )
        print(f"tasks={task_count:3d}/{max(task_counts)} complete")

    raw = pd.DataFrame.from_records(records)
    return raw, summarize_results(raw)


def report_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = summary.pivot(index="tasks", columns="method_label", values=metric)
    labels = [METHOD_LABELS[method] for method in METHODS]
    return table.reindex(columns=labels).reset_index()


def readable_report_table(summary: pd.DataFrame, metric: str, decimals: int) -> pd.DataFrame:
    table = report_table(summary, metric).copy()
    value_columns = [column for column in table.columns if column != "tasks"]
    table[value_columns] = table[value_columns].round(decimals)
    return table


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_metric_plot(summary: pd.DataFrame, *, metric: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in METHODS:
        part = summary[summary["method"] == method].sort_values("tasks")
        ax.plot(part["tasks"], part[metric], marker="o", label=METHOD_LABELS[method])
    ax.set_xlabel("Simultaneous task count")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close(fig)


def save_outputs(raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    ensure_output_dirs()
    raw.to_csv(DATA_DIR / "all_optimizer_raw.csv", index=False)
    summary.to_csv(DATA_DIR / "all_optimizer_summary.csv", index=False)

    metrics = {
        "average_optimality_gap_percent": "report_average_optimality_gap_percent.csv",
        "optimal_cost_match_percent": "report_optimal_cost_match_percent.csv",
        "near_optimal_5pct_percent": "report_near_optimal_5pct_percent.csv",
        "exact_optimal_assignment_percent": "report_exact_optimal_assignment_percent.csv",
        "average_runtime_ms": "report_average_runtime_ms.csv",
    }
    for metric, filename in metrics.items():
        report_table(summary, metric).to_csv(DATA_DIR / filename, index=False)

    save_metric_plot(
        summary,
        metric="average_optimality_gap_percent",
        ylabel="Average optimality gap (%)",
        filename="average_optimality_gap_percent.png",
    )
    save_metric_plot(
        summary,
        metric="optimal_cost_match_percent",
        ylabel="Optimal-cost match rate (%)",
        filename="optimal_cost_match_percent.png",
    )
    save_metric_plot(
        summary,
        metric="average_runtime_ms",
        ylabel="Average optimizer runtime (ms)",
        filename="average_runtime_ms.png",
    )


def parse_task_counts(values: list[int] | None) -> tuple[int, ...]:
    return TASK_COUNTS if values is None else tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Hungarian, Auction, MILP, ACO + Local Search, and Greedy on the same "
            "100 paired complete-information scenarios per task count."
        )
    )
    parser.add_argument("--tasks", nargs="+", type=int, default=None)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--aco-ants", type=int, default=ACOConfig.ants)
    parser.add_argument("--aco-iterations", type=int, default=ACOConfig.iterations)
    parser.add_argument("--aco-alpha", type=float, default=ACOConfig.alpha)
    parser.add_argument("--aco-beta", type=float, default=ACOConfig.beta)
    parser.add_argument("--aco-evaporation", type=float, default=ACOConfig.evaporation)
    parser.add_argument("--aco-candidate-list", type=int, default=ACOConfig.candidate_list_size)
    parser.add_argument("--aco-elite-weight", type=float, default=ACOConfig.elite_weight)
    parser.add_argument("--aco-local-search-moves", type=int, default=ACOConfig.local_search_moves)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aco_config = ACOConfig(
        ants=args.aco_ants,
        iterations=args.aco_iterations,
        alpha=args.aco_alpha,
        beta=args.aco_beta,
        evaporation=args.aco_evaporation,
        candidate_list_size=args.aco_candidate_list,
        elite_weight=args.aco_elite_weight,
        local_search_moves=args.aco_local_search_moves,
    )
    raw, summary = run_experiment(
        task_counts=parse_task_counts(args.tasks),
        trials=args.trials,
        seed=args.seed,
        aco_config=aco_config,
    )
    save_outputs(raw, summary)

    print("\nSaved:")
    print(DATA_DIR / "all_optimizer_raw.csv")
    print(DATA_DIR / "all_optimizer_summary.csv")
    print(FIGURE_DIR)

    print("\nAverage optimality gap (%) - lower is better:")
    print(readable_report_table(summary, "average_optimality_gap_percent", 3).to_string(index=False))

    print("\nOptimal-cost match (%) - higher is better:")
    print(readable_report_table(summary, "optimal_cost_match_percent", 1).to_string(index=False))

    print("\nNear-optimal within 5% (%) - higher is better:")
    print(readable_report_table(summary, "near_optimal_5pct_percent", 1).to_string(index=False))

    print("\nAverage optimizer runtime (ms) - lower is better:")
    print(readable_report_table(summary, "average_runtime_ms", 3).to_string(index=False))


if __name__ == "__main__":
    main()
