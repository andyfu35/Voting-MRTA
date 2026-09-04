from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import eye, kron, vstack

from run_multitask_peer_cost_experiment import (
    DEFAULT_TRIALS,
    NEAR_OPTIMAL_GAP_PERCENT,
    OPTIMAL_COST_TOLERANCE_PERCENT,
    RANDOM_SEED,
    ROBOT_COUNT,
    TASK_COUNTS,
    assignment_total_cost,
    generate_spatial_cost_matrix,
    solve_hungarian_assignment,
    solve_sequential_greedy,
    validate_experiment_config,
)

METHODS = ("hungarian", "milp", "aco_ls", "greedy")
METHOD_LABELS = {
    "hungarian": "Hungarian",
    "milp": "MILP",
    "aco_ls": "ACO + Local Search",
    "greedy": "Greedy Baseline",
}

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "multitask_optimizer_screening"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"
MILP_NUMERICAL_TOLERANCE_PERCENT = 1e-6


@dataclass(frozen=True)
class ACOConfig:
    ants: int = 12
    iterations: int = 15
    alpha: float = 1.0
    beta: float = 3.0
    evaporation: float = 0.20
    candidate_list_size: int = 20
    elite_weight: float = 2.0
    local_search_moves: int = 25


def fail(function: str, category: str, code: str, details: str) -> None:
    """Raise one screening diagnostic at the first named owner boundary."""
    raise ValueError(
        "owner=run_multitask_optimizer_screening "
        f"function={function} category={category} code={code} details={details}"
    )


def validate_aco_config(config: ACOConfig) -> None:
    """Validate fixed ACO search controls before any experiment work starts."""
    if config.ants <= 0:
        fail("validate_aco_config", "contract", "INVALID_ANT_COUNT", f"actual={config.ants}")
    if config.iterations <= 0:
        fail(
            "validate_aco_config",
            "contract",
            "INVALID_ACO_ITERATIONS",
            f"actual={config.iterations}",
        )
    if config.alpha < 0.0 or config.beta < 0.0:
        fail(
            "validate_aco_config",
            "contract",
            "INVALID_ACO_EXPONENT",
            f"alpha={config.alpha} beta={config.beta}",
        )
    if not 0.0 < config.evaporation < 1.0:
        fail(
            "validate_aco_config",
            "contract",
            "INVALID_ACO_EVAPORATION",
            f"actual={config.evaporation}",
        )
    if config.candidate_list_size <= 0:
        fail(
            "validate_aco_config",
            "contract",
            "INVALID_ACO_CANDIDATE_LIST",
            f"actual={config.candidate_list_size}",
        )
    if config.elite_weight < 0.0:
        fail(
            "validate_aco_config",
            "contract",
            "INVALID_ACO_ELITE_WEIGHT",
            f"actual={config.elite_weight}",
        )
    if config.local_search_moves < 0:
        fail(
            "validate_aco_config",
            "contract",
            "INVALID_ACO_LOCAL_SEARCH_MOVES",
            f"actual={config.local_search_moves}",
        )


def cost_match_tolerance_percent(method: str) -> float:
    """Return the numerical objective-match tolerance owned by each solver family."""
    if method == "milp":
        return MILP_NUMERICAL_TOLERANCE_PERCENT
    if method in METHODS:
        return OPTIMAL_COST_TOLERANCE_PERCENT
    fail("cost_match_tolerance_percent", "contract", "UNKNOWN_METHOD", f"method={method}")


@lru_cache(maxsize=None)
def build_milp_assignment_model(
    robot_count: int,
    task_count: int,
) -> tuple[LinearConstraint, Bounds, np.ndarray]:
    """Build and cache the assignment-only MILP constraints for one matrix shape."""
    if robot_count < task_count or task_count <= 0:
        fail(
            "build_milp_assignment_model",
            "contract",
            "INVALID_ASSIGNMENT_SHAPE",
            f"robots={robot_count} tasks={task_count}",
        )

    task_constraints = kron(
        np.ones((1, robot_count)),
        eye(task_count, format="csc"),
        format="csc",
    )
    robot_constraints = kron(
        eye(robot_count, format="csc"),
        np.ones((1, task_count)),
        format="csc",
    )
    matrix = vstack((task_constraints, robot_constraints), format="csc")

    lower = np.concatenate((np.ones(task_count), np.zeros(robot_count)))
    upper = np.ones(task_count + robot_count)
    constraint = LinearConstraint(matrix, lower, upper)

    variable_count = robot_count * task_count
    bounds = Bounds(np.zeros(variable_count), np.ones(variable_count))
    integrality = np.ones(variable_count, dtype=np.uint8)
    return constraint, bounds, integrality


def solve_milp_assignment(costs: np.ndarray) -> np.ndarray | None:
    """Solve capacity-one assignment; +inf edges are treated as unavailable."""
    if costs.ndim != 2 or costs.shape[0] < costs.shape[1]:
        fail("solve_milp_assignment", "contract", "INVALID_COST_MATRIX", f"shape={costs.shape}")
    if np.any(np.isnan(costs)) or np.any(np.isneginf(costs)):
        fail(
            "solve_milp_assignment",
            "data",
            "INVALID_MILP_COST",
            "cost matrix may contain finite values or +inf unavailable edges only",
        )

    finite = np.isfinite(costs)
    if not finite.any(axis=0).all():
        return None

    robot_count, task_count = costs.shape
    constraint, _, integrality = build_milp_assignment_model(robot_count, task_count)
    variable_count = robot_count * task_count
    bounds = Bounds(
        np.zeros(variable_count),
        finite.reshape(-1).astype(float),
    )
    objective = np.where(finite, costs, 0.0).reshape(-1)
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraint,
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        return None

    solution = result.x.reshape(robot_count, task_count)
    tasks = np.arange(task_count)
    assignment = np.argmax(solution, axis=0).astype(int)
    if np.any(solution[assignment, tasks] < 0.5):
        return None
    if len(np.unique(assignment)) != task_count:
        return None
    if np.any(~finite[assignment, tasks]):
        return None
    return assignment


def select_aco_candidates(
    costs: np.ndarray,
    available: np.ndarray,
    task: int,
    candidate_list_size: int,
) -> np.ndarray:
    """Return the cheapest available candidate robots for one ACO construction step."""
    candidates = np.flatnonzero(available & np.isfinite(costs[:, task]))
    if len(candidates) <= candidate_list_size:
        return candidates
    candidate_costs = costs[candidates, task]
    selected = np.argpartition(candidate_costs, candidate_list_size - 1)[:candidate_list_size]
    return candidates[selected]


def construct_aco_assignment(
    costs: np.ndarray,
    pheromone: np.ndarray,
    rng: np.random.Generator,
    config: ACOConfig,
) -> np.ndarray | None:
    """Construct one capacity-one assignment from pheromone and inverse-cost desirability."""
    robot_count, task_count = costs.shape
    if pheromone.shape != costs.shape:
        fail(
            "construct_aco_assignment",
            "contract",
            "PHEROMONE_SHAPE_MISMATCH",
            f"costs={costs.shape} pheromone={pheromone.shape}",
        )

    sorted_costs = np.sort(costs, axis=0)
    if robot_count > 1:
        regret = sorted_costs[1] - sorted_costs[0]
    else:
        regret = np.ones(task_count)
    task_order = np.argsort(-(regret + rng.random(task_count) * 1e-9))

    available = np.ones(robot_count, dtype=bool)
    assignment = np.full(task_count, -1, dtype=int)
    for task in task_order:
        candidates = select_aco_candidates(
            costs,
            available,
            int(task),
            config.candidate_list_size,
        )
        if len(candidates) == 0:
            return None

        trail = np.power(np.maximum(pheromone[candidates, task], 1e-15), config.alpha)
        heuristic = np.power(1.0 / np.maximum(costs[candidates, task], 1e-15), config.beta)
        weights = trail * heuristic
        total_weight = float(weights.sum())
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            chosen_index = int(rng.integers(len(candidates)))
        else:
            chosen_index = int(rng.choice(len(candidates), p=weights / total_weight))

        robot = int(candidates[chosen_index])
        assignment[task] = robot
        available[robot] = False
    return assignment


def improve_aco_assignment_locally(
    costs: np.ndarray,
    assignment: np.ndarray,
    max_moves: int,
) -> np.ndarray:
    """Refine an ACO solution by best improving unused-robot replacements or task-pair swaps."""
    if max_moves <= 0:
        return assignment.copy()

    robot_count, task_count = costs.shape
    improved = assignment.copy()
    tasks = np.arange(task_count)

    for _ in range(max_moves):
        used = np.zeros(robot_count, dtype=bool)
        used[improved] = True
        unused = np.flatnonzero(~used)

        best_replacement_delta = 0.0
        best_replacement: tuple[int, int] | None = None
        if len(unused):
            current = costs[improved, tasks]
            replacement_delta = costs[unused, :] - current[None, :]
            position = np.unravel_index(np.argmin(replacement_delta), replacement_delta.shape)
            candidate_delta = float(replacement_delta[position])
            if candidate_delta < best_replacement_delta:
                best_replacement_delta = candidate_delta
                best_replacement = (int(position[1]), int(unused[position[0]]))

        assigned_costs = costs[improved, :]
        base = assigned_costs[tasks, tasks]
        swap_delta = assigned_costs.T + assigned_costs - base[:, None] - base[None, :]
        swap_delta[np.tril_indices(task_count)] = np.inf
        swap_position = np.unravel_index(np.argmin(swap_delta), swap_delta.shape)
        best_swap_delta = (
            float(swap_delta[swap_position]) if np.isfinite(swap_delta[swap_position]) else 0.0
        )

        if (
            best_replacement is not None
            and best_replacement_delta <= best_swap_delta
            and best_replacement_delta < -1e-12
        ):
            task, robot = best_replacement
            improved[task] = robot
            continue

        if best_swap_delta < -1e-12:
            first_task, second_task = map(int, swap_position)
            improved[first_task], improved[second_task] = (
                improved[second_task],
                improved[first_task],
            )
            continue
        break

    return improved


def solve_aco_assignment(
    costs: np.ndarray,
    task_order: np.ndarray,
    rng: np.random.Generator,
    config: ACOConfig,
) -> np.ndarray | None:
    """Run ACO + local search; +inf edges are treated as unavailable."""
    if costs.ndim != 2 or costs.shape[0] < costs.shape[1]:
        fail("solve_aco_assignment", "contract", "INVALID_COST_MATRIX", f"shape={costs.shape}")
    if np.any(np.isnan(costs)) or np.any(np.isneginf(costs)):
        fail(
            "solve_aco_assignment",
            "data",
            "INVALID_ACO_COST",
            "cost matrix may contain finite values or +inf unavailable edges only",
        )
    if not np.isfinite(costs).any(axis=0).all():
        return None

    greedy_seed = solve_sequential_greedy(costs, task_order)
    robot_count, task_count = costs.shape
    tasks = np.arange(task_count)
    best_assignment: np.ndarray | None = None
    best_cost = np.inf
    if greedy_seed is not None:
        best_assignment = greedy_seed.copy()
        best_cost = assignment_total_cost(costs, best_assignment)
    pheromone = np.ones((robot_count, task_count), dtype=float)

    for _ in range(config.iterations):
        iteration_best: np.ndarray | None = None
        iteration_best_cost = np.inf
        for _ in range(config.ants):
            candidate = construct_aco_assignment(costs, pheromone, rng, config)
            if candidate is None:
                continue
            candidate_cost = assignment_total_cost(costs, candidate)
            if candidate_cost < iteration_best_cost:
                iteration_best = candidate
                iteration_best_cost = candidate_cost

        pheromone *= 1.0 - config.evaporation
        np.maximum(pheromone, 1e-12, out=pheromone)

        if iteration_best is not None:
            iteration_best = improve_aco_assignment_locally(
                costs,
                iteration_best,
                min(config.local_search_moves, task_count),
            )
            iteration_best_cost = assignment_total_cost(costs, iteration_best)
            if best_assignment is None or iteration_best_cost < best_cost:
                best_assignment = iteration_best.copy()
                best_cost = iteration_best_cost
            pheromone[iteration_best, tasks] += 1.0 / max(iteration_best_cost, 1e-12)

        if best_assignment is not None:
            pheromone[best_assignment, tasks] += config.elite_weight / max(best_cost, 1e-12)

    return best_assignment


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
    """Evaluate one optimizer result against the Hungarian exact reference."""
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
    """Require MILP to match Hungarian on representative complete-information cases."""
    rng = np.random.default_rng(seed + 188881)
    for task_count in (1, 5, 50, 100):
        costs = generate_spatial_cost_matrix(ROBOT_COUNT, task_count, rng)
        hungarian = solve_hungarian_assignment(costs)
        milp_assignment = solve_milp_assignment(costs)
        if hungarian is None or milp_assignment is None:
            fail(
                "validate_exact_optimizer_contract",
                "planning",
                "EXACT_OPTIMIZER_INFEASIBLE",
                f"tasks={task_count}",
            )
        hungarian_cost = assignment_total_cost(costs, hungarian)
        milp_cost = assignment_total_cost(costs, milp_assignment)
        gap_percent = 100.0 * (milp_cost - hungarian_cost) / hungarian_cost
        if abs(gap_percent) > MILP_NUMERICAL_TOLERANCE_PERCENT:
            fail(
                "validate_exact_optimizer_contract",
                "planning",
                "MILP_NOT_HUNGARIAN_EXACT",
                (
                    f"tasks={task_count} expected_abs_gap_percent<="
                    f"{MILP_NUMERICAL_TOLERANCE_PERCENT} actual={gap_percent}"
                ),
            )


def run_trial(
    *,
    task_count: int,
    trial: int,
    seed: int,
    aco_config: ACOConfig,
) -> list[dict[str, object]]:
    """Run one paired complete-information multi-task optimizer screening trial."""
    trial_seed = seed + task_count * 100003 + trial * 1009
    trial_rng = np.random.default_rng(trial_seed)
    costs = generate_spatial_cost_matrix(ROBOT_COUNT, task_count, trial_rng)
    task_order = trial_rng.permutation(task_count)

    start = time.perf_counter()
    hungarian = solve_hungarian_assignment(costs)
    hungarian_runtime_ms = 1000.0 * (time.perf_counter() - start)
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

    start = time.perf_counter()
    milp_assignment = solve_milp_assignment(costs)
    milp_runtime_ms = 1000.0 * (time.perf_counter() - start)
    if milp_assignment is None:
        fail("run_trial", "planning", "MILP_INFEASIBLE", f"tasks={task_count} trial={trial}")
    milp_cost = assignment_total_cost(costs, milp_assignment)
    milp_gap = 100.0 * (milp_cost - optimal_cost) / optimal_cost
    if abs(milp_gap) > MILP_NUMERICAL_TOLERANCE_PERCENT:
        fail(
            "run_trial",
            "planning",
            "MILP_NOT_EXACT",
            (
                f"tasks={task_count} trial={trial} expected_abs_gap_percent<="
                f"{MILP_NUMERICAL_TOLERANCE_PERCENT} actual={milp_gap}"
            ),
        )
    records.append(
        evaluate_method(
            task_count=task_count,
            trial=trial,
            method="milp",
            costs=costs,
            assignment=milp_assignment,
            optimal_assignment=hungarian,
            optimal_cost=optimal_cost,
            runtime_ms=milp_runtime_ms,
        )
    )

    aco_rng = np.random.default_rng(trial_seed + 7000003)
    start = time.perf_counter()
    aco_assignment = solve_aco_assignment(costs, task_order, aco_rng, aco_config)
    aco_runtime_ms = 1000.0 * (time.perf_counter() - start)
    if aco_assignment is None:
        fail("run_trial", "planning", "ACO_INFEASIBLE", f"tasks={task_count} trial={trial}")
    records.append(
        evaluate_method(
            task_count=task_count,
            trial=trial,
            method="aco_ls",
            costs=costs,
            assignment=aco_assignment,
            optimal_assignment=hungarian,
            optimal_cost=optimal_cost,
            runtime_ms=aco_runtime_ms,
        )
    )

    start = time.perf_counter()
    greedy_assignment = solve_sequential_greedy(costs, task_order)
    greedy_runtime_ms = 1000.0 * (time.perf_counter() - start)
    if greedy_assignment is None:
        fail("run_trial", "planning", "GREEDY_INFEASIBLE", f"tasks={task_count} trial={trial}")
    records.append(
        evaluate_method(
            task_count=task_count,
            trial=trial,
            method="greedy",
            costs=costs,
            assignment=greedy_assignment,
            optimal_assignment=hungarian,
            optimal_cost=optimal_cost,
            runtime_ms=greedy_runtime_ms,
        )
    )
    return records


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired optimizer quality and runtime metrics."""
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
    """Run the complete-information multi-task optimizer-family screening sweep."""
    validate_experiment_config(ROBOT_COUNT, 0.0, task_counts, trials)
    validate_aco_config(aco_config)
    validate_exact_optimizer_contract(seed)
    print("Exact optimizer contract: PASS (MILP matches Hungarian reference)")

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
    """Return one task-by-method report table in canonical display order."""
    table = summary.pivot(index="tasks", columns="method_label", values=metric)
    labels = [METHOD_LABELS[method] for method in METHODS]
    return table.reindex(columns=labels).reset_index()


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
    """Persist raw data, report tables, and screening plots."""
    ensure_output_dirs()
    raw.to_csv(DATA_DIR / "optimizer_screening_raw.csv", index=False)
    summary.to_csv(DATA_DIR / "optimizer_screening_summary.csv", index=False)

    metrics = {
        "average_optimality_gap_percent": "report_average_optimality_gap_percent.csv",
        "optimal_cost_match_percent": "report_optimal_cost_match_percent.csv",
        "near_optimal_5pct_percent": "report_near_optimal_5pct_percent.csv",
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
    if values is None:
        return TASK_COUNTS
    parsed = tuple(values)
    if not parsed:
        raise argparse.ArgumentTypeError("task counts cannot be empty")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen Hungarian, exact MILP, ACO with local refinement, and Greedy on paired "
            "complete-information multi-task assignment problems before integrating new "
            "optimizers into the lossy P2P Voting experiment."
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
    task_counts = parse_task_counts(args.tasks)
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
        task_counts=task_counts,
        trials=args.trials,
        seed=args.seed,
        aco_config=aco_config,
    )
    save_outputs(raw, summary)

    print("\nSaved:")
    print(DATA_DIR / "optimizer_screening_raw.csv")
    print(DATA_DIR / "optimizer_screening_summary.csv")
    print(FIGURE_DIR)

    print("\nAverage optimality gap (%) - lower is better:")
    print(report_table(summary, "average_optimality_gap_percent").to_string(index=False))

    print("\nOptimal-cost match (% of trials) - higher is better:")
    print(report_table(summary, "optimal_cost_match_percent").to_string(index=False))

    print("\nNear-optimal within 5% (% of trials) - higher is better:")
    print(report_table(summary, "near_optimal_5pct_percent").to_string(index=False))

    print("\nAverage optimizer runtime (ms) - lower is better:")
    print(report_table(summary, "average_runtime_ms").to_string(index=False))


if __name__ == "__main__":
    main()
