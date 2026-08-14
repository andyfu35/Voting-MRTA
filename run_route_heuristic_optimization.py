from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from run_algorithm_experiment import sample_active_robots
from run_decentralized_experiment import assign_sender_policies
from run_execution_simulation import (
    SERVICE_NOISE_SIGMA,
    TRAVEL_NOISE_SIGMA,
    build_base_cost,
    full_information_support,
    generate_scenario,
    lognormal_multipliers,
    simulate_execution,
    task_order_from_base_cost,
)
from run_multitask_experiment import per_task_probability
from run_route_guardrail_experiment import route_dict


RANDOM_SEED = 20260814
ROBOT_COUNTS = (10, 20, 40, 60)
TASK_LOAD_RATIOS = (0.50, 1.00, 1.50)
TRIALS = 40
ROUTE_GUARDRAIL_TOLERANCE = 0.20

METHODS = (
    "append_hardness",
    "append_regret",
    "insertion_hardness",
    "insertion_regret",
)
METHOD_LABELS = {
    "append_hardness": "Append + Hardness Order (Current)",
    "append_regret": "Append + Regret Order",
    "insertion_hardness": "Best Insertion + Hardness Order",
    "insertion_regret": "Best Insertion + Regret Order",
}

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "route_heuristic"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_outputs() -> None:
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def regret_task_order(base_cost: np.ndarray, active: np.ndarray) -> np.ndarray:
    active_cost = base_cost[active, :]
    task_ids = np.arange(base_cost.shape[1])
    best = active_cost.min(axis=0)
    if active_cost.shape[0] < 2:
        return np.lexsort((task_ids, -best))
    ordered = np.sort(active_cost, axis=0)
    regret = ordered[1, :] - ordered[0, :]
    # Larger regret first. If two tasks have the same regret, assign the task
    # with the worse best available cost first, then lower task ID.
    return np.lexsort((task_ids, -best, -regret))


def insertion_candidate(
    scenario: dict[str, np.ndarray],
    robot: int,
    route: list[int],
    task: int,
    current_duration: float,
) -> tuple[float, int]:
    start_time = scenario["start_travel_time"]
    transition_time = scenario["transition_travel_time"]
    service_times = scenario["service_times"]

    if not route:
        return float(start_time[robot, task] + service_times[task]), 0

    best_duration = np.inf
    best_position = 0
    for position in range(len(route) + 1):
        if position == 0:
            next_task = route[0]
            delta = (
                start_time[robot, task]
                + service_times[task]
                + transition_time[robot, task, next_task]
                - start_time[robot, next_task]
            )
        elif position == len(route):
            previous = route[-1]
            delta = transition_time[robot, previous, task] + service_times[task]
        else:
            previous = route[position - 1]
            next_task = route[position]
            delta = (
                transition_time[robot, previous, task]
                + service_times[task]
                + transition_time[robot, task, next_task]
                - transition_time[robot, previous, next_task]
            )

        candidate = current_duration + float(delta)
        if candidate < best_duration - 1e-12:
            best_duration = candidate
            best_position = position

    return float(best_duration), int(best_position)


def append_candidate(
    scenario: dict[str, np.ndarray],
    robot: int,
    route: list[int],
    task: int,
    current_duration: float,
) -> tuple[float, int]:
    if route:
        previous = route[-1]
        travel = scenario["transition_travel_time"][robot, previous, task]
    else:
        travel = scenario["start_travel_time"][robot, task]
    duration = current_duration + travel + scenario["service_times"][task]
    return float(duration), len(route)


def optimized_route_plan(
    scenario: dict[str, np.ndarray],
    base_cost: np.ndarray,
    active: np.ndarray,
    support: np.ndarray,
    *,
    order_mode: str,
    insertion_mode: str,
) -> tuple[tuple[int, ...], ...]:
    n, _ = base_cost.shape
    active_indices = np.flatnonzero(active)
    routes: list[list[int]] = [[] for _ in range(n)]
    route_duration = np.zeros(n, dtype=float)

    if order_mode == "hardness":
        task_order = task_order_from_base_cost(base_cost, active)
    elif order_mode == "regret":
        task_order = regret_task_order(base_cost, active)
    else:
        raise ValueError(f"Unknown task order: {order_mode}")

    candidate_fn = insertion_candidate if insertion_mode == "best_insertion" else append_candidate

    for task in task_order:
        projected = np.empty(len(active_indices), dtype=float)
        positions = np.empty(len(active_indices), dtype=int)

        for local_index, robot in enumerate(active_indices):
            projected[local_index], positions[local_index] = candidate_fn(
                scenario,
                int(robot),
                routes[int(robot)],
                int(task),
                float(route_duration[int(robot)]),
            )

        best_projected = float(projected.min())
        threshold = best_projected * (1.0 + ROUTE_GUARDRAIL_TOLERANCE) + 1e-12
        eligible = np.flatnonzero(projected <= threshold)

        preference = support[active_indices, int(task)]
        eligible_preference = preference[eligible]
        best_preference = float(eligible_preference.max())
        preferred = eligible[
            np.flatnonzero(np.isclose(eligible_preference, best_preference, atol=1e-12))
        ]

        tie_duration = projected[preferred]
        best_duration = float(tie_duration.min())
        finalists = preferred[
            np.flatnonzero(np.isclose(tie_duration, best_duration, atol=1e-12))
        ]
        chosen_local = int(finalists[0])
        chosen_robot = int(active_indices[chosen_local])
        insert_at = int(positions[chosen_local])

        routes[chosen_robot].insert(insert_at, int(task))
        route_duration[chosen_robot] = projected[chosen_local]

    return tuple(tuple(route) for route in routes)


def method_config(method: str) -> tuple[str, str]:
    order = "regret" if method.endswith("regret") else "hardness"
    insertion = "best_insertion" if method.startswith("insertion") else "append"
    return order, insertion


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []

    for n in ROBOT_COUNTS:
        for load_index, load_ratio in enumerate(TASK_LOAD_RATIOS):
            for trial in range(TRIALS):
                rng = np.random.default_rng(
                    RANDOM_SEED + n * 100000 + load_index * 10000 + trial
                )
                active = sample_active_robots(n, rng)
                active_count = int(active.sum())
                task_count = max(1, int(round(active_count * load_ratio)))

                scenario = generate_scenario(n, task_count, rng)
                base_cost = build_base_cost(scenario, active)
                sender_policy = assign_sender_policies(
                    active,
                    "heterogeneous_strong",
                    rng,
                )
                used_policies = sorted(set(sender_policy.values()))
                probability_map = {
                    policy: per_task_probability(base_cost, active, policy)
                    for policy in used_policies
                }
                support = full_information_support(sender_policy, probability_map)

                start_time_multiplier = lognormal_multipliers(
                    rng,
                    TRAVEL_NOISE_SIGMA,
                    (n, task_count),
                )
                transition_time_multiplier = lognormal_multipliers(
                    rng,
                    TRAVEL_NOISE_SIGMA,
                    (task_count, task_count),
                )
                np.fill_diagonal(transition_time_multiplier, 1.0)
                service_time_multiplier = lognormal_multipliers(
                    rng,
                    SERVICE_NOISE_SIGMA,
                    (task_count,),
                )
                success_uniform = rng.random((n, task_count))

                for method in METHODS:
                    order_mode, insertion_mode = method_config(method)
                    plan = optimized_route_plan(
                        scenario,
                        base_cost,
                        active,
                        support,
                        order_mode=order_mode,
                        insertion_mode=insertion_mode,
                    )
                    execution = simulate_execution(
                        route_dict(plan, active),
                        active,
                        scenario,
                        start_time_multiplier,
                        transition_time_multiplier,
                        service_time_multiplier,
                        success_uniform,
                    )
                    completed = float(execution["task_completion_rate"]) * task_count
                    energy_per_completed = (
                        float(execution["total_energy_units"]) / completed
                        if completed > 1e-12
                        else np.nan
                    )
                    records.append(
                        {
                            "robots": n,
                            "active_robots": active_count,
                            "tasks": task_count,
                            "task_load_ratio": load_ratio,
                            "trial": trial + 1,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "route_guardrail_tolerance": ROUTE_GUARDRAIL_TOLERANCE,
                            "energy_per_completed_task": energy_per_completed,
                            **execution,
                        }
                    )

    raw = pd.DataFrame.from_records(records)
    summary = (
        raw.groupby(["robots", "task_load_ratio", "method", "method_label"], as_index=False)
        .agg(
            task_completion_rate=("task_completion_rate", "mean"),
            mission_success_rate=("mission_success", "mean"),
            actual_makespan_sec=("actual_makespan_sec", "mean"),
            energy_per_completed_task=("energy_per_completed_task", "mean"),
            total_travel_distance_m=("total_travel_distance_m", "mean"),
            route_task_count_cv=("route_task_count_cv", "mean"),
            eta_mape=("mean_eta_absolute_percentage_error", "mean"),
            duplicate_task_execution_rate=("duplicate_task_execution_rate", "mean"),
        )
    )
    overall = (
        raw.groupby(["method", "method_label"], as_index=False)
        .agg(
            task_completion_rate=("task_completion_rate", "mean"),
            mission_success_rate=("mission_success", "mean"),
            actual_makespan_sec=("actual_makespan_sec", "mean"),
            energy_per_completed_task=("energy_per_completed_task", "mean"),
            total_travel_distance_m=("total_travel_distance_m", "mean"),
            route_task_count_cv=("route_task_count_cv", "mean"),
            eta_mape=("mean_eta_absolute_percentage_error", "mean"),
            duplicate_task_execution_rate=("duplicate_task_execution_rate", "mean"),
        )
    )
    return raw, summary, overall


def plot_by_load(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    *,
    percent: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(METHODS))
    width = 0.24
    for index, load_ratio in enumerate(TASK_LOAD_RATIOS):
        values = []
        subset = summary[np.isclose(summary["task_load_ratio"], load_ratio)]
        for method in METHODS:
            values.append(float(subset[subset["method"] == method][metric].mean()))
        ax.bar(
            x + (index - 1) * width,
            values,
            width=width,
            label=f"Task load {load_ratio:.0%}",
        )
    ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close(fig)


def plot_tradeoff(overall: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in overall.iterrows():
        ax.scatter(row["actual_makespan_sec"], row["energy_per_completed_task"], s=90)
        ax.annotate(
            str(row["method_label"]),
            (row["actual_makespan_sec"], row["energy_per_completed_task"]),
            xytext=(6, 6),
            textcoords="offset points",
        )
    ax.set_xlabel("Average Actual Makespan (s)")
    ax.set_ylabel("Energy per Completed Task")
    ax.set_title("Route Heuristic Trade-off: Makespan vs Energy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "route_heuristic_tradeoff.png", dpi=180)
    plt.close(fig)


def save_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    raw.to_csv(DATA_DIR / "route_heuristic_raw_results.csv", index=False)
    summary.to_csv(DATA_DIR / "route_heuristic_summary_results.csv", index=False)
    overall.to_csv(DATA_DIR / "route_heuristic_by_method.csv", index=False)

    plot_by_load(
        summary,
        "actual_makespan_sec",
        "Average Actual Makespan (s)",
        "Can Regret Ordering and Best Insertion Shorten the Mission?",
        "route_heuristic_makespan.png",
    )
    plot_by_load(
        summary,
        "energy_per_completed_task",
        "Energy per Completed Task",
        "Route-Heuristic Energy Efficiency",
        "route_heuristic_energy_per_task.png",
    )
    plot_by_load(
        summary,
        "route_task_count_cv",
        "Route Task-Count CV",
        "Route-Heuristic Load Balance",
        "route_heuristic_load_balance.png",
    )
    plot_by_load(
        summary,
        "task_completion_rate",
        "Task Completion Rate",
        "Task Completion under Route-Heuristic Optimization",
        "route_heuristic_completion_rate.png",
        percent=True,
    )
    plot_by_load(
        summary,
        "mission_success_rate",
        "All-Tasks Mission Success Rate",
        "Mission Reliability under Route-Heuristic Optimization",
        "route_heuristic_mission_success.png",
        percent=True,
    )
    plot_tradeoff(overall)


def main() -> None:
    ensure_output_dirs()
    clear_old_outputs()
    raw, summary, overall = run_experiment()
    save_outputs(raw, summary, overall)

    columns = [
        "method_label",
        "task_completion_rate",
        "mission_success_rate",
        "actual_makespan_sec",
        "energy_per_completed_task",
        "total_travel_distance_m",
        "route_task_count_cv",
        "duplicate_task_execution_rate",
    ]
    print(overall[columns].to_string(index=False))


if __name__ == "__main__":
    main()
