from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from run_algorithm_experiment import sample_active_robots
from run_decentralized_experiment import assign_sender_policies
from run_execution_simulation import (
    PREFERENCE_POLICIES,
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


RANDOM_SEED = 20260813
ROBOT_COUNTS = (10, 20, 40, 60)
TASK_LOAD_RATIOS = (0.50, 1.00, 1.50)
TRIALS = 40

# The main communication stress setting remains 30% packet loss with three
# stop-on-success attempts. This experiment intentionally removes communication
# from the sweep so it isolates the route-balancing decision itself. Once a
# guardrail is selected, it can be validated under the fixed 30% setting in the
# decentralized execution experiment.
REFERENCE_PACKET_LOSS = 0.30
REFERENCE_MAX_ATTEMPTS = 3

ROUTE_TOLERANCES = (0.00, 0.05, 0.10, 0.20, 0.50, np.inf)
TOLERANCE_LABELS = {
    0.00: "0%",
    0.05: "5%",
    0.10: "10%",
    0.20: "20%",
    0.50: "50%",
    np.inf: "No guardrail",
}

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "route_guardrail"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_outputs() -> None:
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def guardrail_route_plan(
    scenario: dict[str, np.ndarray],
    base_cost: np.ndarray,
    active: np.ndarray,
    support: np.ndarray,
    tolerance: float,
) -> tuple[tuple[int, ...], ...]:
    """Build one-shot routes using route finish as a guardrail around preference.

    For each task, first compute the projected finish time if the task is appended
    to every active robot's current route. Only robots whose projected finish is
    within (1 + tolerance) of the best projected finish remain eligible. Among
    those eligible robots, select the candidate with the strongest complete
    voting preference. Ties are broken by the lower projected finish time, then
    lower robot index for determinism.

    tolerance = 0     -> effectively route-greedy assignment
    tolerance = inf   -> pure preference assignment with no route protection
    """
    n, _ = base_cost.shape
    active_indices = np.flatnonzero(active)
    routes: list[list[int]] = [[] for _ in range(n)]
    projected_finish = np.zeros(n, dtype=float)

    start_time = scenario["start_travel_time"]
    transition_time = scenario["transition_travel_time"]
    service_times = scenario["service_times"]

    for task in task_order_from_base_cost(base_cost, active):
        projected = np.empty(len(active_indices), dtype=float)

        for local_index, robot in enumerate(active_indices):
            if routes[robot]:
                previous = routes[robot][-1]
                edge_time = transition_time[robot, previous, task]
            else:
                edge_time = start_time[robot, task]
            projected[local_index] = (
                projected_finish[robot] + edge_time + service_times[task]
            )

        best_projected = float(projected.min())
        if np.isinf(tolerance):
            eligible_local = np.arange(len(active_indices), dtype=int)
        else:
            threshold = best_projected * (1.0 + tolerance) + 1e-12
            eligible_local = np.flatnonzero(projected <= threshold)

        preference = support[active_indices, task]
        eligible_preference = preference[eligible_local]
        best_preference = float(eligible_preference.max())
        preferred_local = eligible_local[
            np.flatnonzero(
                np.isclose(eligible_preference, best_preference, atol=1e-12)
            )
        ]

        # Preference ties are resolved by projected finish time, then robot ID.
        tie_finish = projected[preferred_local]
        best_finish = float(tie_finish.min())
        finish_local = preferred_local[
            np.flatnonzero(np.isclose(tie_finish, best_finish, atol=1e-12))
        ]
        chosen_local = int(finish_local[0])
        chosen_robot = int(active_indices[chosen_local])

        routes[chosen_robot].append(int(task))
        projected_finish[chosen_robot] = projected[chosen_local]

    return tuple(tuple(route) for route in routes)


def route_dict(
    plan: tuple[tuple[int, ...], ...],
    active: np.ndarray,
) -> dict[int, tuple[int, ...]]:
    return {int(robot): plan[int(robot)] for robot in np.flatnonzero(active)}


def make_record(
    *,
    robots: int,
    active_robots: int,
    tasks: int,
    load_ratio: float,
    trial: int,
    tolerance: float,
    execution: dict[str, float | bool],
) -> dict[str, object]:
    completed_tasks = float(execution["task_completion_rate"]) * tasks
    energy_per_completed = (
        float(execution["total_energy_units"]) / completed_tasks
        if completed_tasks > 1e-12
        else np.nan
    )
    complete_mission_makespan = (
        float(execution["actual_makespan_sec"])
        if bool(execution["mission_success"])
        else np.nan
    )

    return {
        "robots": robots,
        "active_robots": active_robots,
        "tasks": tasks,
        "task_load_ratio": load_ratio,
        "trial": trial,
        "route_tolerance": tolerance,
        "route_tolerance_label": TOLERANCE_LABELS[tolerance],
        "reference_packet_loss": REFERENCE_PACKET_LOSS,
        "reference_max_attempts": REFERENCE_MAX_ATTEMPTS,
        "energy_per_completed_task": energy_per_completed,
        "complete_mission_makespan_sec": complete_mission_makespan,
        **execution,
    }


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

                # Paired execution uncertainty: every tolerance sees the same
                # scenario, travel/service noise, and success random numbers.
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

                for tolerance in ROUTE_TOLERANCES:
                    plan = guardrail_route_plan(
                        scenario,
                        base_cost,
                        active,
                        support,
                        tolerance,
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
                    records.append(
                        make_record(
                            robots=n,
                            active_robots=active_count,
                            tasks=task_count,
                            load_ratio=load_ratio,
                            trial=trial + 1,
                            tolerance=tolerance,
                            execution=execution,
                        )
                    )

    raw = pd.DataFrame.from_records(records)
    summary = (
        raw.groupby(
            [
                "robots",
                "task_load_ratio",
                "route_tolerance",
                "route_tolerance_label",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            task_completion_rate=("task_completion_rate", "mean"),
            mission_success_rate=("mission_success", "mean"),
            duplicate_task_execution_rate=("duplicate_task_execution_rate", "mean"),
            unexecuted_task_rate=("unexecuted_task_rate", "mean"),
            actual_makespan_sec=("actual_makespan_sec", "mean"),
            complete_mission_makespan_sec=("complete_mission_makespan_sec", "mean"),
            total_energy_units=("total_energy_units", "mean"),
            energy_per_completed_task=("energy_per_completed_task", "mean"),
            total_travel_distance_m=("total_travel_distance_m", "mean"),
            route_task_count_cv=("route_task_count_cv", "mean"),
            eta_mape=("mean_eta_absolute_percentage_error", "mean"),
        )
        .reset_index(drop=True)
    )

    by_tolerance = (
        raw.groupby(
            ["route_tolerance", "route_tolerance_label"],
            as_index=False,
            dropna=False,
        )
        .agg(
            task_completion_rate=("task_completion_rate", "mean"),
            mission_success_rate=("mission_success", "mean"),
            duplicate_task_execution_rate=("duplicate_task_execution_rate", "mean"),
            actual_makespan_sec=("actual_makespan_sec", "mean"),
            complete_mission_makespan_sec=("complete_mission_makespan_sec", "mean"),
            total_energy_units=("total_energy_units", "mean"),
            energy_per_completed_task=("energy_per_completed_task", "mean"),
            total_travel_distance_m=("total_travel_distance_m", "mean"),
            route_task_count_cv=("route_task_count_cv", "mean"),
        )
        .reset_index(drop=True)
    )
    return raw, summary, by_tolerance


def ordered_labels(frame: pd.DataFrame) -> list[str]:
    return [TOLERANCE_LABELS[value] for value in ROUTE_TOLERANCES]


def plot_metric_by_load(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    percent: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(ROUTE_TOLERANCES))

    for load_ratio in TASK_LOAD_RATIOS:
        subset = summary[np.isclose(summary["task_load_ratio"], load_ratio)]
        values = []
        for tolerance in ROUTE_TOLERANCES:
            row = subset[
                np.isclose(subset["route_tolerance"], tolerance)
                if np.isfinite(tolerance)
                else np.isinf(subset["route_tolerance"])
            ]
            values.append(float(row[metric].mean()))
        ax.plot(x, values, marker="o", label=f"Task load {load_ratio:.0%}")

    ax.set_xticks(x, ordered_labels(summary))
    ax.set_xlabel("Route Guardrail Tolerance")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close(fig)


def plot_tradeoff(by_tolerance: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in by_tolerance.iterrows():
        ax.scatter(
            row["actual_makespan_sec"],
            row["energy_per_completed_task"],
            s=90,
        )
        ax.annotate(
            str(row["route_tolerance_label"]),
            (row["actual_makespan_sec"], row["energy_per_completed_task"]),
            xytext=(6, 6),
            textcoords="offset points",
        )
    ax.set_xlabel("Average Actual Makespan (s)")
    ax.set_ylabel("Energy per Completed Task")
    ax.set_title("Route Guardrail Trade-off: Mission Time vs Energy Efficiency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "route_guardrail_tradeoff.png", dpi=180)
    plt.close(fig)


def save_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    by_tolerance: pd.DataFrame,
) -> None:
    raw.to_csv(DATA_DIR / "route_guardrail_raw_results.csv", index=False)
    summary.to_csv(DATA_DIR / "route_guardrail_summary_results.csv", index=False)
    by_tolerance.to_csv(DATA_DIR / "route_guardrail_by_tolerance.csv", index=False)

    plot_metric_by_load(
        summary,
        "actual_makespan_sec",
        "Average Actual Makespan (s)",
        "Does a Route Guardrail Prevent Preference-Induced Mission Delay?",
        "route_guardrail_makespan.png",
    )
    plot_metric_by_load(
        summary,
        "energy_per_completed_task",
        "Energy per Completed Task",
        "Energy Efficiency after Normalizing for Completed Work",
        "route_guardrail_energy_per_completed_task.png",
    )
    plot_metric_by_load(
        summary,
        "route_task_count_cv",
        "Route Task-Count CV",
        "Does the Guardrail Improve Task-Load Balance?",
        "route_guardrail_load_balance.png",
    )
    plot_metric_by_load(
        summary,
        "task_completion_rate",
        "Task Completion Rate",
        "Task Completion under Route Guardrail Tuning",
        "route_guardrail_completion_rate.png",
        percent=True,
    )
    plot_metric_by_load(
        summary,
        "mission_success_rate",
        "All-Tasks Mission Success Rate",
        "Mission Reliability under Route Guardrail Tuning",
        "route_guardrail_mission_success.png",
        percent=True,
    )
    plot_tradeoff(by_tolerance)


def main() -> None:
    ensure_output_dirs()
    clear_old_outputs()
    raw, summary, by_tolerance = run_experiment()
    save_outputs(raw, summary, by_tolerance)

    columns = [
        "route_tolerance_label",
        "task_completion_rate",
        "mission_success_rate",
        "actual_makespan_sec",
        "energy_per_completed_task",
        "route_task_count_cv",
        "duplicate_task_execution_rate",
    ]
    print(by_tolerance[columns].to_string(index=False))


if __name__ == "__main__":
    main()
