from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from run_algorithm_experiment import sample_active_robots
from run_decentralized_experiment import assign_sender_policies
from run_execution_simulation import (
    BASE_COST_FLOOR,
    BASE_ENERGY_WEIGHT,
    BASE_RISK_WEIGHT,
    BASE_TIME_WEIGHT,
    SERVICE_ENERGY_PER_SECOND,
    SERVICE_NOISE_SIGMA,
    TRAVEL_NOISE_SIGMA,
    lognormal_multipliers,
)
from run_hierarchical_communication_experiment import (
    communication_time_proxy_ms,
    flat_preference_exchange,
    hierarchical_preference_exchange,
    preference_payload_bytes,
    probability_support_from_counts,
    sender_policy_one_hot,
)
from run_end_to_end_optimized_experiment import (
    disseminate_qc,
    proposal_quorum_exchange,
)
from run_multitask_experiment import per_task_probability


RANDOM_SEED = 20260815
ROBOT_COUNT = 100
TASK_LOAD_RATIOS = (1.00, 3.00, 5.00, 7.50, 10.00)
TRIALS = 10
ROUTE_GUARDRAIL_TOLERANCE = 0.20
WORKSPACE_SIZE_M = 100.0

METHODS = (
    "centralized_route_greedy",
    "full_info_current",
    "full_info_optimized",
    "hierarchy_backup_optimized",
)
METHOD_LABELS = {
    "centralized_route_greedy": "Centralized Route Greedy",
    "full_info_current": "Full-Info Current Route",
    "full_info_optimized": "Full-Info Optimized",
    "hierarchy_backup_optimized": "5-ary Backup + Optimized",
}

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "high_load_100robots"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_outputs() -> None:
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Memory-safe Euclidean distance matrix without an N x M x 2 temporary."""
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    squared = a2 + b2 - 2.0 * (a @ b.T)
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared)


def generate_compact_scenario(
    n: int,
    task_count: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate the execution scenario without materializing N x M x M travel time."""
    robot_positions = rng.uniform(0.0, WORKSPACE_SIZE_M, size=(n, 2))
    task_positions = rng.uniform(0.0, WORKSPACE_SIZE_M, size=(task_count, 2))
    robot_speeds = rng.uniform(0.9, 1.3, size=n)
    energy_per_meter = rng.uniform(0.8, 1.2, size=n)
    robot_reliability = rng.uniform(0.992, 0.999, size=n)
    service_times = rng.uniform(20.0, 60.0, size=task_count)
    task_success_factor = rng.uniform(0.995, 1.0, size=task_count)

    start_distance = pairwise_distances(robot_positions, task_positions)
    task_distance = pairwise_distances(task_positions, task_positions)
    np.fill_diagonal(task_distance, 0.0)
    start_travel_time = start_distance / robot_speeds[:, None]
    start_energy = start_distance * energy_per_meter[:, None]

    return {
        "robot_positions": robot_positions,
        "task_positions": task_positions,
        "robot_speeds": robot_speeds,
        "energy_per_meter": energy_per_meter,
        "robot_reliability": robot_reliability,
        "service_times": service_times,
        "task_success_factor": task_success_factor,
        "start_distance": start_distance,
        "task_distance": task_distance,
        "start_travel_time": start_travel_time,
        "start_energy": start_energy,
    }


def minmax_scale_by_task(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    scaled = np.zeros_like(values, dtype=float)
    active_values = values[active, :]
    low = active_values.min(axis=0)
    high = active_values.max(axis=0)
    span = high - low
    safe = np.where(span > 1e-12, span, 1.0)
    scaled[active, :] = (active_values - low) / safe
    return scaled


def build_compact_base_cost(
    scenario: dict[str, np.ndarray],
    active: np.ndarray,
) -> np.ndarray:
    time_scaled = minmax_scale_by_task(scenario["start_travel_time"], active)
    energy_scaled = minmax_scale_by_task(scenario["start_energy"], active)

    reliability = scenario["robot_reliability"]
    risk = 1.0 - reliability
    active_risk = risk[active]
    risk_low = active_risk.min()
    risk_high = active_risk.max()
    span = risk_high - risk_low
    if span <= 1e-12:
        risk_scaled = np.zeros_like(risk)
    else:
        risk_scaled = (risk - risk_low) / span

    return (
        BASE_COST_FLOOR
        + BASE_TIME_WEIGHT * time_scaled
        + BASE_ENERGY_WEIGHT * energy_scaled
        + BASE_RISK_WEIGHT * risk_scaled[:, None]
    )


def transition_time(
    scenario: dict[str, np.ndarray],
    robot: int,
    previous: int,
    task: int,
) -> float:
    return float(
        scenario["task_distance"][previous, task]
        / scenario["robot_speeds"][robot]
    )


def hardness_task_order(base_cost: np.ndarray, active: np.ndarray) -> np.ndarray:
    active_cost = base_cost[active, :]
    hardness = active_cost.min(axis=0)
    task_ids = np.arange(base_cost.shape[1])
    return np.lexsort((task_ids, -hardness))


def regret_task_order(base_cost: np.ndarray, active: np.ndarray) -> np.ndarray:
    active_cost = base_cost[active, :]
    task_ids = np.arange(base_cost.shape[1])
    best = active_cost.min(axis=0)
    if active_cost.shape[0] < 2:
        return np.lexsort((task_ids, -best))
    ordered = np.sort(active_cost, axis=0)
    regret = ordered[1, :] - ordered[0, :]
    return np.lexsort((task_ids, -best, -regret))


def append_candidate(
    scenario: dict[str, np.ndarray],
    robot: int,
    route: list[int],
    task: int,
    current_duration: float,
) -> tuple[float, int]:
    if route:
        travel = transition_time(scenario, robot, route[-1], task)
    else:
        travel = float(scenario["start_travel_time"][robot, task])
    return (
        float(current_duration + travel + scenario["service_times"][task]),
        len(route),
    )


def best_insertion_candidate(
    scenario: dict[str, np.ndarray],
    robot: int,
    route: list[int],
    task: int,
    current_duration: float,
) -> tuple[float, int]:
    service = float(scenario["service_times"][task])
    if not route:
        return (
            float(scenario["start_travel_time"][robot, task] + service),
            0,
        )

    best_duration = np.inf
    best_position = 0
    for position in range(len(route) + 1):
        if position == 0:
            next_task = route[0]
            delta = (
                float(scenario["start_travel_time"][robot, task])
                + service
                + transition_time(scenario, robot, task, next_task)
                - float(scenario["start_travel_time"][robot, next_task])
            )
        elif position == len(route):
            previous = route[-1]
            delta = transition_time(scenario, robot, previous, task) + service
        else:
            previous = route[position - 1]
            next_task = route[position]
            delta = (
                transition_time(scenario, robot, previous, task)
                + service
                + transition_time(scenario, robot, task, next_task)
                - transition_time(scenario, robot, previous, next_task)
            )

        candidate = current_duration + delta
        if candidate < best_duration - 1e-12:
            best_duration = candidate
            best_position = position

    return float(best_duration), int(best_position)


def build_route_plan(
    scenario: dict[str, np.ndarray],
    base_cost: np.ndarray,
    active: np.ndarray,
    support: np.ndarray | None,
    *,
    order_mode: str,
    insertion_mode: str,
    guardrail_tolerance: float,
) -> tuple[tuple[int, ...], ...]:
    active_indices = np.flatnonzero(active)
    routes: list[list[int]] = [[] for _ in range(len(active))]
    route_duration = np.zeros(len(active), dtype=float)

    task_order = (
        regret_task_order(base_cost, active)
        if order_mode == "regret"
        else hardness_task_order(base_cost, active)
    )
    candidate_fn = (
        best_insertion_candidate
        if insertion_mode == "best_insertion"
        else append_candidate
    )

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
        if support is None:
            chosen_local = int(np.flatnonzero(np.isclose(projected, best_projected, atol=1e-12))[0])
        else:
            threshold = best_projected * (1.0 + guardrail_tolerance) + 1e-12
            eligible = np.flatnonzero(projected <= threshold)
            preference = support[active_indices, int(task)]
            eligible_preference = preference[eligible]
            best_preference = float(eligible_preference.max())
            preferred = eligible[
                np.flatnonzero(
                    np.isclose(eligible_preference, best_preference, atol=1e-12)
                )
            ]
            tie_projected = projected[preferred]
            best_tie = float(tie_projected.min())
            finalists = preferred[
                np.flatnonzero(np.isclose(tie_projected, best_tie, atol=1e-12))
            ]
            chosen_local = int(finalists[0])

        chosen_robot = int(active_indices[chosen_local])
        routes[chosen_robot].insert(int(positions[chosen_local]), int(task))
        route_duration[chosen_robot] = projected[chosen_local]

    return tuple(tuple(route) for route in routes)


def route_dict(
    plan: tuple[tuple[int, ...], ...],
    active: np.ndarray,
) -> dict[int, tuple[int, ...]]:
    return {int(robot): plan[int(robot)] for robot in np.flatnonzero(active)}


def simulate_compact_execution(
    robot_routes: dict[int, tuple[int, ...]],
    active: np.ndarray,
    scenario: dict[str, np.ndarray],
    start_time_multiplier: np.ndarray,
    transition_time_multiplier: np.ndarray,
    service_time_multiplier: np.ndarray,
    success_uniform: np.ndarray,
) -> dict[str, float]:
    task_count = len(scenario["service_times"])
    task_attempt_count = np.zeros(task_count, dtype=int)
    task_completed = np.zeros(task_count, dtype=bool)
    actual_finish_times: list[float] = []
    total_distance = 0.0
    total_energy = 0.0
    successful_attempts = 0
    total_attempts = 0
    per_robot_task_counts: list[int] = []

    for robot in np.flatnonzero(active):
        route = robot_routes.get(int(robot), ())
        per_robot_task_counts.append(len(route))
        actual_clock = 0.0
        previous: int | None = None

        for task in route:
            task_attempt_count[task] += 1
            total_attempts += 1
            if previous is None:
                est_travel = float(scenario["start_travel_time"][robot, task])
                actual_travel = est_travel * start_time_multiplier[robot, task]
                distance = float(scenario["start_distance"][robot, task])
            else:
                est_travel = transition_time(scenario, int(robot), previous, int(task))
                actual_travel = est_travel * transition_time_multiplier[previous, task]
                distance = float(scenario["task_distance"][previous, task])

            actual_service = (
                float(scenario["service_times"][task])
                * service_time_multiplier[task]
            )
            actual_clock += float(actual_travel + actual_service)

            success_probability = (
                scenario["robot_reliability"][robot]
                * scenario["task_success_factor"][task]
            )
            if success_uniform[robot, task] < success_probability:
                task_completed[task] = True
                successful_attempts += 1

            total_distance += distance
            total_energy += (
                distance * scenario["energy_per_meter"][robot]
                + actual_service * SERVICE_ENERGY_PER_SECOND
            )
            previous = int(task)

        actual_finish_times.append(actual_clock)

    attempted_tasks = int(np.count_nonzero(task_attempt_count > 0))
    duplicate_tasks = int(np.count_nonzero(task_attempt_count > 1))
    completed_tasks = int(np.count_nonzero(task_completed))
    coverage = attempted_tasks / task_count if task_count else 1.0
    completion_rate = completed_tasks / task_count if task_count else 1.0
    duplicate_rate = duplicate_tasks / task_count if task_count else 0.0

    counts = np.asarray(per_robot_task_counts, dtype=float)
    mean_count = float(counts.mean()) if len(counts) else 0.0
    task_count_cv = float(counts.std() / mean_count) if mean_count > 1e-12 else 0.0

    finishes = np.asarray(actual_finish_times, dtype=float)
    mean_finish = float(finishes.mean()) if len(finishes) else 0.0
    finish_time_cv = float(finishes.std() / mean_finish) if mean_finish > 1e-12 else 0.0
    makespan = float(finishes.max()) if len(finishes) else np.nan

    return {
        "task_execution_coverage": float(coverage),
        "task_completion_rate": float(completion_rate),
        "mission_success_rate": float(completed_tasks == task_count and duplicate_tasks == 0),
        "duplicate_task_execution_rate": float(duplicate_rate),
        "unexecuted_task_rate": float(1.0 - coverage),
        "attempt_success_rate": float(successful_attempts / total_attempts if total_attempts else 0.0),
        "actual_makespan_sec": makespan,
        "total_travel_distance_m": float(total_distance),
        "total_energy_units": float(total_energy),
        "route_task_count_cv": task_count_cv,
        "route_finish_time_cv": finish_time_cv,
    }


def execution_record(
    plan_or_routes: tuple[tuple[int, ...], ...] | dict[int, tuple[int, ...]],
    active: np.ndarray,
    scenario: dict[str, np.ndarray],
    start_time_multiplier: np.ndarray,
    transition_time_multiplier: np.ndarray,
    service_time_multiplier: np.ndarray,
    success_uniform: np.ndarray,
) -> dict[str, float]:
    routes = (
        route_dict(plan_or_routes, active)
        if isinstance(plan_or_routes, tuple)
        else plan_or_routes
    )
    result = simulate_compact_execution(
        routes,
        active,
        scenario,
        start_time_multiplier,
        transition_time_multiplier,
        service_time_multiplier,
        success_uniform,
    )
    completed = result["task_completion_rate"] * len(scenario["service_times"])
    result["energy_per_completed_task"] = (
        result["total_energy_units"] / completed if completed > 1e-12 else np.nan
    )
    result["travel_distance_per_completed_task_m"] = (
        result["total_travel_distance_m"] / completed if completed > 1e-12 else np.nan
    )
    result["makespan_per_task_sec"] = (
        result["actual_makespan_sec"] / len(scenario["service_times"])
    )
    return result


def make_record(
    *,
    load_ratio: float,
    trial: int,
    active_count: int,
    task_count: int,
    method: str,
    planning_runtime_sec: float,
    execution: dict[str, float],
    safe_commit_rate: float = 1.0,
    plan_match_reference_rate: float = 1.0,
    qc_delivery_rate: float = 1.0,
    message_attempts: float = np.nan,
    bytes_transmitted: float = np.nan,
    communication_stages: float = np.nan,
    latency_proxy_ms: float = np.nan,
    flat_reference_attempts: float = np.nan,
    flat_reference_bytes: float = np.nan,
    flat_reference_latency_proxy_ms: float = np.nan,
) -> dict[str, object]:
    return {
        "robots": ROBOT_COUNT,
        "active_robots": active_count,
        "task_load_ratio": load_ratio,
        "task_load_percent": load_ratio * 100.0,
        "tasks": task_count,
        "trial": trial,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "planning_runtime_sec": planning_runtime_sec,
        "safe_commit_rate": safe_commit_rate,
        "plan_match_reference_rate": plan_match_reference_rate,
        "qc_delivery_rate": qc_delivery_rate,
        "message_attempts": message_attempts,
        "bytes_transmitted": bytes_transmitted,
        "communication_stages": communication_stages,
        "shared_channel_latency_proxy_ms": latency_proxy_ms,
        "flat_reference_attempts": flat_reference_attempts,
        "flat_reference_bytes": flat_reference_bytes,
        "flat_reference_latency_proxy_ms": flat_reference_latency_proxy_ms,
        **execution,
    }


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    communication_records: list[dict[str, object]] = []

    for load_index, load_ratio in enumerate(TASK_LOAD_RATIOS):
        for trial in range(TRIALS):
            seed = RANDOM_SEED + load_index * 10000 + trial
            rng = np.random.default_rng(seed)
            active = sample_active_robots(ROBOT_COUNT, rng)
            active_count = int(active.sum())
            task_count = max(1, int(round(active_count * load_ratio)))

            scenario = generate_compact_scenario(ROBOT_COUNT, task_count, rng)
            base_cost = build_compact_base_cost(scenario, active)

            policy_rng = np.random.default_rng(seed + 10_000_000)
            sender_policy = assign_sender_policies(active, "heterogeneous_strong", policy_rng)
            policies, one_hot = sender_policy_one_hot(active, sender_policy)
            probability_map = {
                policy: per_task_probability(base_cost, active, policy)
                for policy in policies
            }
            full_counts = one_hot.sum(axis=0)
            full_support = probability_support_from_counts(full_counts, policies, probability_map)

            start_time_multiplier = lognormal_multipliers(
                rng,
                TRAVEL_NOISE_SIGMA,
                (ROBOT_COUNT, task_count),
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
            success_uniform = rng.random((ROBOT_COUNT, task_count))

            payload_bytes = preference_payload_bytes(active_count, task_count)
            flat_rng = np.random.default_rng(seed + 20_000_000)
            _, flat_comm = flat_preference_exchange(
                active,
                one_hot,
                flat_rng,
                payload_bytes,
            )
            flat_latency = communication_time_proxy_ms(
                flat_comm["bytes_transmitted"],
                int(flat_comm["communication_stages"]),
            )

            hierarchy_rng = np.random.default_rng(seed + 30_000_000)
            count_rows, hierarchy_comm = hierarchical_preference_exchange(
                active,
                one_hot,
                trial=trial,
                rng=hierarchy_rng,
                payload_bytes=payload_bytes,
                backup_enabled=True,
            )

            communication_records.extend(
                [
                    {
                        "robots": ROBOT_COUNT,
                        "active_robots": active_count,
                        "task_load_percent": load_ratio * 100.0,
                        "tasks": task_count,
                        "trial": trial + 1,
                        "communication_method": "Flat P2P",
                        "message_attempts": flat_comm["message_attempts"],
                        "bytes_transmitted": flat_comm["bytes_transmitted"],
                        "latency_proxy_ms": flat_latency,
                    },
                    {
                        "robots": ROBOT_COUNT,
                        "active_robots": active_count,
                        "task_load_percent": load_ratio * 100.0,
                        "tasks": task_count,
                        "trial": trial + 1,
                        "communication_method": "5-ary + Backup",
                        "message_attempts": hierarchy_comm["message_attempts"],
                        "bytes_transmitted": hierarchy_comm["bytes_transmitted"],
                        "latency_proxy_ms": communication_time_proxy_ms(
                            hierarchy_comm["bytes_transmitted"],
                            int(hierarchy_comm["communication_stages"]),
                        ),
                    },
                ]
            )

            started = perf_counter()
            centralized_plan = build_route_plan(
                scenario,
                base_cost,
                active,
                None,
                order_mode="hardness",
                insertion_mode="append",
                guardrail_tolerance=0.0,
            )
            centralized_runtime = perf_counter() - started

            started = perf_counter()
            current_plan = build_route_plan(
                scenario,
                base_cost,
                active,
                full_support,
                order_mode="hardness",
                insertion_mode="append",
                guardrail_tolerance=ROUTE_GUARDRAIL_TOLERANCE,
            )
            current_runtime = perf_counter() - started

            started = perf_counter()
            optimized_reference = build_route_plan(
                scenario,
                base_cost,
                active,
                full_support,
                order_mode="regret",
                insertion_mode="best_insertion",
                guardrail_tolerance=ROUTE_GUARDRAIL_TOLERANCE,
            )
            optimized_runtime = perf_counter() - started

            top_plans: dict[int, tuple[tuple[int, ...], ...]] = {}
            plan_cache: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}
            started = perf_counter()
            for receiver, counts in count_rows.items():
                key = tuple(int(value) for value in counts)
                if key not in plan_cache:
                    local_support = probability_support_from_counts(
                        counts,
                        policies,
                        probability_map,
                    )
                    plan_cache[key] = build_route_plan(
                        scenario,
                        base_cost,
                        active,
                        local_support,
                        order_mode="regret",
                        insertion_mode="best_insertion",
                        guardrail_tolerance=ROUTE_GUARDRAIL_TOLERANCE,
                    )
                top_plans[int(receiver)] = plan_cache[key]
            hierarchy_planning_runtime = perf_counter() - started

            quorum_rng = np.random.default_rng(seed + 40_000_000)
            qc_plan, witnesses, proposal_comm = proposal_quorum_exchange(
                top_plans,
                quorum_rng,
            )
            qc_rng = np.random.default_rng(seed + 50_000_000)
            authorized_routes, qc_comm = disseminate_qc(
                qc_plan,
                witnesses,
                active,
                task_count,
                qc_rng,
            )

            hierarchy_attempts = (
                hierarchy_comm["message_attempts"]
                + proposal_comm["proposal_attempts"]
                + qc_comm["qc_attempts"]
            )
            hierarchy_bytes = (
                hierarchy_comm["bytes_transmitted"]
                + proposal_comm["proposal_bytes"]
                + qc_comm["qc_bytes"]
            )
            hierarchy_stages = int(hierarchy_comm["communication_stages"]) + 2
            hierarchy_latency = communication_time_proxy_ms(
                hierarchy_bytes,
                hierarchy_stages,
            )
            safe_commit = float(qc_plan is not None and len(witnesses) > 0)
            plan_match = float(qc_plan == optimized_reference if qc_plan is not None else False)

            base_kwargs = dict(
                load_ratio=load_ratio,
                trial=trial + 1,
                active_count=active_count,
                task_count=task_count,
                flat_reference_attempts=flat_comm["message_attempts"],
                flat_reference_bytes=flat_comm["bytes_transmitted"],
                flat_reference_latency_proxy_ms=flat_latency,
            )

            for method, plan, runtime in (
                ("centralized_route_greedy", centralized_plan, centralized_runtime),
                ("full_info_current", current_plan, current_runtime),
                ("full_info_optimized", optimized_reference, optimized_runtime),
            ):
                execution = execution_record(
                    plan,
                    active,
                    scenario,
                    start_time_multiplier,
                    transition_time_multiplier,
                    service_time_multiplier,
                    success_uniform,
                )
                records.append(
                    make_record(
                        **base_kwargs,
                        method=method,
                        planning_runtime_sec=runtime,
                        execution=execution,
                    )
                )

            final_execution = execution_record(
                authorized_routes,
                active,
                scenario,
                start_time_multiplier,
                transition_time_multiplier,
                service_time_multiplier,
                success_uniform,
            )
            records.append(
                make_record(
                    **base_kwargs,
                    method="hierarchy_backup_optimized",
                    planning_runtime_sec=hierarchy_planning_runtime,
                    execution=final_execution,
                    safe_commit_rate=safe_commit,
                    plan_match_reference_rate=plan_match,
                    qc_delivery_rate=qc_comm["qc_delivery_rate"],
                    message_attempts=hierarchy_attempts,
                    bytes_transmitted=hierarchy_bytes,
                    communication_stages=hierarchy_stages,
                    latency_proxy_ms=hierarchy_latency,
                )
            )

            print(
                f"load={load_ratio * 100:4.0f}% trial={trial + 1:02d}/{TRIALS} "
                f"active={active_count} tasks={task_count} "
                f"opt_plan={optimized_runtime:.2f}s hierarchy_plans={hierarchy_planning_runtime:.2f}s"
            )

    raw = pd.DataFrame.from_records(records)
    communication_raw = pd.DataFrame.from_records(communication_records)

    numeric_metrics = [
        "task_completion_rate",
        "mission_success_rate",
        "unexecuted_task_rate",
        "duplicate_task_execution_rate",
        "actual_makespan_sec",
        "makespan_per_task_sec",
        "energy_per_completed_task",
        "travel_distance_per_completed_task_m",
        "route_task_count_cv",
        "route_finish_time_cv",
        "planning_runtime_sec",
        "safe_commit_rate",
        "plan_match_reference_rate",
        "qc_delivery_rate",
        "message_attempts",
        "bytes_transmitted",
        "shared_channel_latency_proxy_ms",
    ]
    summary = (
        raw.groupby(["task_load_percent", "method", "method_label"], as_index=False)[numeric_metrics]
        .mean()
        .sort_values(["task_load_percent", "method"])
    )

    communication_summary = (
        communication_raw.groupby(
            ["task_load_percent", "communication_method"],
            as_index=False,
        )[["message_attempts", "bytes_transmitted", "latency_proxy_ms"]]
        .mean()
        .sort_values(["task_load_percent", "communication_method"])
    )
    return raw, summary, communication_summary


def save_plot(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    filename: str,
    *,
    percent: bool = False,
    methods: tuple[str, ...] = METHODS,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for method in methods:
        part = summary[summary["method"] == method].sort_values("task_load_percent")
        ax.plot(
            part["task_load_percent"],
            part[metric],
            marker="o",
            linewidth=2,
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Task load (% of active robot count)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close(fig)


def save_communication_plot(communication_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for method in ("Flat P2P", "5-ary + Backup"):
        part = communication_summary[
            communication_summary["communication_method"] == method
        ].sort_values("task_load_percent")
        ax.plot(
            part["task_load_percent"],
            part["bytes_transmitted"] / 1e9,
            marker="o",
            linewidth=2,
            label=method,
        )
    ax.set_xlabel("Task load (% of active robot count)")
    ax.set_ylabel("Preference communication (GB)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "high_load_communication_gb.png", dpi=180)
    plt.close(fig)

    pivot = communication_summary.pivot(
        index="task_load_percent",
        columns="communication_method",
        values="bytes_transmitted",
    )
    reduction = 1.0 - pivot["5-ary + Backup"] / pivot["Flat P2P"]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(reduction.index, reduction.values, marker="o", linewidth=2)
    ax.set_xlabel("Task load (% of active robot count)")
    ax.set_ylabel("Communication reduction vs Flat P2P")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "high_load_communication_reduction.png", dpi=180)
    plt.close(fig)


def save_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    communication_summary: pd.DataFrame,
) -> None:
    ensure_output_dirs()
    clear_old_outputs()
    raw.to_csv(DATA_DIR / "high_load_100robots_raw.csv", index=False)
    summary.to_csv(DATA_DIR / "high_load_100robots_summary.csv", index=False)
    communication_summary.to_csv(
        DATA_DIR / "high_load_100robots_communication.csv",
        index=False,
    )

    save_plot(
        summary,
        "task_completion_rate",
        "Task completion rate",
        "high_load_task_completion.png",
        percent=True,
    )
    save_plot(
        summary,
        "actual_makespan_sec",
        "Actual makespan (s)",
        "high_load_makespan.png",
    )
    save_plot(
        summary,
        "energy_per_completed_task",
        "Energy per completed task",
        "high_load_energy_per_task.png",
    )
    save_plot(
        summary,
        "travel_distance_per_completed_task_m",
        "Travel distance per completed task (m)",
        "high_load_distance_per_task.png",
    )
    save_plot(
        summary,
        "route_finish_time_cv",
        "Route finish-time CV",
        "high_load_finish_time_cv.png",
    )
    save_plot(
        summary,
        "planning_runtime_sec",
        "Planning runtime (s)",
        "high_load_planning_runtime.png",
    )
    save_plot(
        summary,
        "plan_match_reference_rate",
        "Plan match to full-info optimized",
        "high_load_plan_match.png",
        percent=True,
        methods=("hierarchy_backup_optimized",),
    )
    save_plot(
        summary,
        "safe_commit_rate",
        "Safe commit rate",
        "high_load_safe_commit.png",
        percent=True,
        methods=("hierarchy_backup_optimized",),
    )
    save_communication_plot(communication_summary)


if __name__ == "__main__":
    raw_results, summary_results, communication_results = run_experiment()
    save_outputs(raw_results, summary_results, communication_results)

    display_columns = [
        "task_load_percent",
        "method_label",
        "task_completion_rate",
        "actual_makespan_sec",
        "energy_per_completed_task",
        "travel_distance_per_completed_task_m",
        "route_finish_time_cv",
        "planning_runtime_sec",
        "safe_commit_rate",
        "plan_match_reference_rate",
        "qc_delivery_rate",
    ]
    print("\n100-robot high-load summary")
    print(summary_results[display_columns].to_string(index=False))
    print("\nCommunication scaling")
    print(communication_results.to_string(index=False))
