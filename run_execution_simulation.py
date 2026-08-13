from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from run_algorithm_experiment import sample_active_robots
from run_decentralized_experiment import (
    assign_sender_policies,
    peer_delivery_matrix,
    policy_counts_for_receivers,
    required_quorum,
)
from run_multitask_experiment import per_task_probability


RANDOM_SEED = 20260813
ROBOT_COUNTS = (10, 20, 40, 60)
TASK_LOAD_RATIOS = (0.50, 1.00, 1.50)
TRIALS = 40
PACKET_LOSS_RATES = (0.30, 0.50, 0.70)
MAX_ATTEMPTS = 3
QUORUM_THRESHOLD = 2.0 / 3.0
WORKSPACE_SIZE_M = 100.0

BASE_COST_FLOOR = 0.05
BASE_TIME_WEIGHT = 0.70
BASE_ENERGY_WEIGHT = 0.20
BASE_RISK_WEIGHT = 0.10
PREFERENCE_WEIGHT = 0.60
ROUTE_WEIGHT = 0.40
SERVICE_ENERGY_PER_SECOND = 0.03
TRAVEL_NOISE_SIGMA = 0.10
SERVICE_NOISE_SIGMA = 0.15

PREFERENCE_POLICIES = ("inverse_a2", "inverse_a3", "softmax_b2")
METHODS = (
    "centralized_route_greedy",
    "full_info_preference_route",
    "decentralized_preference_route",
)
METHOD_LABELS = {
    "centralized_route_greedy": "Centralized Route Greedy",
    "full_info_preference_route": "Full-Info Preference + Route",
    "decentralized_preference_route": "Decentralized Preference + Quorum",
}

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "execution"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_outputs() -> None:
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def minmax_scale_by_task(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    scaled = np.zeros_like(values, dtype=float)
    active_values = values[active, :]
    low = active_values.min(axis=0)
    high = active_values.max(axis=0)
    span = high - low
    safe = np.where(span > 1e-12, span, 1.0)
    scaled[active, :] = (active_values - low) / safe
    return scaled


def generate_scenario(
    n: int,
    task_count: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    robot_positions = rng.uniform(0.0, WORKSPACE_SIZE_M, size=(n, 2))
    task_positions = rng.uniform(0.0, WORKSPACE_SIZE_M, size=(task_count, 2))
    robot_speeds = rng.uniform(0.9, 1.3, size=n)
    energy_per_meter = rng.uniform(0.8, 1.2, size=n)
    robot_reliability = rng.uniform(0.992, 0.999, size=n)
    service_times = rng.uniform(20.0, 60.0, size=task_count)
    task_success_factor = rng.uniform(0.995, 1.0, size=task_count)

    start_delta = robot_positions[:, None, :] - task_positions[None, :, :]
    start_distance = np.sqrt(np.sum(start_delta * start_delta, axis=2))

    task_delta = task_positions[:, None, :] - task_positions[None, :, :]
    task_distance = np.sqrt(np.sum(task_delta * task_delta, axis=2))

    start_travel_time = start_distance / robot_speeds[:, None]
    transition_travel_time = task_distance[None, :, :] / robot_speeds[:, None, None]
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
        "transition_travel_time": transition_travel_time,
        "start_energy": start_energy,
    }


def build_base_cost(
    scenario: dict[str, np.ndarray],
    active: np.ndarray,
) -> np.ndarray:
    """Pre-allocation cost that never depends on future task count or route order."""
    time_scaled = minmax_scale_by_task(scenario["start_travel_time"], active)
    energy_scaled = minmax_scale_by_task(scenario["start_energy"], active)

    reliability = scenario["robot_reliability"]
    risk = 1.0 - reliability
    active_risk = risk[active]
    risk_low = active_risk.min()
    risk_high = active_risk.max()
    risk_span = risk_high - risk_low
    if risk_span <= 1e-12:
        risk_scaled = np.zeros_like(risk)
    else:
        risk_scaled = (risk - risk_low) / risk_span

    base_cost = (
        BASE_COST_FLOOR
        + BASE_TIME_WEIGHT * time_scaled
        + BASE_ENERGY_WEIGHT * energy_scaled
        + BASE_RISK_WEIGHT * risk_scaled[:, None]
    )
    return base_cost


def aggregate_preference_support(
    counts: np.ndarray,
    policies: list[str],
    probability_map: dict[str, np.ndarray],
) -> np.ndarray:
    total = float(counts.sum())
    if total <= 0:
        raise RuntimeError("Every active receiver must retain at least its own preference")
    support = np.zeros_like(next(iter(probability_map.values())), dtype=float)
    for policy, count in zip(policies, counts, strict=True):
        if count > 0:
            support += (count / total) * probability_map[policy]
    return support


def task_order_from_base_cost(base_cost: np.ndarray, active: np.ndarray) -> np.ndarray:
    active_cost = base_cost[active, :]
    hardness = active_cost.min(axis=0)
    task_ids = np.arange(base_cost.shape[1])
    return np.lexsort((task_ids, -hardness))


def normalized_preference_for_task(
    support: np.ndarray,
    active_indices: np.ndarray,
    task: int,
) -> np.ndarray:
    values = support[active_indices, task]
    low = float(values.min())
    high = float(values.max())
    if high - low <= 1e-12:
        return np.ones_like(values)
    return (values - low) / (high - low)


def route_plan(
    scenario: dict[str, np.ndarray],
    base_cost: np.ndarray,
    active: np.ndarray,
    support: np.ndarray | None,
) -> tuple[tuple[int, ...], ...]:
    """Assign all tasks once and determine route order with a route-aware heuristic.

    Base cost is only used to rank which tasks are difficult. Future workload is
    not hidden inside C_ij. During planning, the current end point of each route
    is used to estimate the next travel segment and projected route finish time.
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

        for index, robot in enumerate(active_indices):
            if routes[robot]:
                previous = routes[robot][-1]
                edge_time = transition_time[robot, previous, task]
            else:
                edge_time = start_time[robot, task]
            projected[index] = projected_finish[robot] + edge_time + service_times[task]

        proj_low = float(projected.min())
        proj_high = float(projected.max())
        if proj_high - proj_low <= 1e-12:
            route_score = np.ones_like(projected)
        else:
            route_score = 1.0 - (projected - proj_low) / (proj_high - proj_low)

        if support is None:
            combined = route_score
        else:
            preference_score = normalized_preference_for_task(
                support,
                active_indices,
                int(task),
            )
            combined = PREFERENCE_WEIGHT * preference_score + ROUTE_WEIGHT * route_score

        best_value = float(combined.max())
        best_local = np.flatnonzero(np.isclose(combined, best_value, atol=1e-12))[0]
        chosen = int(active_indices[best_local])
        routes[chosen].append(int(task))
        projected_finish[chosen] = projected[best_local]

    return tuple(tuple(route) for route in routes)


def full_information_support(
    sender_policy: dict[int, str],
    probability_map: dict[str, np.ndarray],
) -> np.ndarray:
    counts = Counter(sender_policy.values())
    policies = sorted(counts)
    values = np.asarray([counts[policy] for policy in policies], dtype=float)
    return aggregate_preference_support(values, policies, probability_map)


def local_route_plans(
    delivered: np.ndarray,
    active: np.ndarray,
    sender_policy: dict[int, str],
    probability_map: dict[str, np.ndarray],
    scenario: dict[str, np.ndarray],
    base_cost: np.ndarray,
) -> dict[int, tuple[tuple[int, ...], ...]]:
    policies, receiver_counts = policy_counts_for_receivers(delivered, active, sender_policy)
    plans: dict[int, tuple[tuple[int, ...], ...]] = {}
    cache: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}

    for receiver in np.flatnonzero(active):
        key = tuple(int(value) for value in receiver_counts[receiver])
        if key not in cache:
            support = aggregate_preference_support(
                receiver_counts[receiver],
                policies,
                probability_map,
            )
            cache[key] = route_plan(scenario, base_cost, active, support)
        plans[int(receiver)] = cache[key]
    return plans


def proposal_labels(
    plans: dict[int, tuple[tuple[int, ...], ...]],
) -> tuple[dict[int, int], dict[int, tuple[tuple[int, ...], ...]], Counter[int]]:
    signatures = sorted(set(plans.values()))
    signature_to_label = {signature: index for index, signature in enumerate(signatures)}
    sender_labels = {robot: signature_to_label[plan] for robot, plan in plans.items()}
    label_to_plan = {label: signature for signature, label in signature_to_label.items()}
    return sender_labels, label_to_plan, Counter(sender_labels.values())


def local_quorum_commits(
    plans: dict[int, tuple[tuple[int, ...], ...]],
    proposal_delivered: np.ndarray,
    active: np.ndarray,
) -> tuple[dict[int, int], dict[int, tuple[tuple[int, ...], ...]], Counter[int]]:
    sender_labels, label_to_plan, global_counts = proposal_labels(plans)
    active_indices = np.flatnonzero(active)
    needed = required_quorum(len(active_indices), QUORUM_THRESHOLD)
    commits: dict[int, int] = {}

    for receiver in active_indices:
        visible = [
            sender_labels[int(sender)]
            for sender in active_indices
            if proposal_delivered[receiver, sender]
        ]
        if not visible:
            continue
        counts = Counter(visible)
        best_count = max(counts.values())
        if best_count < needed:
            continue
        best_labels = [label for label, count in counts.items() if count == best_count]
        commits[int(receiver)] = min(best_labels)

    return commits, label_to_plan, global_counts


def baseline_robot_routes(
    plan: tuple[tuple[int, ...], ...],
    active: np.ndarray,
) -> dict[int, tuple[int, ...]]:
    return {int(robot): plan[int(robot)] for robot in np.flatnonzero(active)}


def decentralized_robot_routes(
    commits: dict[int, int],
    label_to_plan: dict[int, tuple[tuple[int, ...], ...]],
) -> dict[int, tuple[int, ...]]:
    routes: dict[int, tuple[int, ...]] = {}
    for robot, label in commits.items():
        routes[int(robot)] = label_to_plan[label][int(robot)]
    return routes


def lognormal_multipliers(
    rng: np.random.Generator,
    sigma: float,
    shape: tuple[int, ...],
) -> np.ndarray:
    mean = -0.5 * sigma * sigma
    return rng.lognormal(mean=mean, sigma=sigma, size=shape)


def simulate_execution(
    robot_routes: dict[int, tuple[int, ...]],
    active: np.ndarray,
    scenario: dict[str, np.ndarray],
    start_time_multiplier: np.ndarray,
    transition_time_multiplier: np.ndarray,
    service_time_multiplier: np.ndarray,
    success_uniform: np.ndarray,
) -> dict[str, float | bool]:
    task_count = len(scenario["service_times"])
    start_time = scenario["start_travel_time"]
    transition_time = scenario["transition_travel_time"]
    start_distance = scenario["start_distance"]
    task_distance = scenario["task_distance"]
    service_times = scenario["service_times"]
    energy_per_meter = scenario["energy_per_meter"]
    reliability = scenario["robot_reliability"]
    task_success_factor = scenario["task_success_factor"]

    task_attempt_count = np.zeros(task_count, dtype=int)
    task_completed = np.zeros(task_count, dtype=bool)
    estimated_finishes: list[float] = []
    actual_finishes: list[float] = []
    eta_absolute_errors: list[float] = []
    eta_percentage_errors: list[float] = []
    total_distance = 0.0
    total_energy = 0.0
    successful_attempts = 0
    total_attempts = 0
    per_robot_task_counts = []

    for robot in np.flatnonzero(active):
        route = robot_routes.get(int(robot), ())
        per_robot_task_counts.append(len(route))
        estimated_clock = 0.0
        actual_clock = 0.0
        previous: int | None = None

        for task in route:
            task_attempt_count[task] += 1
            total_attempts += 1

            if previous is None:
                est_travel = start_time[robot, task]
                actual_travel = est_travel * start_time_multiplier[robot, task]
                distance = start_distance[robot, task]
            else:
                est_travel = transition_time[robot, previous, task]
                actual_travel = est_travel * transition_time_multiplier[previous, task]
                distance = task_distance[previous, task]

            est_service = service_times[task]
            actual_service = est_service * service_time_multiplier[task]
            estimated_clock += est_travel + est_service
            actual_clock += actual_travel + actual_service

            success_probability = reliability[robot] * task_success_factor[task]
            success = bool(success_uniform[robot, task] < success_probability)
            if success:
                task_completed[task] = True
                successful_attempts += 1

            estimated_finishes.append(estimated_clock)
            actual_finishes.append(actual_clock)
            abs_error = abs(actual_clock - estimated_clock)
            eta_absolute_errors.append(abs_error)
            eta_percentage_errors.append(abs_error / max(actual_clock, 1e-9))

            total_distance += distance
            total_energy += distance * energy_per_meter[robot] + actual_service * SERVICE_ENERGY_PER_SECOND
            previous = int(task)

    attempted_tasks = int(np.count_nonzero(task_attempt_count > 0))
    duplicate_tasks = int(np.count_nonzero(task_attempt_count > 1))
    completed_tasks = int(np.count_nonzero(task_completed))
    coverage = attempted_tasks / task_count if task_count else 1.0
    duplicate_rate = duplicate_tasks / task_count if task_count else 0.0
    completion_rate = completed_tasks / task_count if task_count else 1.0
    unexecuted_rate = 1.0 - coverage
    mission_success = bool(completed_tasks == task_count and duplicate_tasks == 0)

    counts = np.asarray(per_robot_task_counts, dtype=float)
    mean_count = float(counts.mean()) if len(counts) else 0.0
    load_cv = float(counts.std() / mean_count) if mean_count > 1e-12 else 0.0

    estimated_makespan = max(estimated_finishes) if estimated_finishes else np.nan
    actual_makespan = max(actual_finishes) if actual_finishes else np.nan
    if estimated_finishes and actual_finishes:
        makespan_error = abs(actual_makespan - estimated_makespan) / max(actual_makespan, 1e-9)
    else:
        makespan_error = np.nan

    return {
        "task_execution_coverage": coverage,
        "task_completion_rate": completion_rate,
        "mission_success": mission_success,
        "duplicate_task_execution_rate": duplicate_rate,
        "unexecuted_task_rate": unexecuted_rate,
        "attempt_success_rate": successful_attempts / total_attempts if total_attempts else 0.0,
        "estimated_makespan_sec": estimated_makespan,
        "actual_makespan_sec": actual_makespan,
        "makespan_absolute_percentage_error": makespan_error,
        "mean_eta_absolute_error_sec": float(np.mean(eta_absolute_errors)) if eta_absolute_errors else np.nan,
        "mean_eta_absolute_percentage_error": float(np.mean(eta_percentage_errors)) if eta_percentage_errors else np.nan,
        "total_travel_distance_m": total_distance,
        "total_energy_units": total_energy,
        "route_task_count_cv": load_cv,
    }


def make_method_record(
    *,
    n: int,
    task_count: int,
    load_ratio: float,
    trial: int,
    loss_rate: float,
    method: str,
    execution: dict[str, float | bool],
    active_count: int,
    preference_receive_rate: float = 1.0,
    proposal_receive_rate: float = 1.0,
    modal_plan_share: float = 1.0,
    strict_plan_agreement: bool = True,
    any_commit: bool = True,
    safe_commit: bool = True,
    split_brain: bool = False,
    committed_robot_fraction: float = 1.0,
) -> dict[str, object]:
    return {
        "robots": n,
        "active_robots": active_count,
        "tasks": task_count,
        "task_load_ratio": load_ratio,
        "trial": trial,
        "packet_loss_rate": loss_rate,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "preference_receive_rate": preference_receive_rate,
        "proposal_receive_rate": proposal_receive_rate,
        "modal_plan_share": modal_plan_share,
        "strict_plan_agreement": strict_plan_agreement,
        "any_commit": any_commit,
        "safe_commit": safe_commit,
        "split_brain": split_brain,
        "committed_robot_fraction": committed_robot_fraction,
        **execution,
    }


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []

    for n in ROBOT_COUNTS:
        for load_index, load_ratio in enumerate(TASK_LOAD_RATIOS):
            for trial in range(TRIALS):
                rng = np.random.default_rng(RANDOM_SEED + n * 100000 + load_index * 10000 + trial)
                active = sample_active_robots(n, rng)
                active_count = int(active.sum())
                task_count = max(1, int(round(active_count * load_ratio)))

                scenario = generate_scenario(n, task_count, rng)
                base_cost = build_base_cost(scenario, active)

                sender_policy = assign_sender_policies(active, "heterogeneous_strong", rng)
                used_policies = sorted(set(sender_policy.values()))
                probability_map = {
                    policy: per_task_probability(base_cost, active, policy)
                    for policy in used_policies
                }
                full_support = full_information_support(sender_policy, probability_map)

                centralized_plan = route_plan(scenario, base_cost, active, support=None)
                full_info_plan = route_plan(scenario, base_cost, active, support=full_support)

                start_time_multiplier = lognormal_multipliers(rng, TRAVEL_NOISE_SIGMA, (n, task_count))
                transition_time_multiplier = lognormal_multipliers(
                    rng,
                    TRAVEL_NOISE_SIGMA,
                    (task_count, task_count),
                )
                np.fill_diagonal(transition_time_multiplier, 1.0)
                service_time_multiplier = lognormal_multipliers(rng, SERVICE_NOISE_SIGMA, (task_count,))
                success_uniform = rng.random((n, task_count))

                preference_attempt_random = rng.random((MAX_ATTEMPTS, n, n))
                proposal_attempt_random = rng.random((MAX_ATTEMPTS, n, n))

                centralized_execution = simulate_execution(
                    baseline_robot_routes(centralized_plan, active),
                    active,
                    scenario,
                    start_time_multiplier,
                    transition_time_multiplier,
                    service_time_multiplier,
                    success_uniform,
                )
                full_info_execution = simulate_execution(
                    baseline_robot_routes(full_info_plan, active),
                    active,
                    scenario,
                    start_time_multiplier,
                    transition_time_multiplier,
                    service_time_multiplier,
                    success_uniform,
                )

                for loss_rate in PACKET_LOSS_RATES:
                    records.append(
                        make_method_record(
                            n=n,
                            task_count=task_count,
                            load_ratio=load_ratio,
                            trial=trial + 1,
                            loss_rate=loss_rate,
                            method="centralized_route_greedy",
                            execution=centralized_execution,
                            active_count=active_count,
                        )
                    )
                    records.append(
                        make_method_record(
                            n=n,
                            task_count=task_count,
                            load_ratio=load_ratio,
                            trial=trial + 1,
                            loss_rate=loss_rate,
                            method="full_info_preference_route",
                            execution=full_info_execution,
                            active_count=active_count,
                        )
                    )

                    preference_delivered, preference_receive_rate, _ = peer_delivery_matrix(
                        active,
                        preference_attempt_random,
                        loss_rate,
                    )
                    plans = local_route_plans(
                        preference_delivered,
                        active,
                        sender_policy,
                        probability_map,
                        scenario,
                        base_cost,
                    )
                    _, _, plan_counts = proposal_labels(plans)
                    modal_plan_share = max(plan_counts.values()) / active_count
                    strict_plan_agreement = len(plan_counts) == 1

                    proposal_delivered, proposal_receive_rate, _ = peer_delivery_matrix(
                        active,
                        proposal_attempt_random,
                        loss_rate,
                    )
                    commits, label_to_plan, _ = local_quorum_commits(plans, proposal_delivered, active)
                    committed_labels = set(commits.values())
                    any_commit = bool(commits)
                    split_brain = len(committed_labels) > 1
                    safe_commit = any_commit and not split_brain
                    committed_robot_fraction = len(commits) / active_count

                    decentralized_execution = simulate_execution(
                        decentralized_robot_routes(commits, label_to_plan),
                        active,
                        scenario,
                        start_time_multiplier,
                        transition_time_multiplier,
                        service_time_multiplier,
                        success_uniform,
                    )
                    records.append(
                        make_method_record(
                            n=n,
                            task_count=task_count,
                            load_ratio=load_ratio,
                            trial=trial + 1,
                            loss_rate=loss_rate,
                            method="decentralized_preference_route",
                            execution=decentralized_execution,
                            active_count=active_count,
                            preference_receive_rate=preference_receive_rate,
                            proposal_receive_rate=proposal_receive_rate,
                            modal_plan_share=modal_plan_share,
                            strict_plan_agreement=strict_plan_agreement,
                            any_commit=any_commit,
                            safe_commit=safe_commit,
                            split_brain=split_brain,
                            committed_robot_fraction=committed_robot_fraction,
                        )
                    )

    raw = pd.DataFrame.from_records(records)
    summary = (
        raw.groupby(
            ["robots", "task_load_ratio", "packet_loss_rate", "method", "method_label"],
            as_index=False,
        )
        .agg(
            average_tasks=("tasks", "mean"),
            average_active_robots=("active_robots", "mean"),
            task_execution_coverage=("task_execution_coverage", "mean"),
            task_completion_rate=("task_completion_rate", "mean"),
            mission_success_rate=("mission_success", "mean"),
            duplicate_task_execution_rate=("duplicate_task_execution_rate", "mean"),
            unexecuted_task_rate=("unexecuted_task_rate", "mean"),
            attempt_success_rate=("attempt_success_rate", "mean"),
            estimated_makespan_sec=("estimated_makespan_sec", "mean"),
            actual_makespan_sec=("actual_makespan_sec", "mean"),
            makespan_absolute_percentage_error=("makespan_absolute_percentage_error", "mean"),
            mean_eta_absolute_error_sec=("mean_eta_absolute_error_sec", "mean"),
            mean_eta_absolute_percentage_error=("mean_eta_absolute_percentage_error", "mean"),
            total_travel_distance_m=("total_travel_distance_m", "mean"),
            total_energy_units=("total_energy_units", "mean"),
            route_task_count_cv=("route_task_count_cv", "mean"),
            preference_receive_rate=("preference_receive_rate", "mean"),
            proposal_receive_rate=("proposal_receive_rate", "mean"),
            modal_plan_share=("modal_plan_share", "mean"),
            strict_plan_agreement_rate=("strict_plan_agreement", "mean"),
            any_commit_rate=("any_commit", "mean"),
            safe_commit_rate=("safe_commit", "mean"),
            split_brain_rate=("split_brain", "mean"),
            committed_robot_fraction=("committed_robot_fraction", "mean"),
        )
        .reset_index(drop=True)
    )

    by_method = (
        summary.groupby(["method", "method_label"], as_index=False)
        .agg(
            task_completion_rate=("task_completion_rate", "mean"),
            mission_success_rate=("mission_success_rate", "mean"),
            duplicate_task_execution_rate=("duplicate_task_execution_rate", "mean"),
            unexecuted_task_rate=("unexecuted_task_rate", "mean"),
            mean_eta_absolute_percentage_error=("mean_eta_absolute_percentage_error", "mean"),
            actual_makespan_sec=("actual_makespan_sec", "mean"),
            total_energy_units=("total_energy_units", "mean"),
            route_task_count_cv=("route_task_count_cv", "mean"),
            safe_commit_rate=("safe_commit_rate", "mean"),
        )
        .reset_index(drop=True)
    )
    return raw, summary, by_method


def subset_reference(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[np.isclose(summary["packet_loss_rate"], 0.50)].copy()


def plot_completion_rate(summary: pd.DataFrame) -> None:
    data = (
        subset_reference(summary)
        .groupby(["task_load_ratio", "method", "method_label"], as_index=False)
        .agg(rate=("task_completion_rate", "mean"))
    )
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for method in METHODS:
        subset = data[data["method"] == method]
        ax.plot(subset["task_load_ratio"] * 100.0, subset["rate"], marker="o", linewidth=2.0, label=METHOD_LABELS[method])
    ax.set_xlabel("Tasks as % of Active Robot Count")
    ax.set_ylabel("Task Completion Rate")
    ax.set_title("Execution-Level Task Completion after One-Shot Allocation")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "execution_task_completion_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mission_success(summary: pd.DataFrame) -> None:
    data = (
        subset_reference(summary)
        .groupby(["task_load_ratio", "method", "method_label"], as_index=False)
        .agg(rate=("mission_success_rate", "mean"))
    )
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for method in METHODS:
        subset = data[data["method"] == method]
        ax.plot(subset["task_load_ratio"] * 100.0, subset["rate"], marker="o", linewidth=2.0, label=METHOD_LABELS[method])
    ax.set_xlabel("Tasks as % of Active Robot Count")
    ax.set_ylabel("All-Tasks Mission Success Rate")
    ax.set_title("Probability that Every Task Finishes without Duplicate Execution")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "execution_mission_success_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_eta_error(summary: pd.DataFrame) -> None:
    data = summary[(np.isclose(summary["task_load_ratio"], 1.0)) & (np.isclose(summary["packet_loss_rate"], 0.50))]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for method in METHODS:
        subset = data[data["method"] == method]
        ax.plot(subset["robots"], subset["mean_eta_absolute_percentage_error"], marker="o", linewidth=2.0, label=METHOD_LABELS[method])
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Mean ETA Absolute Percentage Error")
    ax.set_title("Estimated vs Actual Task Completion Time")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "execution_eta_error.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_duplicate_rate(summary: pd.DataFrame) -> None:
    data = summary[(summary["method"] == "decentralized_preference_route") & (np.isclose(summary["task_load_ratio"], 1.0))]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for loss_rate in PACKET_LOSS_RATES:
        subset = data[np.isclose(data["packet_loss_rate"], loss_rate)]
        ax.plot(subset["robots"], subset["duplicate_task_execution_rate"], marker="o", linewidth=2.0, label=f"{int(loss_rate * 100)}% packet loss")
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Duplicate Task Execution Rate")
    ax.set_title("Does Incomplete Information Cause Two Robots to Execute One Task?")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "execution_duplicate_task_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_unexecuted_rate(summary: pd.DataFrame) -> None:
    data = summary[(summary["method"] == "decentralized_preference_route") & (np.isclose(summary["task_load_ratio"], 1.0))]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for loss_rate in PACKET_LOSS_RATES:
        subset = data[np.isclose(data["packet_loss_rate"], loss_rate)]
        ax.plot(subset["robots"], subset["unexecuted_task_rate"], marker="o", linewidth=2.0, label=f"{int(loss_rate * 100)}% packet loss")
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Unexecuted Task Rate")
    ax.set_title("Execution Availability after Local Quorum Authorization")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "execution_unexecuted_task_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_makespan(summary: pd.DataFrame) -> None:
    data = summary[(np.isclose(summary["task_load_ratio"], 1.0)) & (np.isclose(summary["packet_loss_rate"], 0.50))]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for method in METHODS:
        subset = data[data["method"] == method]
        ax.plot(subset["robots"], subset["actual_makespan_sec"], marker="o", linewidth=2.0, label=METHOD_LABELS[method])
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Average Actual Mission Makespan (s)")
    ax.set_title("Actual Mission Time after Route Sequencing")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "execution_actual_makespan.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_energy(summary: pd.DataFrame) -> None:
    data = (
        subset_reference(summary)
        .groupby(["task_load_ratio", "method", "method_label"], as_index=False)
        .agg(energy=("total_energy_units", "mean"))
    )
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for method in METHODS:
        subset = data[data["method"] == method]
        ax.plot(subset["task_load_ratio"] * 100.0, subset["energy"], marker="o", linewidth=2.0, label=METHOD_LABELS[method])
    ax.set_xlabel("Tasks as % of Active Robot Count")
    ax.set_ylabel("Average Execution Energy (synthetic units)")
    ax.set_title("Route Energy Cost after One-Shot Multi-Task Allocation")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "execution_energy_consumption.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_safe_commit(summary: pd.DataFrame) -> None:
    data = summary[(summary["method"] == "decentralized_preference_route") & (np.isclose(summary["task_load_ratio"], 1.0))]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for loss_rate in PACKET_LOSS_RATES:
        subset = data[np.isclose(data["packet_loss_rate"], loss_rate)]
        ax.plot(subset["robots"], subset["safe_commit_rate"], marker="o", linewidth=2.0, label=f"{int(loss_rate * 100)}% packet loss")
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Safe Quorum Commit Rate")
    ax.set_title("Can the Decentralized Route Plan Be Safely Authorized?")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "execution_safe_commit_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_outputs(raw: pd.DataFrame, summary: pd.DataFrame, by_method: pd.DataFrame) -> None:
    raw.to_csv(DATA_DIR / "execution_raw_results.csv", index=False)
    summary.to_csv(DATA_DIR / "execution_summary_results.csv", index=False)
    by_method.to_csv(DATA_DIR / "execution_by_method.csv", index=False)

    plot_completion_rate(summary)
    plot_mission_success(summary)
    plot_eta_error(summary)
    plot_duplicate_rate(summary)
    plot_unexecuted_rate(summary)
    plot_makespan(summary)
    plot_energy(summary)
    plot_safe_commit(summary)


def main() -> None:
    ensure_output_dirs()
    clear_old_outputs()
    raw, summary, by_method = run_experiment()
    save_outputs(raw, summary, by_method)
    print(by_method.to_string(index=False))


if __name__ == "__main__":
    main()
