from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FullRound:
    costs: np.ndarray
    probabilities: np.ndarray
    ballots: np.ndarray
    full_counts: np.ndarray
    full_winner: int
    full_tie: bool
    tie_priority: np.ndarray
    delivery_random: np.ndarray


@dataclass(frozen=True)
class LossResult:
    delivered: np.ndarray
    received_counts: np.ndarray
    winner: int | None
    tie: bool


def generate_costs(
    n: int,
    cost_start: float = 10.0,
    cost_step: float = 5.0,
) -> np.ndarray:
    """Return fixed deterministic costs for n robots."""
    if n <= 0:
        raise ValueError("n must be positive")
    if cost_start <= 0:
        raise ValueError("cost_start must be positive")
    if cost_step < 0:
        raise ValueError("cost_step must be non-negative")

    return cost_start + cost_step * np.arange(n, dtype=float)


def cost_to_probability(costs: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Convert lower costs into larger normalized voting probabilities."""
    costs = np.asarray(costs, dtype=float)

    if costs.ndim != 1 or len(costs) == 0:
        raise ValueError("costs must be a non-empty 1-D array")
    if np.any(costs <= 0):
        raise ValueError("all costs must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    weights = np.power(1.0 / costs, alpha)
    return weights / weights.sum()


def get_winner(
    vote_counts: np.ndarray,
    tie_priority: np.ndarray,
) -> tuple[int | None, bool]:
    """Select a winner; resolve ties using a pre-generated random priority."""
    vote_counts = np.asarray(vote_counts)
    tie_priority = np.asarray(tie_priority)

    if vote_counts.shape != tie_priority.shape:
        raise ValueError("vote_counts and tie_priority must have the same shape")

    max_vote = int(vote_counts.max())
    if max_vote == 0:
        return None, False

    candidates = np.flatnonzero(vote_counts == max_vote)
    is_tie = len(candidates) > 1
    winner = int(candidates[np.argmax(tie_priority[candidates])])
    return winner, is_tie


def generate_full_round(
    n: int,
    rng: np.random.Generator,
    *,
    cost_start: float = 10.0,
    cost_step: float = 5.0,
    alpha: float = 1.0,
) -> FullRound:
    """Generate one complete voting round before communication loss is applied."""
    costs = generate_costs(n, cost_start=cost_start, cost_step=cost_step)
    probabilities = cost_to_probability(costs, alpha=alpha)

    ballots = rng.choice(n, size=n, p=probabilities)
    full_counts = np.bincount(ballots, minlength=n)

    # A fixed random priority avoids systematic Robot-ID bias when counts tie.
    tie_priority = rng.random(n)
    full_winner, full_tie = get_winner(full_counts, tie_priority)
    if full_winner is None:
        raise RuntimeError("a complete round with n > 0 must have a winner")

    # Reusing these random numbers across loss rates makes each higher loss rate
    # a degraded version of the same underlying communication realization.
    delivery_random = rng.random(n)

    return FullRound(
        costs=costs,
        probabilities=probabilities,
        ballots=ballots,
        full_counts=full_counts,
        full_winner=full_winner,
        full_tie=full_tie,
        tie_priority=tie_priority,
        delivery_random=delivery_random,
    )


def apply_vote_loss(round_data: FullRound, loss_rate: float) -> LossResult:
    """Drop vote messages independently using a Bernoulli packet-loss model."""
    if not 0.0 <= loss_rate <= 1.0:
        raise ValueError("loss_rate must be between 0 and 1")

    delivered = round_data.delivery_random >= loss_rate
    received_ballots = round_data.ballots[delivered]

    received_counts = np.bincount(
        received_ballots,
        minlength=len(round_data.costs),
    )

    winner, tie = get_winner(received_counts, round_data.tie_priority)

    return LossResult(
        delivered=delivered,
        received_counts=received_counts,
        winner=winner,
        tie=tie,
    )
