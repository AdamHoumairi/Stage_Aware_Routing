#!/usr/bin/env python3
"""Exact N=2 exploration for the corrected average-cost cloud CTMDP.

This is a clean reference implementation of the mathematical model in
``average_cost_ctmdp_formulation.tex``.  It deliberately does not import or
reuse transition/cost code from the legacy discounted-MDP experiments.

Model
-----
Each state is ``(q1, n1, q2, n2)`` where ``qi`` is the number of jobs in the
finite VM-side stage and ``ni`` is the number of busy homogeneous containers.
Every accepted arrival enters a VM-side stage; there is no bypass.  A VM-side
completion promotes one job to an idle container, and container completions
leave the system.  Routing decisions occur only on external arrivals.

Primary numerical methods
-------------------------
* constant-rate uniformization;
* span-normalized relative value iteration (RVI);
* exact stationary policy evaluation through the CTMC generator;
* Poisson-equation verification of the average cost;
* optional continuous-time simulation used only as a validation check.

The script also audits the decision-relevant oracle: exact ties, VM-swap
symmetry, bias and routing-score monotonicity candidates, composition
sensitivity, tie-aware JSQ accuracy/regret, and the exact JSQ optimality gap.

Examples
--------
Run a quick corrected exploration for K=2 and K=4, explicitly comparing the
original one-place VM-side stage with a two-place stage::

    python n2_average_cost_ctmdp.py \
        --K 2 4 --B 1 2 --lambdas 0.5 1 1.5 2 3 4 6 8 \
        --output-dir corrected_n2_results

Add a continuous-time simulation check for each scenario::

    python n2_average_cost_ctmdp.py --K 2 4 --lambdas 1 2 4 \
        --simulate --simulation-horizon 50000

The exact evaluator, not simulation, is the primary source of reported
performance for the finite N=2 model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


State = tuple[int, int, int, int]
Policy = np.ndarray


@dataclass(frozen=True)
class ModelParameters:
    """Parameters for the homogeneous two-VM, two-stage system."""

    K: int = 4
    B: int = 1
    arrival_rate: float = 1.0
    mu_vm: float = 1.0
    mu_container: float = 1.0
    holding_vm: float = 1.0
    holding_container: float = 1.0
    blocking_cost: float = 10.0
    uniformization_slack: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.K < 1 or self.B < 1:
            raise ValueError("K and B must be positive integers.")
        positive = {
            "arrival_rate": self.arrival_rate,
            "mu_vm": self.mu_vm,
            "mu_container": self.mu_container,
        }
        if any(value <= 0.0 for value in positive.values()):
            raise ValueError(f"Rates must be strictly positive: {positive}")
        nonnegative = {
            "holding_vm": self.holding_vm,
            "holding_container": self.holding_container,
            "blocking_cost": self.blocking_cost,
            "uniformization_slack": self.uniformization_slack,
        }
        if any(value < 0.0 for value in nonnegative.values()):
            raise ValueError(f"Costs and slack must be nonnegative: {nonnegative}")

    @property
    def nominal_capacity(self) -> float:
        return 2.0 * min(self.mu_vm, self.K * self.mu_container)

    @property
    def rho(self) -> float:
        return self.arrival_rate / self.nominal_capacity

    @property
    def maximum_internal_rate(self) -> float:
        per_vm = max(
            self.K * self.mu_container,
            self.mu_vm + (self.K - 1) * self.mu_container,
        )
        return 2.0 * per_vm

    @property
    def uniformization_rate(self) -> float:
        base = self.arrival_rate + self.maximum_internal_rate
        # A strict positive slack creates a dummy self-loop even in a
        # maximum-rate state.  The relative slack avoids an arbitrary unit.
        return base * (1.0 + max(self.uniformization_slack, 1.0e-12))


@dataclass
class RelativeValueResult:
    gain: float
    bias: np.ndarray
    policy: Policy
    optimal_action_sets: tuple[tuple[int, ...], ...]
    iterations: int
    span_residual: float
    bellman_residual: float


@dataclass
class PolicyEvaluation:
    policy_name: str
    gain: float
    stationary: np.ndarray
    generator_residual: float
    poisson_gain: float
    poisson_gain_error: float
    blocking_probability: float
    mean_vm_occupancy: float
    mean_container_occupancy: float
    mean_total_occupancy: float
    admitted_throughput: float
    promotion_throughput: float
    departure_throughput: float
    mean_response_time: float
    flow_residual_arrival_promotion: float
    flow_residual_promotion_departure: float
    cost_decomposition_error: float

    def summary_dict(self) -> dict[str, float | str]:
        result = asdict(self)
        result.pop("stationary")
        return result


@dataclass
class SimulationEstimate:
    policy_name: str
    observation_time: float
    events: int
    arrivals: int
    blocked: int
    blocking_probability: float
    mean_vm_occupancy: float
    mean_container_occupancy: float
    mean_total_occupancy: float
    average_cost: float

    def summary_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


class N2AverageCostCTMDP:
    """Finite aggregate CTMDP and exact solvers for N=2."""

    def __init__(self, parameters: ModelParameters):
        self.p = parameters
        self.states: tuple[State, ...] = tuple(
            (q1, n1, q2, n2)
            for q1, n1, q2, n2 in product(
                range(self.p.B + 1),
                range(self.p.K + 1),
                range(self.p.B + 1),
                range(self.p.K + 1),
            )
        )
        self.index: dict[State, int] = {
            state: idx for idx, state in enumerate(self.states)
        }
        self.reference_index = self.index[(0, 0, 0, 0)]
        self._row_cache: dict[tuple[int, int], np.ndarray] = {}
        self._tick_cost_cache: dict[tuple[int, int], float] = {}

    @property
    def number_of_states(self) -> int:
        return len(self.states)

    @staticmethod
    def local_state(state: State, vm: int) -> tuple[int, int]:
        if vm == 1:
            return state[0], state[1]
        if vm == 2:
            return state[2], state[3]
        raise ValueError("This reference implementation supports VMs 1 and 2.")

    @staticmethod
    def swap_vms(state: State) -> State:
        return state[2], state[3], state[0], state[1]

    def feasible_actions(self, state: State) -> tuple[int, ...]:
        feasible = tuple(
            vm for vm in (1, 2) if self.local_state(state, vm)[0] < self.p.B
        )
        return feasible if feasible else (0,)

    def validate_action(self, state: State, action: int) -> None:
        if action not in self.feasible_actions(state):
            raise ValueError(
                f"Action {action} is infeasible in state {state}; "
                f"feasible actions are {self.feasible_actions(state)}."
            )

    def post_arrival(self, state: State, action: int) -> State:
        self.validate_action(state, action)
        if action == 0:
            return state
        successor = list(state)
        successor[0 if action == 1 else 2] += 1
        return tuple(successor)  # type: ignore[return-value]

    def arrival_impulse_cost(self, action: int) -> float:
        return self.p.blocking_cost if action == 0 else 0.0

    def holding_cost_rate(self, state: State) -> float:
        q1, n1, q2, n2 = state
        return (
            self.p.holding_vm * (q1 + q2)
            + self.p.holding_container * (n1 + n2)
        )

    def internal_events(self, state: State) -> tuple[tuple[float, State, str], ...]:
        """Return ``(rate, successor, event_name)`` for autonomous events."""

        events: list[tuple[float, State, str]] = []
        for vm in (1, 2):
            q, n = self.local_state(state, vm)
            q_pos, n_pos = (0, 1) if vm == 1 else (2, 3)

            if q > 0 and n < self.p.K:
                successor = list(state)
                successor[q_pos] -= 1
                successor[n_pos] += 1
                events.append(
                    (
                        self.p.mu_vm,
                        tuple(successor),  # type: ignore[arg-type]
                        f"promotion_vm{vm}",
                    )
                )

            if n > 0:
                successor = list(state)
                successor[n_pos] -= 1
                events.append(
                    (
                        n * self.p.mu_container,
                        tuple(successor),  # type: ignore[arg-type]
                        f"departure_vm{vm}",
                    )
                )
        return tuple(events)

    def total_internal_rate(self, state: State) -> float:
        return sum(rate for rate, _, _ in self.internal_events(state))

    def uniformized_row(self, state_index: int, action: int) -> np.ndarray:
        """Return one row of the constant-uniformized transition kernel."""

        key = (state_index, action)
        cached = self._row_cache.get(key)
        if cached is not None:
            return cached

        state = self.states[state_index]
        self.validate_action(state, action)
        nu = self.p.uniformization_rate
        row = np.zeros(self.number_of_states, dtype=float)

        internal = self.internal_events(state)
        internal_rate = sum(rate for rate, _, _ in internal)
        dummy_probability = 1.0 - (self.p.arrival_rate + internal_rate) / nu
        if dummy_probability < -1.0e-13:
            raise AssertionError(
                "Uniformization rate is too small: "
                f"dummy probability={dummy_probability} in state {state}."
            )
        row[state_index] += max(dummy_probability, 0.0)

        arrival_state = self.post_arrival(state, action)
        row[self.index[arrival_state]] += self.p.arrival_rate / nu

        for rate, successor, _ in internal:
            row[self.index[successor]] += rate / nu

        if np.min(row) < -1.0e-14 or not np.isclose(row.sum(), 1.0, atol=1e-12):
            raise AssertionError(
                f"Invalid uniformized row for state={state}, action={action}: "
                f"min={row.min()}, sum={row.sum()}."
            )

        row.setflags(write=False)
        self._row_cache[key] = row
        return row

    def tick_cost(self, state_index: int, action: int) -> float:
        key = (state_index, action)
        cached = self._tick_cost_cache.get(key)
        if cached is not None:
            return cached
        state = self.states[state_index]
        self.validate_action(state, action)
        result = (
            self.holding_cost_rate(state)
            + self.p.arrival_rate * self.arrival_impulse_cost(action)
        ) / self.p.uniformization_rate
        self._tick_cost_cache[key] = result
        return result

    def bellman_operator(self, bias: np.ndarray) -> tuple[np.ndarray, Policy]:
        if bias.shape != (self.number_of_states,):
            raise ValueError("Bias vector has the wrong shape.")
        values = np.empty(self.number_of_states, dtype=float)
        policy = np.empty(self.number_of_states, dtype=int)
        for idx, state in enumerate(self.states):
            actions = self.feasible_actions(state)
            action_values = [
                self.tick_cost(idx, action)
                + float(self.uniformized_row(idx, action) @ bias)
                for action in actions
            ]
            best_position = int(np.argmin(action_values))
            values[idx] = action_values[best_position]
            policy[idx] = actions[best_position]
        return values, policy

    def arrival_score(self, state: State, action: int, bias: np.ndarray) -> float:
        successor = self.post_arrival(state, action)
        return self.arrival_impulse_cost(action) + bias[self.index[successor]]

    def optimal_action_sets(
        self, bias: np.ndarray, tie_tolerance: float = 1.0e-10
    ) -> tuple[tuple[int, ...], ...]:
        result: list[tuple[int, ...]] = []
        for state in self.states:
            actions = self.feasible_actions(state)
            scores = np.array(
                [self.arrival_score(state, action, bias) for action in actions]
            )
            minimum = float(scores.min())
            optimal = tuple(
                action
                for action, score in zip(actions, scores)
                if score <= minimum + tie_tolerance
            )
            result.append(optimal)
        return tuple(result)

    def relative_value_iteration(
        self,
        tolerance: float = 1.0e-11,
        max_iterations: int = 200_000,
        tie_tolerance: float = 1.0e-10,
    ) -> RelativeValueResult:
        """Solve the constant-uniformized average-cost Bellman equation."""

        bias = np.zeros(self.number_of_states, dtype=float)
        span_residual = math.inf
        iteration = 0

        for iteration in range(1, max_iterations + 1):
            unnormalized, _ = self.bellman_operator(bias)
            next_bias = unnormalized - unnormalized[self.reference_index]
            difference = next_bias - bias
            span_residual = float(difference.max() - difference.min())
            bias = next_bias
            if span_residual <= tolerance:
                break
        else:
            raise RuntimeError(
                f"RVI did not converge after {max_iterations} iterations; "
                f"span residual={span_residual:.3e}."
            )

        bellman_values, _ = self.bellman_operator(bias)
        gain_per_tick = float(bellman_values[self.reference_index])
        gain = self.p.uniformization_rate * gain_per_tick
        residual_vector = bellman_values - bias - gain_per_tick
        bellman_residual = float(np.max(np.abs(residual_vector)))

        action_sets = self.optimal_action_sets(bias, tie_tolerance)
        policy = np.array([actions[0] for actions in action_sets], dtype=int)
        return RelativeValueResult(
            gain=gain,
            bias=bias,
            policy=policy,
            optimal_action_sets=action_sets,
            iterations=iteration,
            span_residual=span_residual,
            bellman_residual=bellman_residual,
        )

    def generator(self, policy: Policy) -> np.ndarray:
        self.validate_policy(policy)
        generator = np.zeros(
            (self.number_of_states, self.number_of_states), dtype=float
        )
        for idx, state in enumerate(self.states):
            action = int(policy[idx])
            arrival_state = self.post_arrival(state, action)
            arrival_idx = self.index[arrival_state]
            if arrival_idx != idx:
                generator[idx, arrival_idx] += self.p.arrival_rate

            for rate, successor, _ in self.internal_events(state):
                successor_idx = self.index[successor]
                if successor_idx != idx:
                    generator[idx, successor_idx] += rate

            generator[idx, idx] = -float(generator[idx, :].sum())
        return generator

    def validate_policy(self, policy: Policy) -> None:
        if policy.shape != (self.number_of_states,):
            raise ValueError(
                f"Policy must have shape ({self.number_of_states},), "
                f"not {policy.shape}."
            )
        for idx, state in enumerate(self.states):
            self.validate_action(state, int(policy[idx]))

    @staticmethod
    def _stationary_distribution(generator: np.ndarray) -> tuple[np.ndarray, float]:
        n_states = generator.shape[0]
        system = generator.T.copy()
        rhs = np.zeros(n_states, dtype=float)
        system[-1, :] = 1.0
        rhs[-1] = 1.0
        try:
            stationary = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            stationary, *_ = np.linalg.lstsq(system, rhs, rcond=None)

        if stationary.min() < -1.0e-9:
            raise AssertionError(
                "Stationary solution has a material negative entry: "
                f"{stationary.min():.3e}."
            )
        stationary = np.maximum(stationary, 0.0)
        stationary /= stationary.sum()
        residual = float(np.max(np.abs(stationary @ generator)))
        return stationary, residual

    def _poisson_gain(
        self, generator: np.ndarray, cost_rate: np.ndarray
    ) -> tuple[float, np.ndarray, float]:
        n_states = self.number_of_states
        system = np.zeros((n_states + 1, n_states + 1), dtype=float)
        rhs = np.zeros(n_states + 1, dtype=float)
        system[:n_states, :n_states] = generator
        system[:n_states, n_states] = -1.0
        rhs[:n_states] = -cost_rate
        system[n_states, self.reference_index] = 1.0
        solution = np.linalg.solve(system, rhs)
        bias = solution[:n_states]
        gain = float(solution[n_states])
        residual = float(
            np.max(np.abs(cost_rate + generator @ bias - gain))
        )
        return gain, bias, residual

    def evaluate_policy(self, policy: Policy, policy_name: str) -> PolicyEvaluation:
        """Evaluate a stationary policy exactly in continuous time."""

        generator = self.generator(policy)
        stationary, generator_residual = self._stationary_distribution(generator)
        cost_rate = np.array(
            [
                self.holding_cost_rate(state)
                + self.p.arrival_rate
                * self.arrival_impulse_cost(int(policy[idx]))
                for idx, state in enumerate(self.states)
            ],
            dtype=float,
        )
        gain = float(stationary @ cost_rate)
        poisson_gain, _, poisson_residual = self._poisson_gain(generator, cost_rate)

        q_values = np.array([state[0] + state[2] for state in self.states])
        c_values = np.array([state[1] + state[3] for state in self.states])
        mean_q = float(stationary @ q_values)
        mean_c = float(stationary @ c_values)
        mean_l = mean_q + mean_c
        blocked_indicator = (policy == 0).astype(float)
        blocking_probability = float(stationary @ blocked_indicator)

        admitted = self.p.arrival_rate * (1.0 - blocking_probability)
        promotion_rates = np.array(
            [
                self.p.mu_vm
                * (
                    int(state[0] > 0 and state[1] < self.p.K)
                    + int(state[2] > 0 and state[3] < self.p.K)
                )
                for state in self.states
            ],
            dtype=float,
        )
        promotions = float(stationary @ promotion_rates)
        departures = self.p.mu_container * mean_c
        response_time = mean_l / admitted if admitted > 0.0 else math.inf

        decomposed_gain = (
            self.p.holding_vm * mean_q
            + self.p.holding_container * mean_c
            + self.p.arrival_rate
            * self.p.blocking_cost
            * blocking_probability
        )

        return PolicyEvaluation(
            policy_name=policy_name,
            gain=gain,
            stationary=stationary,
            generator_residual=generator_residual,
            poisson_gain=poisson_gain,
            poisson_gain_error=max(
                abs(poisson_gain - gain), poisson_residual
            ),
            blocking_probability=blocking_probability,
            mean_vm_occupancy=mean_q,
            mean_container_occupancy=mean_c,
            mean_total_occupancy=mean_l,
            admitted_throughput=admitted,
            promotion_throughput=promotions,
            departure_throughput=departures,
            mean_response_time=response_time,
            flow_residual_arrival_promotion=abs(admitted - promotions),
            flow_residual_promotion_departure=abs(promotions - departures),
            cost_decomposition_error=abs(gain - decomposed_gain),
        )

    def policy_from_rule(
        self, rule: Callable[[State, tuple[int, ...]], int]
    ) -> Policy:
        policy = np.empty(self.number_of_states, dtype=int)
        for idx, state in enumerate(self.states):
            feasible = self.feasible_actions(state)
            action = int(rule(state, feasible))
            self.validate_action(state, action)
            policy[idx] = action
        return policy

    def jsq_policy(self) -> Policy:
        def rule(state: State, feasible: tuple[int, ...]) -> int:
            if feasible == (0,):
                return 0
            scores = {
                vm: sum(self.local_state(state, vm)) for vm in feasible
            }
            return min(feasible, key=lambda vm: (scores[vm], vm))

        return self.policy_from_rule(rule)

    def weighted_composition_policy(
        self, queue_weight: float, container_weight: float
    ) -> Policy:
        def rule(state: State, feasible: tuple[int, ...]) -> int:
            if feasible == (0,):
                return 0
            scores = {
                vm: queue_weight * self.local_state(state, vm)[0]
                + container_weight * self.local_state(state, vm)[1]
                for vm in feasible
            }
            return min(feasible, key=lambda vm: (scores[vm], vm))

        return self.policy_from_rule(rule)

    def stage_aware_lexicographic_policy(self) -> Policy:
        """Prioritize the shorter VM-side stage, then fewer busy containers.

        This rule is parameter-free.  For fixed ``K`` it is equivalently
        implemented by the scalar score ``(K + 1) * q_i + n_i``.
        """

        def rule(state: State, feasible: tuple[int, ...]) -> int:
            if feasible == (0,):
                return 0
            return min(
                feasible,
                key=lambda vm: (*self.local_state(state, vm), vm),
            )

        return self.policy_from_rule(rule)

    def policy_oracle_metrics(
        self,
        candidate_policy: Policy,
        optimal: RelativeValueResult,
        expert_stationary: np.ndarray,
    ) -> dict[str, float | int]:
        self.validate_policy(candidate_policy)
        eligible = np.array(
            [len(self.feasible_actions(state)) > 1 for state in self.states]
        )
        regrets = np.zeros(self.number_of_states, dtype=float)
        correct = np.ones(self.number_of_states, dtype=bool)

        for idx, state in enumerate(self.states):
            action = int(candidate_policy[idx])
            scores = {
                feasible_action: self.arrival_score(
                    state, feasible_action, optimal.bias
                )
                for feasible_action in self.feasible_actions(state)
            }
            regrets[idx] = scores[action] - min(scores.values())
            correct[idx] = action in optimal.optimal_action_sets[idx]

        if not np.any(eligible):
            raise AssertionError("No states have a genuine routing choice.")
        routing_weights = expert_stationary * eligible
        routing_mass = float(routing_weights.sum())
        if routing_mass > 0.0:
            routing_weights /= routing_mass

        return {
            "routing_states": int(eligible.sum()),
            "uniform_tie_aware_accuracy": float(correct[eligible].mean()),
            "uniform_mean_arrival_regret": float(regrets[eligible].mean()),
            "uniform_max_arrival_regret": float(regrets[eligible].max()),
            "expert_routing_state_mass": routing_mass,
            "expert_weighted_tie_aware_accuracy": float(
                routing_weights @ correct.astype(float)
            ),
            "expert_weighted_mean_arrival_regret": float(
                routing_weights @ regrets
            ),
        }

    def structural_audit(
        self,
        optimal: RelativeValueResult,
        tie_tolerance: float = 1.0e-10,
    ) -> tuple[dict[str, float | int], list[dict[str, object]]]:
        """Audit exact structure and return explicit composition examples."""

        bias = optimal.bias
        symmetry_errors = []
        antisymmetry_errors = []
        routing_ties = 0
        near_ties = 0
        deltas: dict[State, float] = {}

        for idx, state in enumerate(self.states):
            swapped_idx = self.index[self.swap_vms(state)]
            symmetry_errors.append(abs(bias[idx] - bias[swapped_idx]))
            if self.feasible_actions(state) == (1, 2):
                delta = self.arrival_score(state, 1, bias) - self.arrival_score(
                    state, 2, bias
                )
                deltas[state] = delta
                swapped = self.swap_vms(state)
                swapped_delta = self.arrival_score(
                    swapped, 1, bias
                ) - self.arrival_score(swapped, 2, bias)
                antisymmetry_errors.append(abs(delta + swapped_delta))
                routing_ties += int(abs(delta) <= tie_tolerance)
                near_ties += int(abs(delta) <= 1.0e-4)

        bias_comparisons = 0
        bias_violations = 0
        maximum_bias_violation = 0.0
        own_load_comparisons = 0
        own_load_violations = 0
        maximum_own_load_violation = 0.0

        for state in self.states:
            for coordinate, capacity in (
                (0, self.p.B),
                (1, self.p.K),
                (2, self.p.B),
                (3, self.p.K),
            ):
                if state[coordinate] >= capacity:
                    continue
                larger = list(state)
                larger[coordinate] += 1
                larger_state: State = tuple(larger)  # type: ignore[assignment]
                difference = bias[self.index[larger_state]] - bias[self.index[state]]
                bias_comparisons += 1
                if difference < -tie_tolerance:
                    bias_violations += 1
                    maximum_bias_violation = max(
                        maximum_bias_violation, -difference
                    )

            # Decision-relevant own-load audit for VM 1.  Both actions must
            # remain feasible in the original and enlarged state.
            if self.feasible_actions(state) == (1, 2):
                for coordinate, capacity in ((0, self.p.B), (1, self.p.K)):
                    if state[coordinate] >= capacity:
                        continue
                    larger = list(state)
                    larger[coordinate] += 1
                    larger_state = tuple(larger)  # type: ignore[assignment]
                    if self.feasible_actions(larger_state) != (1, 2):
                        continue
                    old_delta = deltas[state]
                    new_delta = deltas[larger_state]
                    own_load_comparisons += 1
                    if new_delta < old_delta - tie_tolerance:
                        own_load_violations += 1
                        maximum_own_load_violation = max(
                            maximum_own_load_violation,
                            old_delta - new_delta,
                        )

        composition_groups: dict[
            tuple[int, int, tuple[int, ...]], list[tuple[State, float, tuple[int, ...]]]
        ] = {}
        for idx, state in enumerate(self.states):
            if self.feasible_actions(state) != (1, 2):
                continue
            key = (
                state[0] + state[1],
                state[2] + state[3],
                self.feasible_actions(state),
            )
            composition_groups.setdefault(key, []).append(
                (state, deltas[state], optimal.optimal_action_sets[idx])
            )

        varying_classes = 0
        strict_reversal_classes = 0
        maximum_composition_delta_range = 0.0
        examples: list[dict[str, object]] = []
        for key, members in composition_groups.items():
            if len(members) < 2:
                continue
            action_sets = {actions for _, _, actions in members}
            member_deltas = [delta for _, delta, _ in members]
            delta_range = max(member_deltas) - min(member_deltas)
            maximum_composition_delta_range = max(
                maximum_composition_delta_range, delta_range
            )
            if len(action_sets) > 1:
                varying_classes += 1
            has_vm1 = any(actions == (1,) for _, _, actions in members)
            has_vm2 = any(actions == (2,) for _, _, actions in members)
            if has_vm1 and has_vm2:
                strict_reversal_classes += 1

            if len(action_sets) > 1 or delta_range > tie_tolerance:
                minimum_member = min(members, key=lambda item: item[1])
                maximum_member = max(members, key=lambda item: item[1])
                examples.append(
                    {
                        "local_total_vm1": key[0],
                        "local_total_vm2": key[1],
                        "class_size": len(members),
                        "delta_range": delta_range,
                        "minimum_delta_state": str(minimum_member[0]),
                        "minimum_delta": minimum_member[1],
                        "minimum_optimal_actions": str(minimum_member[2]),
                        "maximum_delta_state": str(maximum_member[0]),
                        "maximum_delta": maximum_member[1],
                        "maximum_optimal_actions": str(maximum_member[2]),
                        "strict_reversal": has_vm1 and has_vm2,
                    }
                )

        examples.sort(
            key=lambda row: (
                bool(row["strict_reversal"]), float(row["delta_range"])
            ),
            reverse=True,
        )

        summary: dict[str, float | int] = {
            "both_feasible_states": len(deltas),
            "exact_routing_ties": routing_ties,
            "near_routing_ties_1e-4": near_ties,
            "max_bias_swap_error": max(symmetry_errors, default=0.0),
            "max_advantage_antisymmetry_error": max(
                antisymmetry_errors, default=0.0
            ),
            "bias_monotonicity_comparisons": bias_comparisons,
            "bias_monotonicity_violations": bias_violations,
            "max_bias_monotonicity_violation": maximum_bias_violation,
            "own_load_delta_comparisons": own_load_comparisons,
            "own_load_delta_violations": own_load_violations,
            "max_own_load_delta_violation": maximum_own_load_violation,
            "composition_classes": len(composition_groups),
            "composition_classes_with_varying_optimal_sets": varying_classes,
            "composition_classes_with_strict_reversal": strict_reversal_classes,
            "max_composition_delta_range": maximum_composition_delta_range,
        }
        return summary, examples

    def oracle_rows(
        self,
        optimal: RelativeValueResult,
        jsq: Policy,
        stage_aware: Policy,
        optimal_stationary: np.ndarray,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for idx, state in enumerate(self.states):
            feasible = self.feasible_actions(state)
            score1 = (
                self.arrival_score(state, 1, optimal.bias)
                if 1 in feasible
                else math.nan
            )
            score2 = (
                self.arrival_score(state, 2, optimal.bias)
                if 2 in feasible
                else math.nan
            )
            delta = score1 - score2 if feasible == (1, 2) else math.nan
            rows.append(
                {
                    "q1": state[0],
                    "n1": state[1],
                    "q2": state[2],
                    "n2": state[3],
                    "total1": state[0] + state[1],
                    "total2": state[2] + state[3],
                    "feasible_actions": str(feasible),
                    "bias": optimal.bias[idx],
                    "arrival_score_1": score1,
                    "arrival_score_2": score2,
                    "delta_12": delta,
                    "optimal_actions": str(optimal.optimal_action_sets[idx]),
                    "deterministic_optimal_action": int(optimal.policy[idx]),
                    "jsq_action": int(jsq[idx]),
                    "jsq_is_optimal": int(
                        int(jsq[idx]) in optimal.optimal_action_sets[idx]
                    ),
                    "stage_aware_action": int(stage_aware[idx]),
                    "stage_aware_is_optimal": int(
                        int(stage_aware[idx]) in optimal.optimal_action_sets[idx]
                    ),
                    "optimal_stationary_probability": optimal_stationary[idx],
                }
            )
        return rows

    def run_invariant_tests(
        self,
        optimal: RelativeValueResult,
        evaluation: PolicyEvaluation,
        tolerance: float = 1.0e-8,
    ) -> dict[str, float | int]:
        minimum_probability = math.inf
        maximum_row_sum_error = 0.0
        checked_rows = 0
        for idx, state in enumerate(self.states):
            for action in self.feasible_actions(state):
                row = self.uniformized_row(idx, action)
                minimum_probability = min(minimum_probability, float(row.min()))
                maximum_row_sum_error = max(
                    maximum_row_sum_error, abs(float(row.sum()) - 1.0)
                )
                checked_rows += 1

        gain_error = abs(optimal.gain - evaluation.gain)
        checks = {
            "checked_uniformized_rows": checked_rows,
            "minimum_transition_probability": minimum_probability,
            "maximum_row_sum_error": maximum_row_sum_error,
            "rvi_stationary_gain_error": gain_error,
            "stationary_generator_residual": evaluation.generator_residual,
            "stationary_poisson_gain_error": evaluation.poisson_gain_error,
            "arrival_promotion_flow_error": evaluation.flow_residual_arrival_promotion,
            "promotion_departure_flow_error": evaluation.flow_residual_promotion_departure,
            "cost_decomposition_error": evaluation.cost_decomposition_error,
        }
        failures = {
            key: value
            for key, value in checks.items()
            if key != "checked_uniformized_rows"
            and (
                (key == "minimum_transition_probability" and value < -tolerance)
                or (key != "minimum_transition_probability" and value > tolerance)
            )
        }
        if failures:
            raise AssertionError(f"Correctness checks failed: {failures}")
        return checks

    def simulate(
        self,
        policy: Policy,
        policy_name: str,
        horizon: float,
        burn_in: float,
        seed: int,
    ) -> SimulationEstimate:
        """Continuous-time validation simulation with time-weighted statistics."""

        if horizon <= burn_in or burn_in < 0.0:
            raise ValueError("Require horizon > burn_in >= 0.")
        self.validate_policy(policy)
        rng = np.random.default_rng(seed)
        state: State = (0, 0, 0, 0)
        time = 0.0
        area_q = 0.0
        area_c = 0.0
        arrivals = 0
        blocked = 0
        events = 0

        while time < horizon:
            internal = self.internal_events(state)
            total_rate = self.p.arrival_rate + sum(
                rate for rate, _, _ in internal
            )
            holding_time = float(rng.exponential(1.0 / total_rate))
            next_time = min(time + holding_time, horizon)

            observed_start = max(time, burn_in)
            observed_end = max(min(next_time, horizon), burn_in)
            observed_duration = max(0.0, observed_end - observed_start)
            if observed_duration > 0.0:
                q_total = state[0] + state[2]
                c_total = state[1] + state[3]
                area_q += q_total * observed_duration
                area_c += c_total * observed_duration

            if time + holding_time > horizon:
                time = horizon
                break

            event_time = time + holding_time
            draw = float(rng.random()) * total_rate
            if draw < self.p.arrival_rate:
                if event_time >= burn_in:
                    arrivals += 1
                action = int(policy[self.index[state]])
                if action == 0:
                    if event_time >= burn_in:
                        blocked += 1
                else:
                    state = self.post_arrival(state, action)
            else:
                draw -= self.p.arrival_rate
                cumulative = 0.0
                selected_successor: State | None = None
                for rate, successor, _ in internal:
                    cumulative += rate
                    if draw < cumulative:
                        selected_successor = successor
                        break
                if selected_successor is None:
                    selected_successor = internal[-1][1]
                state = selected_successor

            time = event_time
            events += 1

        observation_time = horizon - burn_in
        mean_q = area_q / observation_time
        mean_c = area_c / observation_time
        blocking_probability = blocked / arrivals if arrivals else math.nan
        average_cost = (
            self.p.holding_vm * area_q
            + self.p.holding_container * area_c
            + self.p.blocking_cost * blocked
        ) / observation_time
        return SimulationEstimate(
            policy_name=policy_name,
            observation_time=observation_time,
            events=events,
            arrivals=arrivals,
            blocked=blocked,
            blocking_probability=blocking_probability,
            mean_vm_occupancy=mean_q,
            mean_container_occupancy=mean_c,
            mean_total_occupancy=mean_q + mean_c,
            average_cost=average_cost,
        )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assert_finite_summary(summary: Mapping[str, object]) -> None:
    for key, value in summary.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise AssertionError(f"Non-finite summary value {key}={value}.")


def run_scenario(
    parameters: ModelParameters,
    output_directory: Path,
    simulate: bool,
    simulation_horizon: float,
    simulation_burn_in: float,
    seed: int,
    rvi_tolerance: float,
) -> dict[str, object]:
    model = N2AverageCostCTMDP(parameters)
    optimal = model.relative_value_iteration(tolerance=rvi_tolerance)
    optimal_eval = model.evaluate_policy(optimal.policy, "optimal")
    invariant_checks = model.run_invariant_tests(optimal, optimal_eval)

    jsq = model.jsq_policy()
    jsq_eval = model.evaluate_policy(jsq, "jsq")
    jsq_oracle = model.policy_oracle_metrics(
        jsq, optimal, optimal_eval.stationary
    )
    stage_aware = model.stage_aware_lexicographic_policy()
    stage_aware_eval = model.evaluate_policy(stage_aware, "stage_aware")
    stage_aware_oracle = model.policy_oracle_metrics(
        stage_aware, optimal, optimal_eval.stationary
    )
    structural, composition_examples = model.structural_audit(optimal)

    scenario_id = (
        f"K{parameters.K}_B{parameters.B}_lambda{parameters.arrival_rate:g}_"
        f"muvm{parameters.mu_vm:g}_muc{parameters.mu_container:g}"
    ).replace(".", "p")
    write_csv(
        output_directory / "oracle_tables" / f"oracle_{scenario_id}.csv",
        model.oracle_rows(
            optimal, jsq, stage_aware, optimal_eval.stationary
        ),
    )
    write_csv(
        output_directory
        / "composition_examples"
        / f"composition_{scenario_id}.csv",
        composition_examples,
    )

    relative_gap = (
        (jsq_eval.gain - optimal_eval.gain) / optimal_eval.gain
        if optimal_eval.gain > 0.0
        else 0.0
    )
    stage_aware_relative_gap = (
        (stage_aware_eval.gain - optimal_eval.gain) / optimal_eval.gain
        if optimal_eval.gain > 0.0
        else 0.0
    )
    summary: dict[str, object] = {
        "scenario_id": scenario_id,
        "K": parameters.K,
        "B": parameters.B,
        "lambda": parameters.arrival_rate,
        "rho": parameters.rho,
        "mu_vm": parameters.mu_vm,
        "mu_container": parameters.mu_container,
        "holding_vm": parameters.holding_vm,
        "holding_container": parameters.holding_container,
        "blocking_cost": parameters.blocking_cost,
        "states": model.number_of_states,
        "uniformization_rate": parameters.uniformization_rate,
        "rvi_iterations": optimal.iterations,
        "rvi_span_residual": optimal.span_residual,
        "bellman_residual": optimal.bellman_residual,
        "optimal_gain": optimal_eval.gain,
        "jsq_gain": jsq_eval.gain,
        "jsq_absolute_gain_gap": jsq_eval.gain - optimal_eval.gain,
        "jsq_relative_gain_gap": relative_gap,
        "stage_aware_gain": stage_aware_eval.gain,
        "stage_aware_absolute_gain_gap": (
            stage_aware_eval.gain - optimal_eval.gain
        ),
        "stage_aware_relative_gain_gap": stage_aware_relative_gap,
        "optimal_blocking_probability": optimal_eval.blocking_probability,
        "jsq_blocking_probability": jsq_eval.blocking_probability,
        "stage_aware_blocking_probability": (
            stage_aware_eval.blocking_probability
        ),
        "optimal_mean_total_occupancy": optimal_eval.mean_total_occupancy,
        "jsq_mean_total_occupancy": jsq_eval.mean_total_occupancy,
        "stage_aware_mean_total_occupancy": (
            stage_aware_eval.mean_total_occupancy
        ),
        **{f"jsq_{key}": value for key, value in jsq_oracle.items()},
        **{
            f"stage_aware_{key}": value
            for key, value in stage_aware_oracle.items()
        },
        **structural,
        **{f"check_{key}": value for key, value in invariant_checks.items()},
    }

    if simulate:
        optimal_sim = model.simulate(
            optimal.policy,
            "optimal",
            horizon=simulation_horizon,
            burn_in=simulation_burn_in,
            seed=seed,
        )
        jsq_sim = model.simulate(
            jsq,
            "jsq",
            horizon=simulation_horizon,
            burn_in=simulation_burn_in,
            seed=seed,
        )
        stage_aware_sim = model.simulate(
            stage_aware,
            "stage_aware",
            horizon=simulation_horizon,
            burn_in=simulation_burn_in,
            seed=seed,
        )
        write_csv(
            output_directory / "simulation" / f"simulation_{scenario_id}.csv",
            [
                optimal_sim.summary_dict(),
                jsq_sim.summary_dict(),
                stage_aware_sim.summary_dict(),
            ],
        )
        summary.update(
            {
                "simulation_optimal_gain": optimal_sim.average_cost,
                "simulation_jsq_gain": jsq_sim.average_cost,
                "simulation_optimal_gain_error": (
                    optimal_sim.average_cost - optimal_eval.gain
                ),
                "simulation_jsq_gain_error": (
                    jsq_sim.average_cost - jsq_eval.gain
                ),
                "simulation_stage_aware_gain": stage_aware_sim.average_cost,
                "simulation_stage_aware_gain_error": (
                    stage_aware_sim.average_cost - stage_aware_eval.gain
                ),
            }
        )

    assert_finite_summary(summary)
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve and audit the corrected N=2 average-cost cloud CTMDP."
        )
    )
    parser.add_argument("--K", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--B", nargs="+", type=int, default=[1, 2])
    parser.add_argument(
        "--lambdas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0]
    )
    parser.add_argument("--mu-vm", type=float, default=1.0)
    parser.add_argument("--mu-container", type=float, default=1.0)
    parser.add_argument("--holding-vm", type=float, default=1.0)
    parser.add_argument("--holding-container", type=float, default=1.0)
    parser.add_argument("--blocking-cost", type=float, default=10.0)
    parser.add_argument("--uniformization-slack", type=float, default=1.0e-3)
    parser.add_argument("--rvi-tolerance", type=float, default=1.0e-11)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("corrected_n2_results")
    )
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--simulation-horizon", type=float, default=50_000.0)
    parser.add_argument("--simulation-burn-in", type=float, default=5_000.0)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []

    for B in args.B:
        for K in args.K:
            for arrival_rate in args.lambdas:
                parameters = ModelParameters(
                    K=K,
                    B=B,
                    arrival_rate=arrival_rate,
                    mu_vm=args.mu_vm,
                    mu_container=args.mu_container,
                    holding_vm=args.holding_vm,
                    holding_container=args.holding_container,
                    blocking_cost=args.blocking_cost,
                    uniformization_slack=args.uniformization_slack,
                )
                summary = run_scenario(
                    parameters=parameters,
                    output_directory=args.output_dir,
                    simulate=args.simulate,
                    simulation_horizon=args.simulation_horizon,
                    simulation_burn_in=args.simulation_burn_in,
                    seed=args.seed,
                    rvi_tolerance=args.rvi_tolerance,
                )
                summaries.append(summary)
                print(
                    f"B={B:>2d}, K={K:>2d}, lambda={arrival_rate:>6g}, "
                    f"rho={float(summary['rho']):.3f}, "
                    f"g*={float(summary['optimal_gain']):.6f}, "
                    f"g_JSQ={float(summary['jsq_gain']):.6f}, "
                    f"JSQ gap={100.0 * float(summary['jsq_relative_gain_gap']):.3f}%, "
                    f"stage gap={100.0 * float(summary['stage_aware_relative_gain_gap']):.3f}%, "
                    f"ties={int(summary['exact_routing_ties'])}"
                )

    write_csv(args.output_dir / "scenario_summary.csv", summaries)
    metadata = {
        "model": "corrected_N2_average_cost_CTMDP",
        "state_order": ["q1", "n1", "q2", "n2"],
        "parameters": {
            "K": args.K,
            "B": args.B,
            "lambdas": args.lambdas,
            "mu_vm": args.mu_vm,
            "mu_container": args.mu_container,
            "holding_vm": args.holding_vm,
            "holding_container": args.holding_container,
            "blocking_cost": args.blocking_cost,
            "uniformization_slack": args.uniformization_slack,
        },
        "simulation_enabled": args.simulate,
        "seed": args.seed,
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved corrected exploration to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
