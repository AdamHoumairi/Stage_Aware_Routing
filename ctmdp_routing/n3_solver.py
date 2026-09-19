#!/usr/bin/env python3
"""Finite-state sparse solver for the corrected N=3 cloud CTMDP.

The model is the direct three-VM extension of ``n2_average_cost_ctmdp.py``.
Each job accepted at VM i follows its finite VM-side stage (q_i) and then one
of K_i parallel containers (n_i).  Routing decisions occur only at external
arrivals.  The solver uses constant-rate uniformization, relative value
iteration, and stationary CTMC evaluation without simulation.

The implementation supports homogeneous and service-rate-heterogeneous VMs.
It is intentionally separate from the N=2 reference solver so that the
validated N=2 evidence remains immutable.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from itertools import permutations, product
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse


State = tuple[int, ...]
Policy = np.ndarray
ActionProbabilities = np.ndarray
SCRIPT_VERSION = "1.0"


@dataclass(frozen=True)
class N3ModelParameters:
    """Parameters for a three-VM, two-stage finite-buffer system."""

    K: tuple[int, int, int] = (2, 2, 2)
    B: tuple[int, int, int] = (2, 2, 2)
    arrival_rate: float = 1.0
    mu_vm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mu_container: tuple[float, float, float] = (1.0, 1.0, 1.0)
    holding_vm: float = 1.0
    holding_container: float = 1.0
    blocking_cost: float = 10.0
    uniformization_slack: float = 1.0e-3

    def __post_init__(self) -> None:
        if len(self.K) != 3 or len(self.B) != 3:
            raise ValueError("N3ModelParameters requires exactly three VMs.")
        if len(self.mu_vm) != 3 or len(self.mu_container) != 3:
            raise ValueError("Exactly three service rates are required.")
        if any(value < 1 for value in (*self.K, *self.B)):
            raise ValueError("All K_i and B_i must be positive integers.")
        if self.arrival_rate <= 0.0:
            raise ValueError("The arrival rate must be strictly positive.")
        if any(value <= 0.0 for value in (*self.mu_vm, *self.mu_container)):
            raise ValueError("All service rates must be strictly positive.")
        if min(
            self.holding_vm,
            self.holding_container,
            self.blocking_cost,
            self.uniformization_slack,
        ) < 0.0:
            raise ValueError("Costs and uniformization slack must be nonnegative.")

    @classmethod
    def homogeneous(
        cls,
        *,
        K: int,
        B: int,
        arrival_rate: float,
        mu_vm: float,
        mu_container: float = 1.0,
        holding_vm: float = 1.0,
        holding_container: float = 1.0,
        blocking_cost: float = 10.0,
        uniformization_slack: float = 1.0e-3,
    ) -> "N3ModelParameters":
        return cls(
            K=(K, K, K),
            B=(B, B, B),
            arrival_rate=arrival_rate,
            mu_vm=(mu_vm, mu_vm, mu_vm),
            mu_container=(mu_container, mu_container, mu_container),
            holding_vm=holding_vm,
            holding_container=holding_container,
            blocking_cost=blocking_cost,
            uniformization_slack=uniformization_slack,
        )

    @property
    def number_of_vms(self) -> int:
        return 3

    @property
    def nominal_capacity(self) -> float:
        return sum(
            min(self.mu_vm[i], self.K[i] * self.mu_container[i])
            for i in range(3)
        )

    @property
    def rho(self) -> float:
        return self.arrival_rate / self.nominal_capacity

    @property
    def stage_capacity_ratios(self) -> tuple[float, float, float]:
        return tuple(
            self.mu_vm[i] / (self.K[i] * self.mu_container[i])
            for i in range(3)
        )

    @property
    def homogeneous_system(self) -> bool:
        return (
            len(set(self.K)) == 1
            and len(set(self.B)) == 1
            and max(self.mu_vm) - min(self.mu_vm) <= 1.0e-13
            and max(self.mu_container) - min(self.mu_container) <= 1.0e-13
        )

    @property
    def bottleneck_stage(self) -> str:
        ratios = self.stage_capacity_ratios
        if max(ratios) < 1.0 - 1.0e-12:
            return "vm"
        if min(ratios) > 1.0 + 1.0e-12:
            return "container"
        if max(abs(ratio - 1.0) for ratio in ratios) <= 1.0e-12:
            return "balanced"
        return "mixed"

    @property
    def uniformization_rate(self) -> float:
        maximum_internal = sum(
            max(
                self.K[i] * self.mu_container[i],
                self.mu_vm[i] + (self.K[i] - 1) * self.mu_container[i],
            )
            for i in range(3)
        )
        base = self.arrival_rate + maximum_internal
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
    stationary_iterations: int
    poisson_residual: float | None = None
    poisson_gain_error: float | None = None

    def summary_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.pop("stationary")
        return result


@dataclass
class LocalBiasResult:
    vm: int
    arrival_share: float
    gain: float
    marginal_scores: dict[tuple[int, int], float]
    poisson_residual: float


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

    def summary_dict(self) -> dict[str, object]:
        return asdict(self)


def relative_gain_gap(candidate_gain: float, optimal_gain: float) -> float:
    denominator = max(abs(optimal_gain), 1.0e-15)
    return max(0.0, (candidate_gain - optimal_gain) / denominator)


def action_set_signature(action_sets: Sequence[Sequence[int]]) -> str:
    encoded = ";".join(",".join(map(str, actions)) for actions in action_sets)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class N3AverageCostCTMDP:
    """Sparse exact CTMDP solver specialized to three VMs."""

    def __init__(self, parameters: N3ModelParameters):
        self.p = parameters
        local_domains = [
            tuple(product(range(self.p.B[i] + 1), range(self.p.K[i] + 1)))
            for i in range(3)
        ]
        self.states: tuple[State, ...] = tuple(
            tuple(component for pair in local_tuple for component in pair)
            for local_tuple in product(*local_domains)
        )
        self.index = {state: idx for idx, state in enumerate(self.states)}
        self.state_array = np.asarray(self.states, dtype=np.int16).reshape(-1, 3, 2)
        self.reference_index = self.index[(0, 0, 0, 0, 0, 0)]
        self.number_of_states = len(self.states)
        self.feasible_mask = np.column_stack(
            [self.state_array[:, i, 0] < self.p.B[i] for i in range(3)]
        )
        self.arrival_successors = np.full(
            (self.number_of_states, 3), -1, dtype=np.int64
        )
        for idx, state in enumerate(self.states):
            for vm in range(3):
                if self.feasible_mask[idx, vm]:
                    successor = list(state)
                    successor[2 * vm] += 1
                    self.arrival_successors[idx, vm] = self.index[tuple(successor)]

        self.holding_rates = (
            self.p.holding_vm * self.state_array[:, :, 0].sum(axis=1)
            + self.p.holding_container * self.state_array[:, :, 1].sum(axis=1)
        ).astype(float)
        self._internal_generator, internal_exit_rates = self._build_internal_generator()
        dummy = 1.0 - (
            self.p.arrival_rate + internal_exit_rates
        ) / self.p.uniformization_rate
        if float(dummy.min()) < -1.0e-12:
            raise AssertionError(
                f"Uniformization rate too small; minimum dummy={dummy.min():.3e}."
            )
        internal_offdiag = self._internal_generator.copy().tolil()
        internal_offdiag.setdiag(0.0)
        internal_offdiag = internal_offdiag.tocsr()
        self._uniformized_base = (
            internal_offdiag / self.p.uniformization_rate
            + sparse.diags(np.maximum(dummy, 0.0), format="csr")
        )

    def local_state(self, state: State, vm: int) -> tuple[int, int]:
        if vm not in (1, 2, 3):
            raise ValueError("VM must be 1, 2, or 3.")
        pos = 2 * (vm - 1)
        return state[pos], state[pos + 1]

    def feasible_actions(self, state: State) -> tuple[int, ...]:
        feasible = tuple(
            vm for vm in (1, 2, 3)
            if state[2 * (vm - 1)] < self.p.B[vm - 1]
        )
        return feasible if feasible else (0,)

    def post_arrival(self, state: State, action: int) -> State:
        if action not in self.feasible_actions(state):
            raise ValueError(f"Infeasible action {action} in state {state}.")
        if action == 0:
            return state
        successor = list(state)
        successor[2 * (action - 1)] += 1
        return tuple(successor)

    def internal_events(self, state: State) -> tuple[tuple[float, State], ...]:
        """List autonomous promotion and departure events explicitly."""

        events: list[tuple[float, State]] = []
        for vm in range(3):
            q, n = state[2 * vm], state[2 * vm + 1]
            if q > 0 and n < self.p.K[vm]:
                successor = list(state)
                successor[2 * vm] -= 1
                successor[2 * vm + 1] += 1
                events.append((self.p.mu_vm[vm], tuple(successor)))
            if n > 0:
                successor = list(state)
                successor[2 * vm + 1] -= 1
                events.append((n * self.p.mu_container[vm], tuple(successor)))
        return tuple(events)

    def _build_internal_generator(self) -> tuple[sparse.csr_matrix, np.ndarray]:
        rows: list[int] = []
        columns: list[int] = []
        data: list[float] = []
        exit_rates = np.zeros(self.number_of_states, dtype=float)
        for idx, state in enumerate(self.states):
            for vm in range(3):
                q, n = state[2 * vm], state[2 * vm + 1]
                if q > 0 and n < self.p.K[vm]:
                    successor = list(state)
                    successor[2 * vm] -= 1
                    successor[2 * vm + 1] += 1
                    rate = self.p.mu_vm[vm]
                    rows.append(idx)
                    columns.append(self.index[tuple(successor)])
                    data.append(rate)
                    exit_rates[idx] += rate
                if n > 0:
                    successor = list(state)
                    successor[2 * vm + 1] -= 1
                    rate = n * self.p.mu_container[vm]
                    rows.append(idx)
                    columns.append(self.index[tuple(successor)])
                    data.append(rate)
                    exit_rates[idx] += rate
        rows.extend(range(self.number_of_states))
        columns.extend(range(self.number_of_states))
        data.extend((-exit_rates).tolist())
        generator = sparse.csr_matrix(
            (data, (rows, columns)),
            shape=(self.number_of_states, self.number_of_states),
        )
        return generator, exit_rates

    def _arrival_scores(self, bias: np.ndarray) -> np.ndarray:
        scores = np.full((self.number_of_states, 3), np.inf, dtype=float)
        for vm in range(3):
            mask = self.feasible_mask[:, vm]
            scores[mask, vm] = bias[self.arrival_successors[mask, vm]]
        return scores

    def bellman_operator(self, bias: np.ndarray) -> tuple[np.ndarray, Policy]:
        if bias.shape != (self.number_of_states,):
            raise ValueError("Bias vector has the wrong shape.")
        scores = self._arrival_scores(bias)
        any_feasible = self.feasible_mask.any(axis=1)
        best_scores = np.empty(self.number_of_states, dtype=float)
        policy = np.zeros(self.number_of_states, dtype=np.int8)
        best_scores[any_feasible] = scores[any_feasible].min(axis=1)
        policy[any_feasible] = (
            scores[any_feasible].argmin(axis=1) + 1
        ).astype(np.int8)
        blocked = ~any_feasible
        best_scores[blocked] = self.p.blocking_cost + bias[blocked]
        common = (
            self.holding_rates / self.p.uniformization_rate
            + self._uniformized_base @ bias
        )
        values = common + (
            self.p.arrival_rate / self.p.uniformization_rate
        ) * best_scores
        return np.asarray(values).ravel(), policy

    def optimal_action_sets(
        self, bias: np.ndarray, tie_tolerance: float = 1.0e-10
    ) -> tuple[tuple[int, ...], ...]:
        scores = self._arrival_scores(bias)
        result: list[tuple[int, ...]] = []
        for idx in range(self.number_of_states):
            feasible = np.flatnonzero(self.feasible_mask[idx])
            if feasible.size == 0:
                result.append((0,))
                continue
            minimum = float(scores[idx, feasible].min())
            result.append(
                tuple(
                    int(vm + 1)
                    for vm in feasible
                    if scores[idx, vm] <= minimum + tie_tolerance
                )
            )
        return tuple(result)

    def relative_value_iteration(
        self,
        tolerance: float = 1.0e-11,
        max_iterations: int = 300_000,
        tie_tolerance: float = 1.0e-10,
    ) -> RelativeValueResult:
        bias = np.zeros(self.number_of_states, dtype=float)
        span_residual = math.inf
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
                f"RVI failed after {max_iterations} iterations; "
                f"span={span_residual:.3e}."
            )
        bellman_values, _ = self.bellman_operator(bias)
        gain_per_tick = float(bellman_values[self.reference_index])
        residual = bellman_values - bias - gain_per_tick
        action_sets = self.optimal_action_sets(bias, tie_tolerance)
        policy = np.asarray([actions[0] for actions in action_sets], dtype=np.int8)
        return RelativeValueResult(
            gain=self.p.uniformization_rate * gain_per_tick,
            bias=bias,
            policy=policy,
            optimal_action_sets=action_sets,
            iterations=iteration,
            span_residual=span_residual,
            bellman_residual=float(np.max(np.abs(residual))),
        )

    def validate_policy(self, policy: Policy) -> None:
        if policy.shape != (self.number_of_states,):
            raise ValueError("Policy has the wrong shape.")
        for idx, action in enumerate(policy.astype(int)):
            if action == 0:
                if self.feasible_mask[idx].any():
                    raise ValueError(f"Blocking is infeasible in state {self.states[idx]}.")
            elif not self.feasible_mask[idx, action - 1]:
                raise ValueError(
                    f"Action {action} is infeasible in state {self.states[idx]}."
                )

    def deterministic_action_probabilities(self, policy: Policy) -> ActionProbabilities:
        self.validate_policy(policy)
        probabilities = np.zeros((self.number_of_states, 4), dtype=float)
        probabilities[np.arange(self.number_of_states), policy.astype(int)] = 1.0
        return probabilities

    def validate_action_probabilities(self, probabilities: ActionProbabilities) -> None:
        if probabilities.shape != (self.number_of_states, 4):
            raise ValueError("Action-probability array has the wrong shape.")
        if float(probabilities.min()) < -1.0e-13:
            raise ValueError("Action probabilities cannot be negative.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-12):
            raise ValueError("Every action-probability row must sum to one.")
        for action in range(4):
            if action == 0:
                feasible = ~self.feasible_mask.any(axis=1)
            else:
                feasible = self.feasible_mask[:, action - 1]
            if np.any(probabilities[~feasible, action] > 1.0e-13):
                raise ValueError(f"Positive mass assigned to infeasible action {action}.")

    def generator_from_action_probabilities(
        self, probabilities: ActionProbabilities
    ) -> sparse.csr_matrix:
        self.validate_action_probabilities(probabilities)
        generator = self._internal_generator.tolil(copy=True)
        arrival_exit = np.zeros(self.number_of_states, dtype=float)
        for vm in range(3):
            mask = self.feasible_mask[:, vm] & (probabilities[:, vm + 1] > 0.0)
            rows = np.flatnonzero(mask)
            values = self.p.arrival_rate * probabilities[mask, vm + 1]
            for row, column, value in zip(
                rows, self.arrival_successors[mask, vm], values
            ):
                generator[int(row), int(column)] += float(value)
            arrival_exit[mask] += values
        current_diagonal = np.asarray(generator.diagonal()).ravel()
        generator.setdiag(current_diagonal - arrival_exit)
        return generator.tocsr()

    def generator(self, policy: Policy) -> sparse.csr_matrix:
        return self.generator_from_action_probabilities(
            self.deterministic_action_probabilities(policy)
        )

    def _stationary_distribution(
        self,
        generator: sparse.csr_matrix,
        tolerance: float = 5.0e-15,
        max_iterations: int = 2_000_000,
    ) -> tuple[np.ndarray, float, int]:
        transition = sparse.eye(self.number_of_states, format="csr") + (
            generator / self.p.uniformization_rate
        )
        minimum = float(transition.data.min()) if transition.nnz else 0.0
        row_error = float(
            np.max(np.abs(np.asarray(transition.sum(axis=1)).ravel() - 1.0))
        )
        if minimum < -1.0e-13 or row_error > 1.0e-12:
            raise AssertionError(
                f"Invalid evaluation kernel: min={minimum:.3e}, "
                f"row error={row_error:.3e}."
            )
        stationary = np.full(self.number_of_states, 1.0 / self.number_of_states)
        transposed = transition.T.tocsr()
        for iteration in range(1, max_iterations + 1):
            next_stationary = np.asarray(transposed @ stationary).ravel()
            next_stationary = np.maximum(next_stationary, 0.0)
            next_stationary /= next_stationary.sum()
            if float(np.max(np.abs(next_stationary - stationary))) <= tolerance:
                stationary = next_stationary
                break
            stationary = next_stationary
        else:
            raise RuntimeError("Stationary power iteration did not converge.")
        residual = float(
            np.max(np.abs(np.asarray(stationary @ generator).ravel()))
        )
        return stationary, residual, iteration

    def _poisson_check(
        self,
        generator: sparse.csr_matrix,
        cost_rate: np.ndarray,
    ) -> tuple[float, float]:
        """Solve the sparse augmented Poisson system with unknown gain."""

        count = self.number_of_states
        gain_column = sparse.csr_matrix(-np.ones((count, 1), dtype=float))
        normalization_row = sparse.csr_matrix(
            ([1.0], ([0], [self.reference_index])), shape=(1, count)
        )
        system = sparse.bmat(
            [
                [generator, gain_column],
                [normalization_row, sparse.csr_matrix((1, 1))],
            ],
            format="csc",
        )
        solution = sparse.linalg.spsolve(system, np.r_[-cost_rate, 0.0])
        bias = solution[:count]
        poisson_gain = float(solution[count])
        residual = float(
            np.max(np.abs(cost_rate + generator @ bias - poisson_gain))
        )
        return poisson_gain, residual

    def evaluate_action_probabilities(
        self,
        probabilities: ActionProbabilities,
        policy_name: str,
        *,
        poisson_check: bool = False,
    ) -> PolicyEvaluation:
        generator = self.generator_from_action_probabilities(probabilities)
        stationary, generator_residual, stationary_iterations = (
            self._stationary_distribution(generator)
        )
        blocking_by_state = probabilities[:, 0]
        cost_rate = (
            self.holding_rates
            + self.p.arrival_rate * self.p.blocking_cost * blocking_by_state
        )
        gain = float(stationary @ cost_rate)
        q_values = self.state_array[:, :, 0].sum(axis=1).astype(float)
        n_values = self.state_array[:, :, 1].sum(axis=1).astype(float)
        mean_q = float(stationary @ q_values)
        mean_n = float(stationary @ n_values)
        blocking = float(stationary @ blocking_by_state)
        admitted = self.p.arrival_rate * (1.0 - blocking)
        promotion_rates = np.zeros(self.number_of_states, dtype=float)
        departure_rates = np.zeros(self.number_of_states, dtype=float)
        for vm in range(3):
            q = self.state_array[:, vm, 0]
            n = self.state_array[:, vm, 1]
            promotion_rates += self.p.mu_vm[vm] * (
                (q > 0) & (n < self.p.K[vm])
            )
            departure_rates += self.p.mu_container[vm] * n
        promotions = float(stationary @ promotion_rates)
        departures = float(stationary @ departure_rates)
        decomposed = (
            self.p.holding_vm * mean_q
            + self.p.holding_container * mean_n
            + self.p.arrival_rate * self.p.blocking_cost * blocking
        )
        if poisson_check:
            poisson_gain, poisson_residual = self._poisson_check(
                generator, cost_rate
            )
            poisson_gain_error = abs(poisson_gain - gain)
        else:
            poisson_residual = None
            poisson_gain_error = None
        return PolicyEvaluation(
            policy_name=policy_name,
            gain=gain,
            stationary=stationary,
            generator_residual=generator_residual,
            blocking_probability=blocking,
            mean_vm_occupancy=mean_q,
            mean_container_occupancy=mean_n,
            mean_total_occupancy=mean_q + mean_n,
            admitted_throughput=admitted,
            promotion_throughput=promotions,
            departure_throughput=departures,
            mean_response_time=(mean_q + mean_n) / admitted,
            flow_residual_arrival_promotion=abs(admitted - promotions),
            flow_residual_promotion_departure=abs(promotions - departures),
            cost_decomposition_error=abs(gain - decomposed),
            stationary_iterations=stationary_iterations,
            poisson_residual=poisson_residual,
            poisson_gain_error=poisson_gain_error,
        )

    def evaluate_policy(
        self, policy: Policy, policy_name: str, *, poisson_check: bool = False
    ) -> PolicyEvaluation:
        return self.evaluate_action_probabilities(
            self.deterministic_action_probabilities(policy),
            policy_name,
            poisson_check=poisson_check,
        )

    def policy_from_rule(
        self, rule: Callable[[State, tuple[int, ...]], int]
    ) -> Policy:
        result = np.empty(self.number_of_states, dtype=np.int8)
        for idx, state in enumerate(self.states):
            result[idx] = rule(state, self.feasible_actions(state))
        self.validate_policy(result)
        return result

    def jsq_policy(self) -> Policy:
        def rule(state: State, feasible: tuple[int, ...]) -> int:
            if feasible == (0,):
                return 0
            return min(
                feasible,
                key=lambda vm: (sum(self.local_state(state, vm)), vm),
            )
        return self.policy_from_rule(rule)

    def randomized_tie_jsq_action_probabilities(self) -> ActionProbabilities:
        probabilities = np.zeros((self.number_of_states, 4), dtype=float)
        for idx, state in enumerate(self.states):
            feasible = self.feasible_actions(state)
            if feasible == (0,):
                probabilities[idx, 0] = 1.0
                continue
            totals = {vm: sum(self.local_state(state, vm)) for vm in feasible}
            minimum = min(totals.values())
            minimizers = [vm for vm in feasible if totals[vm] == minimum]
            for vm in minimizers:
                probabilities[idx, vm] = 1.0 / len(minimizers)
        self.validate_action_probabilities(probabilities)
        return probabilities

    def stage_aware_policy(self) -> Policy:
        def rule(state: State, feasible: tuple[int, ...]) -> int:
            if feasible == (0,):
                return 0
            return min(
                feasible,
                key=lambda vm: (*self.local_state(state, vm), vm),
            )
        return self.policy_from_rule(rule)

    def rate_weighted_backlog_policy(self) -> Policy:
        """Route by capacity-normalized upstream and downstream backlog."""

        def rule(state: State, feasible: tuple[int, ...]) -> int:
            if feasible == (0,):
                return 0
            def score(vm: int) -> float:
                q, n = self.local_state(state, vm)
                i = vm - 1
                return (
                    q / self.p.mu_vm[i]
                    + n / (self.p.K[i] * self.p.mu_container[i])
                )
            return min(feasible, key=lambda vm: (score(vm), vm))
        return self.policy_from_rule(rule)

    def decoupled_local_bias(
        self, vm: int, arrival_share: float
    ) -> LocalBiasResult:
        if vm not in (1, 2, 3) or arrival_share <= 0.0:
            raise ValueError("Invalid VM or arrival share.")
        i = vm - 1
        local_lambda = arrival_share * self.p.arrival_rate
        states = tuple(product(range(self.p.B[i] + 1), range(self.p.K[i] + 1)))
        index = {state: idx for idx, state in enumerate(states)}
        count = len(states)
        generator = np.zeros((count, count), dtype=float)
        cost_rate = np.zeros(count, dtype=float)
        for idx, (q, n) in enumerate(states):
            cost_rate[idx] = (
                self.p.holding_vm * q
                + self.p.holding_container * n
                + local_lambda * self.p.blocking_cost * int(q == self.p.B[i])
            )
            if q < self.p.B[i]:
                generator[idx, index[(q + 1, n)]] += local_lambda
            if q > 0 and n < self.p.K[i]:
                generator[idx, index[(q - 1, n + 1)]] += self.p.mu_vm[i]
            if n > 0:
                generator[idx, index[(q, n - 1)]] += n * self.p.mu_container[i]
            generator[idx, idx] = -generator[idx].sum()
        augmented = np.zeros((count + 1, count + 1), dtype=float)
        rhs = np.zeros(count + 1, dtype=float)
        augmented[:count, :count] = -generator
        augmented[:count, count] = 1.0
        rhs[:count] = cost_rate
        augmented[count, index[(0, 0)]] = 1.0
        solution = np.linalg.solve(augmented, rhs)
        bias = solution[:count]
        gain = float(solution[count])
        residual = float(np.max(np.abs(cost_rate + generator @ bias - gain)))
        marginal = {
            (q, n): float(bias[index[(q + 1, n)]] - bias[index[(q, n)]])
            for q in range(self.p.B[i])
            for n in range(self.p.K[i] + 1)
        }
        return LocalBiasResult(
            vm=vm,
            arrival_share=arrival_share,
            gain=gain,
            marginal_scores=marginal,
            poisson_residual=residual,
        )

    def dmb_policy(
        self, arrival_share: float
    ) -> tuple[Policy, tuple[LocalBiasResult, ...]]:
        local = tuple(
            self.decoupled_local_bias(vm, arrival_share) for vm in (1, 2, 3)
        )
        def rule(state: State, feasible: tuple[int, ...]) -> int:
            if feasible == (0,):
                return 0
            return min(
                feasible,
                key=lambda vm: (
                    local[vm - 1].marginal_scores[self.local_state(state, vm)],
                    vm,
                ),
            )
        return self.policy_from_rule(rule), local

    def adaptive_policy(
        self, frozen_arrival_share: float = 0.30
    ) -> tuple[Policy, str, tuple[LocalBiasResult, ...] | None]:
        if max(self.p.stage_capacity_ratios) < 1.0 - 1.0e-12:
            return self.stage_aware_policy(), "stage_aware", None
        policy, local = self.dmb_policy(frozen_arrival_share)
        return policy, "frozen_n2_dmb", local

    def arrival_score(self, state: State, action: int, bias: np.ndarray) -> float:
        if action == 0:
            return self.p.blocking_cost + float(bias[self.index[state]])
        return float(bias[self.index[self.post_arrival(state, action)]])

    def policy_oracle_metrics(
        self,
        policy: Policy,
        optimal: RelativeValueResult,
        optimal_stationary: np.ndarray,
    ) -> dict[str, float | int]:
        self.validate_policy(policy)
        choice = self.feasible_mask.sum(axis=1) >= 2
        regrets = np.zeros(self.number_of_states, dtype=float)
        correct = np.ones(self.number_of_states, dtype=bool)
        for idx in np.flatnonzero(choice):
            action = int(policy[idx])
            correct[idx] = action in optimal.optimal_action_sets[idx]
            candidate = self.arrival_score(self.states[idx], action, optimal.bias)
            best = min(
                self.arrival_score(self.states[idx], a, optimal.bias)
                for a in self.feasible_actions(self.states[idx])
            )
            regrets[idx] = max(0.0, candidate - best)
        count = int(choice.sum())
        mass = float(optimal_stationary[choice].sum())
        return {
            "choice_states": count,
            "uniform_tie_aware_accuracy": float(correct[choice].mean()),
            "uniform_mean_arrival_regret": float(regrets[choice].mean()),
            "uniform_max_arrival_regret": float(regrets[choice].max()),
            "optimal_stationary_choice_mass": mass,
            "stationary_weighted_error_mass": float(
                optimal_stationary[choice & ~correct].sum()
            ),
            "stationary_weighted_arrival_regret": float(
                optimal_stationary @ regrets
            ),
        }

    def jsq_error_breakdown(
        self, optimal: RelativeValueResult
    ) -> dict[str, int]:
        jsq = self.jsq_policy()
        wrong = tie_related = strict_order = 0
        for idx, state in enumerate(self.states):
            feasible = self.feasible_actions(state)
            if len(feasible) < 2 or int(jsq[idx]) in optimal.optimal_action_sets[idx]:
                continue
            wrong += 1
            totals = {vm: sum(self.local_state(state, vm)) for vm in feasible}
            minimum = min(totals.values())
            minimizers = {vm for vm, total in totals.items() if total == minimum}
            if minimizers.intersection(optimal.optimal_action_sets[idx]):
                tie_related += 1
            else:
                strict_order += 1
        return {
            "wrong_choice_states": wrong,
            "tie_related_errors": tie_related,
            "strict_order_errors": strict_order,
        }

    def composition_reversal_audit(
        self,
        optimal: RelativeValueResult,
        max_certificates: int = 4,
    ) -> tuple[dict[str, int], list[dict[str, object]]]:
        groups: dict[
            tuple[tuple[int, int, int], tuple[int, ...]],
            list[int],
        ] = {}
        for idx, state in enumerate(self.states):
            feasible = self.feasible_actions(state)
            if len(feasible) < 2:
                continue
            totals = tuple(sum(self.local_state(state, vm)) for vm in (1, 2, 3))
            groups.setdefault((totals, feasible), []).append(idx)

        reversal_groups = 0
        strict_pairs = 0
        certificates: list[dict[str, object]] = []
        certificate_pairs = 0
        for (totals, feasible), members in groups.items():
            by_action: dict[int, list[int]] = {}
            for idx in members:
                actions = optimal.optimal_action_sets[idx]
                if len(actions) == 1:
                    by_action.setdefault(actions[0], []).append(idx)
            if len(by_action) < 2:
                continue
            reversal_groups += 1
            actions = sorted(by_action)
            for left_pos, left_action in enumerate(actions):
                for right_action in actions[left_pos + 1:]:
                    strict_pairs += len(by_action[left_action]) * len(by_action[right_action])
                    if certificate_pairs >= max_certificates:
                        continue
                    left_idx = by_action[left_action][0]
                    right_idx = by_action[right_action][0]
                    for label, idx in (("left", left_idx), ("right", right_idx)):
                        state = self.states[idx]
                        scores = {
                            action: self.arrival_score(state, action, optimal.bias)
                            for action in feasible
                        }
                        ordered = sorted(scores.items(), key=lambda item: item[1])
                        certificates.append(
                            {
                                "certificate_pair": certificate_pairs + 1,
                                "member": label,
                                "totals": str(totals),
                                "feasible_actions": str(feasible),
                                "state": str(state),
                                "optimal_action_set": str(optimal.optimal_action_sets[idx]),
                                "arrival_scores": str(scores),
                                "best_runner_up_margin": ordered[1][1] - ordered[0][1],
                            }
                        )
                    certificate_pairs += 1
                    if certificate_pairs >= max_certificates:
                        break
                if certificate_pairs >= max_certificates:
                    break
            if certificate_pairs >= max_certificates:
                continue
        return {
            "composition_equivalence_classes": len(groups),
            "composition_reversal_classes": reversal_groups,
            "strict_reversal_pairs": strict_pairs,
        }, certificates

    def b1_collapse_audit(self) -> dict[str, int | bool]:
        choice_indices = np.flatnonzero(self.feasible_mask.sum(axis=1) >= 2)
        violations = 0
        policy_mismatches = 0
        jsq = self.jsq_policy()
        stage = self.stage_aware_policy()
        for idx in choice_indices:
            for vm in np.flatnonzero(self.feasible_mask[idx]):
                if self.state_array[idx, vm, 0] != 0:
                    violations += 1
            if jsq[idx] != stage[idx]:
                policy_mismatches += 1
        applies = all(value == 1 for value in self.p.B)
        return {
            "applies": applies,
            "choice_states": int(choice_indices.size),
            "feasible_nonempty_feeder_violations": violations if applies else -1,
            "jsq_stage_policy_mismatches": policy_mismatches if applies else -1,
        }

    @staticmethod
    def _permute_state(state: State, order: tuple[int, int, int]) -> State:
        pairs = [(state[2 * i], state[2 * i + 1]) for i in range(3)]
        return tuple(component for i in order for component in pairs[i])

    def symmetry_audit(
        self, optimal: RelativeValueResult, tolerance: float = 1.0e-8
    ) -> dict[str, float | int | bool]:
        if not self.p.homogeneous_system:
            return {
                "applicable": False,
                "bias_max_error": 0.0,
                "action_set_violations": 0,
            }
        max_bias_error = 0.0
        action_violations = 0
        for order in permutations((0, 1, 2)):
            inverse = {old: new for new, old in enumerate(order)}
            for idx, state in enumerate(self.states):
                permuted = self._permute_state(state, order)
                target_idx = self.index[permuted]
                max_bias_error = max(
                    max_bias_error,
                    abs(float(optimal.bias[idx] - optimal.bias[target_idx])),
                )
                mapped = tuple(
                    0 if action == 0 else inverse[action - 1] + 1
                    for action in optimal.optimal_action_sets[idx]
                )
                if tuple(sorted(mapped)) != optimal.optimal_action_sets[target_idx]:
                    action_violations += 1
        return {
            "applicable": True,
            "bias_max_error": max_bias_error,
            "action_set_violations": action_violations,
            "passes": max_bias_error <= tolerance and action_violations == 0,
        }

    def invariant_checks(
        self,
        optimal: RelativeValueResult,
        optimal_evaluation: PolicyEvaluation,
    ) -> dict[str, float | int | bool]:
        generator = self.generator(optimal.policy)
        kernel = sparse.eye(self.number_of_states, format="csr") + (
            generator / self.p.uniformization_rate
        )
        row_error = float(
            np.max(np.abs(np.asarray(generator.sum(axis=1)).ravel()))
        )
        kernel_row_error = float(
            np.max(np.abs(np.asarray(kernel.sum(axis=1)).ravel() - 1.0))
        )
        minimum_kernel_probability = float(kernel.data.min())
        minimum_off_diagonal = math.inf
        coo = generator.tocoo()
        for row, column, value in zip(coo.row, coo.col, coo.data):
            if row != column:
                minimum_off_diagonal = min(minimum_off_diagonal, float(value))
        if math.isinf(minimum_off_diagonal):
            minimum_off_diagonal = 0.0
        return {
            "state_count_matches": self.number_of_states
            == math.prod((self.p.B[i] + 1) * (self.p.K[i] + 1) for i in range(3)),
            "generator_row_sum_error": row_error,
            "minimum_off_diagonal_rate": minimum_off_diagonal,
            "uniformized_row_sum_error": kernel_row_error,
            "minimum_uniformized_probability": minimum_kernel_probability,
            "bellman_residual": optimal.bellman_residual,
            "rvi_evaluation_gain_error": abs(
                optimal.gain - optimal_evaluation.gain
            ),
            "stationary_generator_residual": optimal_evaluation.generator_residual,
            "poisson_residual": optimal_evaluation.poisson_residual or 0.0,
            "poisson_gain_error": optimal_evaluation.poisson_gain_error or 0.0,
            "arrival_promotion_flow_error": (
                optimal_evaluation.flow_residual_arrival_promotion
            ),
            "promotion_departure_flow_error": (
                optimal_evaluation.flow_residual_promotion_departure
            ),
            "cost_decomposition_error": optimal_evaluation.cost_decomposition_error,
        }

    def simulate(
        self,
        policy: Policy,
        policy_name: str,
        *,
        horizon: float,
        burn_in: float,
        seed: int,
    ) -> SimulationEstimate:
        """Independent continuous-time, time-weighted validation simulation."""

        if horizon <= burn_in or burn_in < 0.0:
            raise ValueError("Require horizon > burn_in >= 0.")
        self.validate_policy(policy)
        rng = np.random.default_rng(seed)
        state: State = (0, 0, 0, 0, 0, 0)
        current_time = 0.0
        area_q = 0.0
        area_n = 0.0
        arrivals = 0
        blocked = 0
        events = 0

        while current_time < horizon:
            internal = self.internal_events(state)
            total_rate = self.p.arrival_rate + sum(rate for rate, _ in internal)
            holding_time = float(rng.exponential(1.0 / total_rate))
            next_time = min(current_time + holding_time, horizon)
            observed_start = max(current_time, burn_in)
            observed_end = max(min(next_time, horizon), burn_in)
            observed_duration = max(0.0, observed_end - observed_start)
            if observed_duration > 0.0:
                area_q += sum(state[0::2]) * observed_duration
                area_n += sum(state[1::2]) * observed_duration

            if current_time + holding_time > horizon:
                break
            event_time = current_time + holding_time
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
                selected = internal[-1][1]
                for rate, successor in internal:
                    cumulative += rate
                    if draw < cumulative:
                        selected = successor
                        break
                state = selected
            current_time = event_time
            events += 1

        observation_time = horizon - burn_in
        mean_q = area_q / observation_time
        mean_n = area_n / observation_time
        return SimulationEstimate(
            policy_name=policy_name,
            observation_time=observation_time,
            events=events,
            arrivals=arrivals,
            blocked=blocked,
            blocking_probability=blocked / arrivals,
            mean_vm_occupancy=mean_q,
            mean_container_occupancy=mean_n,
            mean_total_occupancy=mean_q + mean_n,
            average_cost=(
                self.p.holding_vm * area_q
                + self.p.holding_container * area_n
                + self.p.blocking_cost * blocked
            ) / observation_time,
        )


def flatten_evaluation(
    prefix: str,
    evaluation: PolicyEvaluation,
    optimal_gain: float,
) -> dict[str, object]:
    values = evaluation.summary_dict()
    values["relative_gain_gap"] = relative_gain_gap(evaluation.gain, optimal_gain)
    return {f"{prefix}_{key}": value for key, value in values.items()}


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot take a percentile of an empty collection.")
    position = int(math.floor(probability * (len(ordered) - 1)))
    return ordered[position]
