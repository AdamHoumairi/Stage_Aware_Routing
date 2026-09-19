"""Numerical optimization of stationary randomized Z-measurable policies.

For N=2, every observation at which both VMs are feasible has one scalar
parameter ``theta[z] = P(route to VM 1 | z)``.  The VM-2 probability is
``1 - theta[z]``.  Forced routing and blocking observations have no parameter.

The invariant distribution depends nonlinearly on ``theta``.  This module
therefore optimizes the exact average-cost CTMC objective directly and uses an
analytic gradient obtained from the continuous-time Poisson equation.  The
problem is nonconvex; multistart convergence is reported, but global
optimality is not claimed without an external global certificate.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse import linalg as sparse_linalg

from .n2_solver import ActionProbabilities, N2AverageCostCTMDP
from .z_policy_milp import Observation, backlog_feasibility_observation


@dataclass(frozen=True)
class RandomizedZOptimizationRun:
    """One local bounded-optimization run."""

    start_name: str
    initial_gain: float
    gain: float
    success: bool
    status: int
    message: str
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    projected_gradient_residual: float
    solve_seconds: float


@dataclass
class RandomizedZPolicyResult:
    """Best multistart R-Z candidate and numerical diagnostics."""

    gain: float
    fast_objective: float
    independently_evaluated_gain: float
    gain_evaluation_error: float
    action_probabilities: ActionProbabilities
    stationary: np.ndarray
    parameter_vector: np.ndarray
    observation_action_probabilities: dict[Observation, tuple[float, float, float]]
    runs: tuple[RandomizedZOptimizationRun, ...]
    best_start_name: str
    number_of_states: int
    number_of_observations: int
    number_of_randomized_decision_observations: int
    start_count: int
    successful_start_count: int
    best_gain_start_count: int
    distinct_final_gain_count: int
    final_gain_range: float
    number_of_strictly_randomized_observations: int
    projected_gradient_residual: float
    fast_generator_residual: float
    poisson_residual: float
    independent_generator_residual: float
    independent_poisson_gain_error: float
    independent_cost_decomposition_error: float
    global_lower_bound: float | None
    global_bound_relative_gap: float | None
    globally_certified: bool
    optimality_note: str

    def summary_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key in (
            "action_probabilities",
            "stationary",
            "parameter_vector",
            "observation_action_probabilities",
            "runs",
        ):
            result.pop(key)
        return result

    def run_dicts(self) -> list[dict[str, object]]:
        return [asdict(run) for run in self.runs]


@dataclass(frozen=True)
class _DecisionFibre:
    observation: Observation
    state_indices: np.ndarray
    vm1_successors: np.ndarray
    vm2_successors: np.ndarray


class N2RandomizedZPolicyProblem:
    """Exact objective and gradient for the N=2 stationary R-Z problem."""

    def __init__(self, model: N2AverageCostCTMDP):
        self.model = model
        observation_actions: dict[Observation, tuple[int, ...]] = {}
        observation_states: dict[Observation, list[int]] = {}
        self.state_observations: list[Observation] = []
        for state_index, state in enumerate(model.states):
            observation = backlog_feasibility_observation(model, state)
            actions = model.feasible_actions(state)
            previous = observation_actions.setdefault(observation, actions)
            if previous != actions:
                raise AssertionError(
                    "Action feasibility must be determined by Z: "
                    f"{observation}, {previous} != {actions}."
                )
            observation_states.setdefault(observation, []).append(state_index)
            self.state_observations.append(observation)

        self.observation_actions = observation_actions
        self.ordered_observations = tuple(sorted(observation_actions))
        self.decision_observations = tuple(
            observation
            for observation in self.ordered_observations
            if len(observation_actions[observation]) > 1
        )
        self.variable_index = {
            observation: index
            for index, observation in enumerate(self.decision_observations)
        }

        fibres: list[_DecisionFibre] = []
        for observation in self.decision_observations:
            if observation_actions[observation] != (1, 2):
                raise AssertionError(
                    "The N=2 randomized parameterization expects actions (1, 2)."
                )
            state_indices = np.asarray(
                observation_states[observation], dtype=np.int64
            )
            vm1_successors = np.asarray(
                [
                    model.index[model.post_arrival(model.states[index], 1)]
                    for index in state_indices
                ],
                dtype=np.int64,
            )
            vm2_successors = np.asarray(
                [
                    model.index[model.post_arrival(model.states[index], 2)]
                    for index in state_indices
                ],
                dtype=np.int64,
            )
            fibres.append(
                _DecisionFibre(
                    observation=observation,
                    state_indices=state_indices,
                    vm1_successors=vm1_successors,
                    vm2_successors=vm2_successors,
                )
            )
        self.fibres = tuple(fibres)

        self.base_action_probabilities = np.zeros(
            (model.number_of_states, 3), dtype=float
        )
        for state_index, state in enumerate(model.states):
            actions = model.feasible_actions(state)
            if actions == (1, 2):
                self.base_action_probabilities[state_index, 2] = 1.0
            else:
                self.base_action_probabilities[state_index, actions[0]] = 1.0
        model.validate_action_probabilities(self.base_action_probabilities)
        self.base_generator = sparse.csr_matrix(
            model.generator_from_action_probabilities(
                self.base_action_probabilities
            )
        )
        blocking = self.base_action_probabilities[:, 0]
        self.cost_rate = np.asarray(
            [
                model.holding_cost_rate(state)
                + model.p.arrival_rate * model.p.blocking_cost * blocking[index]
                for index, state in enumerate(model.states)
            ],
            dtype=float,
        )

        derivative_rows: list[int] = []
        derivative_columns: list[int] = []
        derivative_variables: list[int] = []
        derivative_signs: list[float] = []
        for variable, fibre in enumerate(self.fibres):
            for state_index, vm1_successor, vm2_successor in zip(
                fibre.state_indices,
                fibre.vm1_successors,
                fibre.vm2_successors,
            ):
                derivative_rows.extend((int(state_index), int(state_index)))
                derivative_columns.extend(
                    (int(vm1_successor), int(vm2_successor))
                )
                derivative_variables.extend((variable, variable))
                derivative_signs.extend(
                    (model.p.arrival_rate, -model.p.arrival_rate)
                )
        self._derivative_rows = np.asarray(derivative_rows, dtype=np.int64)
        self._derivative_columns = np.asarray(
            derivative_columns, dtype=np.int64
        )
        self._derivative_variables = np.asarray(
            derivative_variables, dtype=np.int64
        )
        self._derivative_signs = np.asarray(derivative_signs, dtype=float)

        self._cached_parameters: np.ndarray | None = None
        self._cached_gain: float | None = None
        self._cached_gradient: np.ndarray | None = None
        self._cached_stationary: np.ndarray | None = None
        self._cached_generator_residual: float | None = None
        self._cached_poisson_residual: float | None = None

    @property
    def number_of_variables(self) -> int:
        return len(self.decision_observations)

    def validate_parameter_vector(self, parameters: np.ndarray) -> np.ndarray:
        vector = np.asarray(parameters, dtype=float)
        if vector.shape != (self.number_of_variables,):
            raise ValueError(
                f"R-Z parameter vector must have shape "
                f"({self.number_of_variables},), not {vector.shape}."
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("R-Z parameters must be finite.")
        if np.min(vector) < -1.0e-12 or np.max(vector) > 1.0 + 1.0e-12:
            raise ValueError("R-Z parameters must lie in [0, 1].")
        return np.clip(vector, 0.0, 1.0)

    def action_probabilities(
        self, parameters: np.ndarray
    ) -> ActionProbabilities:
        vector = self.validate_parameter_vector(parameters)
        probabilities = self.base_action_probabilities.copy()
        for variable, fibre in enumerate(self.fibres):
            probability_vm1 = vector[variable]
            probabilities[fibre.state_indices, 1] = probability_vm1
            probabilities[fibre.state_indices, 2] = 1.0 - probability_vm1
        self.model.validate_action_probabilities(probabilities)
        return probabilities

    def vector_from_action_probabilities(
        self,
        probabilities: ActionProbabilities,
        *,
        measurability_tolerance: float = 1.0e-10,
    ) -> np.ndarray:
        self.model.validate_action_probabilities(probabilities)
        vector = np.empty(self.number_of_variables, dtype=float)
        for variable, fibre in enumerate(self.fibres):
            values = probabilities[fibre.state_indices, 1]
            if float(values.max() - values.min()) > measurability_tolerance:
                raise ValueError(
                    "Initial policy is not Z-measurable at observation "
                    f"{fibre.observation}."
                )
            vector[variable] = float(values.mean())
        return self.validate_parameter_vector(vector)

    def generator(self, parameters: np.ndarray) -> sparse.csr_matrix:
        vector = self.validate_parameter_vector(parameters)
        if self._derivative_rows.size == 0:
            return self.base_generator.copy()
        values = (
            self._derivative_signs * vector[self._derivative_variables]
        )
        delta = sparse.coo_matrix(
            (values, (self._derivative_rows, self._derivative_columns)),
            shape=self.base_generator.shape,
            dtype=float,
        ).tocsr()
        return self.base_generator + delta

    def _stationary_distribution(
        self, generator: sparse.csr_matrix
    ) -> tuple[np.ndarray, float]:
        state_count = self.model.number_of_states
        system = generator.T.tolil(copy=True)
        system[-1, :] = np.ones(state_count, dtype=float)
        rhs = np.zeros(state_count, dtype=float)
        rhs[-1] = 1.0
        stationary = np.asarray(
            sparse_linalg.spsolve(system.tocsc(), rhs), dtype=float
        )
        if float(stationary.min()) < -1.0e-8:
            raise AssertionError(
                "R-Z stationary solve returned a material negative value: "
                f"{stationary.min():.3e}."
            )
        stationary = np.maximum(stationary, 0.0)
        stationary /= stationary.sum()
        residual = float(
            np.max(np.abs(np.asarray(stationary @ generator).ravel()))
        )
        return stationary, residual

    def _poisson_bias(
        self, generator: sparse.csr_matrix, gain: float
    ) -> tuple[np.ndarray, float]:
        system = generator.tolil(copy=True)
        reference = self.model.reference_index
        system[reference, :] = 0.0
        system[reference, reference] = 1.0
        rhs = gain - self.cost_rate
        rhs = rhs.copy()
        rhs[reference] = 0.0
        bias = np.asarray(
            sparse_linalg.spsolve(system.tocsc(), rhs), dtype=float
        )
        residual = float(
            np.max(
                np.abs(
                    self.cost_rate
                    + np.asarray(generator @ bias).ravel()
                    - gain
                )
            )
        )
        return bias, residual

    def _evaluate(
        self, parameters: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, float, float]:
        vector = self.validate_parameter_vector(parameters)
        if (
            self._cached_parameters is not None
            and np.array_equal(vector, self._cached_parameters)
        ):
            assert self._cached_gain is not None
            assert self._cached_gradient is not None
            assert self._cached_stationary is not None
            assert self._cached_generator_residual is not None
            assert self._cached_poisson_residual is not None
            return (
                self._cached_gain,
                self._cached_gradient.copy(),
                self._cached_stationary.copy(),
                self._cached_generator_residual,
                self._cached_poisson_residual,
            )

        generator = self.generator(vector)
        stationary, generator_residual = self._stationary_distribution(generator)
        gain = float(stationary @ self.cost_rate)
        bias, poisson_residual = self._poisson_bias(generator, gain)
        gradient = np.empty(self.number_of_variables, dtype=float)
        arrival_rate = self.model.p.arrival_rate
        for variable, fibre in enumerate(self.fibres):
            gradient[variable] = arrival_rate * float(
                np.sum(
                    stationary[fibre.state_indices]
                    * (
                        bias[fibre.vm1_successors]
                        - bias[fibre.vm2_successors]
                    )
                )
            )

        self._cached_parameters = vector.copy()
        self._cached_gain = gain
        self._cached_gradient = gradient.copy()
        self._cached_stationary = stationary.copy()
        self._cached_generator_residual = generator_residual
        self._cached_poisson_residual = poisson_residual
        return gain, gradient, stationary, generator_residual, poisson_residual

    def objective_and_gradient(
        self, parameters: np.ndarray
    ) -> tuple[float, np.ndarray]:
        gain, gradient, _, _, _ = self._evaluate(parameters)
        return gain, gradient

    def objective(self, parameters: np.ndarray) -> float:
        return self._evaluate(parameters)[0]

    def projected_gradient_residual(self, parameters: np.ndarray) -> float:
        vector = self.validate_parameter_vector(parameters)
        _, gradient = self.objective_and_gradient(vector)
        projected = vector - np.clip(vector - gradient, 0.0, 1.0)
        return float(np.max(np.abs(projected)))

    def observation_probabilities(
        self, parameters: np.ndarray
    ) -> dict[Observation, tuple[float, float, float]]:
        probabilities = self.action_probabilities(parameters)
        result: dict[Observation, tuple[float, float, float]] = {}
        for observation in self.ordered_observations:
            state_index = self.state_observations.index(observation)
            row = probabilities[state_index]
            result[observation] = (
                float(row[0]),
                float(row[1]),
                float(row[2]),
            )
        return result


def _deduplicate_starts(
    starts: list[tuple[str, np.ndarray]],
) -> list[tuple[str, np.ndarray]]:
    unique: list[tuple[str, np.ndarray]] = []
    signatures: set[bytes] = set()
    for name, vector in starts:
        signature = np.round(vector, 13).tobytes()
        if signature not in signatures:
            signatures.add(signature)
            unique.append((name, vector))
    return unique


def solve_randomized_z_policy(
    model: N2AverageCostCTMDP,
    *,
    initial_action_probabilities: Mapping[str, ActionProbabilities] | None = None,
    random_starts: int = 12,
    seed: int = 20260917,
    maximum_iterations: int = 500,
    function_tolerance: float = 1.0e-14,
    gradient_tolerance: float = 1.0e-10,
    global_lower_bound: float | None = None,
    global_certification_relative_gap: float = 1.0e-8,
) -> RandomizedZPolicyResult:
    """Numerically optimize the stationary randomized Z-policy class.

    The returned policy is the best feasible candidate found across all local
    starts.  Since the memoryless observation-constrained problem is nonconvex,
    ``globally_certified`` is true only when the candidate reaches a supplied
    valid global lower bound within the requested relative tolerance.
    """

    if random_starts < 0:
        raise ValueError("random_starts must be nonnegative.")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive.")
    if function_tolerance <= 0.0 or gradient_tolerance <= 0.0:
        raise ValueError("Optimization tolerances must be positive.")
    if global_certification_relative_gap < 0.0:
        raise ValueError("global_certification_relative_gap must be nonnegative.")
    if global_lower_bound is not None and not np.isfinite(global_lower_bound):
        raise ValueError("The global lower bound must be finite.")

    problem = N2RandomizedZPolicyProblem(model)
    starts: list[tuple[str, np.ndarray]] = [
        (
            "uniform_tie_jsq",
            problem.vector_from_action_probabilities(
                model.randomized_tie_jsq_action_probabilities()
            ),
        ),
        (
            "lowest_index_jsq",
            problem.vector_from_action_probabilities(
                model.deterministic_action_probabilities(model.jsq_policy())
            ),
        ),
        ("neutral_half", np.full(problem.number_of_variables, 0.5)),
        ("always_vm1_when_both_feasible", np.ones(problem.number_of_variables)),
        ("always_vm2_when_both_feasible", np.zeros(problem.number_of_variables)),
    ]
    if initial_action_probabilities:
        for name, probabilities in initial_action_probabilities.items():
            starts.append(
                (name, problem.vector_from_action_probabilities(probabilities))
            )

    rng = np.random.default_rng(seed)
    for number in range(random_starts):
        starts.append(
            (
                f"random_{number + 1:02d}",
                rng.uniform(0.0, 1.0, size=problem.number_of_variables),
            )
        )
    starts = _deduplicate_starts(starts)

    run_records: list[RandomizedZOptimizationRun] = []
    final_vectors: list[np.ndarray] = []
    for start_name, initial_vector in starts:
        initial_gain = problem.objective(initial_vector)
        started = time.perf_counter()
        optimization = minimize(
            problem.objective_and_gradient,
            initial_vector,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, 1.0)] * problem.number_of_variables,
            options={
                "maxiter": maximum_iterations,
                "ftol": function_tolerance,
                "gtol": gradient_tolerance,
                "maxls": 50,
                "maxcor": 20,
            },
        )
        solve_seconds = time.perf_counter() - started
        final_vector = np.clip(np.asarray(optimization.x, dtype=float), 0.0, 1.0)
        final_gain = problem.objective(final_vector)
        projected_residual = problem.projected_gradient_residual(final_vector)
        final_vectors.append(final_vector)
        run_records.append(
            RandomizedZOptimizationRun(
                start_name=start_name,
                initial_gain=initial_gain,
                gain=final_gain,
                success=bool(optimization.success),
                status=int(optimization.status),
                message=str(optimization.message),
                iterations=int(optimization.nit),
                function_evaluations=int(optimization.nfev),
                gradient_evaluations=int(optimization.njev),
                projected_gradient_residual=projected_residual,
                solve_seconds=solve_seconds,
            )
        )

    best_position = min(
        range(len(run_records)), key=lambda index: run_records[index].gain
    )
    best_run = run_records[best_position]
    best_vector = final_vectors[best_position]
    fast_gain, _, fast_stationary, fast_generator_residual, poisson_residual = (
        problem._evaluate(best_vector)
    )
    probabilities = problem.action_probabilities(best_vector)
    independent = model.evaluate_action_probabilities(
        probabilities, "optimized_randomized_z_policy"
    )

    rounded_gains = {round(run.gain, 10) for run in run_records}
    final_gains = [run.gain for run in run_records]
    best_gain_start_count = sum(
        abs(run.gain - best_run.gain)
        <= 1.0e-9 * max(1.0, abs(best_run.gain))
        for run in run_records
    )
    strictly_randomized = int(
        np.sum((best_vector > 1.0e-8) & (best_vector < 1.0 - 1.0e-8))
    )
    if global_lower_bound is None:
        global_bound_relative_gap = None
        globally_certified = False
    else:
        lower_bound_tolerance = 1.0e-10 * max(
            1.0, abs(independent.gain), abs(global_lower_bound)
        )
        if global_lower_bound > independent.gain + lower_bound_tolerance:
            raise ValueError(
                "The supplied global lower bound exceeds the feasible R-Z "
                "candidate and is therefore invalid."
            )
        global_bound_relative_gap = max(
            0.0, independent.gain - global_lower_bound
        ) / max(1.0, abs(global_lower_bound))
        globally_certified = (
            global_bound_relative_gap <= global_certification_relative_gap
        )
    if globally_certified:
        optimality_note = (
            "The feasible R-Z policy reaches the supplied global lower bound "
            "within the requested tolerance."
        )
    else:
        optimality_note = (
            "Best feasible stationary randomized Z-policy found by analytic-"
            "gradient multistart L-BFGS-B; global optimality is not certified."
        )
    return RandomizedZPolicyResult(
        gain=independent.gain,
        fast_objective=fast_gain,
        independently_evaluated_gain=independent.gain,
        gain_evaluation_error=abs(independent.gain - fast_gain),
        action_probabilities=probabilities,
        stationary=independent.stationary.copy(),
        parameter_vector=best_vector.copy(),
        observation_action_probabilities=problem.observation_probabilities(
            best_vector
        ),
        runs=tuple(run_records),
        best_start_name=best_run.start_name,
        number_of_states=model.number_of_states,
        number_of_observations=len(problem.ordered_observations),
        number_of_randomized_decision_observations=problem.number_of_variables,
        start_count=len(run_records),
        successful_start_count=sum(run.success for run in run_records),
        best_gain_start_count=best_gain_start_count,
        distinct_final_gain_count=len(rounded_gains),
        final_gain_range=max(final_gains) - min(final_gains),
        number_of_strictly_randomized_observations=strictly_randomized,
        projected_gradient_residual=best_run.projected_gradient_residual,
        fast_generator_residual=fast_generator_residual,
        poisson_residual=poisson_residual,
        independent_generator_residual=independent.generator_residual,
        independent_poisson_gain_error=independent.poisson_gain_error,
        independent_cost_decomposition_error=independent.cost_decomposition_error,
        global_lower_bound=global_lower_bound,
        global_bound_relative_gap=global_bound_relative_gap,
        globally_certified=globally_certified,
        optimality_note=optimality_note,
    )
