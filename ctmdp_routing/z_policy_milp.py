"""Exact deterministic observation-constrained policy optimization.

This module solves the deterministic ``D-Z`` problem for the two-VM CTMDP.
The controller observes only the ordered backlog-and-feasibility vector

``Z(s) = ((q_1 + n_1, 1{q_1 < B}), (q_2 + n_2, 1{q_2 < B}))``

and must choose the same routing action in every state with the same
observation.  The optimization is an exact mixed-integer linear program over
stationary state-action occupation measures.  It is deliberately separate
from the frozen paper runners so that adding the reviewer-requested benchmark
does not mutate the committed baseline results.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import asdict, dataclass
from typing import TypeAlias

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from .n2_solver import N2AverageCostCTMDP, Policy


Observation: TypeAlias = tuple[tuple[int, int], tuple[int, int]]


def backlog_feasibility_observation(
    model: N2AverageCostCTMDP,
    state: tuple[int, int, int, int],
) -> Observation:
    """Return the ordered observation ``Z(s)`` used by JSQ.

    VM labels remain part of the observation.  The map removes only the
    feeder/container composition, not the identity of a destination.
    """

    return tuple(
        (
            sum(model.local_state(state, vm)),
            int(model.local_state(state, vm)[0] < model.p.B),
        )
        for vm in (1, 2)
    )  # type: ignore[return-value]


@dataclass
class DeterministicZPolicyResult:
    """Solution and independent diagnostics for the D-Z MILP."""

    gain: float
    milp_objective: float
    independently_evaluated_gain: float
    gain_evaluation_error: float
    certified_absolute_gap: float
    certified_relative_gap: float
    certified_to_requested_gap: bool
    requested_certification_relative_gap: float
    requested_mip_relative_gap: float
    requested_mip_absolute_gap: float
    mip_feasibility_tolerance: float
    lp_feasibility_tolerance: float
    policy: Policy
    stationary: np.ndarray
    occupation: np.ndarray
    observation_actions: dict[Observation, int]
    solver_status: int
    solver_message: str
    solver_success: bool
    solve_seconds: float
    mip_gap: float | None
    mip_dual_bound: float | None
    mip_node_count: int | None
    number_of_states: int
    number_of_observations: int
    number_of_ambiguous_observations: int
    number_of_occupation_variables: int
    number_of_binary_variables: int
    balance_residual: float
    milp_occupation_balance_residual: float
    normalization_error: float
    integrality_error: float
    independent_generator_residual: float

    def summary_dict(self) -> dict[str, object]:
        """Return JSON-serializable scalar diagnostics."""

        result = asdict(self)
        for key in (
            "policy",
            "stationary",
            "occupation",
            "observation_actions",
        ):
            result.pop(key)
        return result


def _cost_rate(
    model: N2AverageCostCTMDP,
    state_index: int,
    action: int,
) -> float:
    state = model.states[state_index]
    return (
        model.holding_cost_rate(state)
        + model.p.arrival_rate * model.arrival_impulse_cost(action)
    )


def solve_deterministic_z_policy(
    model: N2AverageCostCTMDP,
    *,
    mip_relative_gap: float = 1.0e-9,
    mip_absolute_gap: float = 1.0e-10,
    mip_feasibility_tolerance: float = 1.0e-9,
    lp_feasibility_tolerance: float = 1.0e-10,
    certification_relative_gap: float = 1.0e-8,
    time_limit: float | None = None,
    node_limit: int | None = None,
    require_optimal: bool = True,
    display_solver_output: bool = False,
) -> DeterministicZPolicyResult:
    """Solve the exact deterministic stationary Z-measurable policy problem.

    Parameters
    ----------
    model:
        Existing two-VM CTMDP instance.  Its constant-uniformized transition
        rows and continuous-time cost rates are used without approximation.
    mip_relative_gap:
        Requested relative MIP optimality gap passed to HiGHS through SciPy.
    mip_absolute_gap:
        Requested absolute MIP optimality gap.  This must be set explicitly:
        HiGHS otherwise permits its looser default absolute gap to terminate a
        run even when ``mip_relative_gap`` is much smaller.
    mip_feasibility_tolerance:
        HiGHS feasibility/integrality tolerance.  SciPy forwards this native
        HiGHS option even though it is not part of SciPy's short public option
        list.  Tightening it is important because small violations of the
        binary linking constraints can otherwise improve the apparent MILP
        objective without defining a feasible deterministic policy.
    lp_feasibility_tolerance:
        HiGHS primal and dual LP feasibility tolerances.  Tight balance
        tolerances matter because the occupation equations contain one row
        per CTMDP state; small row violations can otherwise make the raw MILP
        objective visibly lower than the independently evaluated policy.
    certification_relative_gap:
        Maximum permitted relative gap between the independently evaluated
        recovered policy and the solver's global dual bound.
    time_limit, node_limit:
        Optional solver limits.  By default no explicit limit is imposed.
    require_optimal:
        If true, reject a time-limited feasible incumbent.  This is the safe
        setting for publication-facing results that are called exact.
    display_solver_output:
        Forward the solver log to standard output.

    Notes
    -----
    ``x[s,a]`` is the stationary occupation measure and ``y[z,a]`` is the
    binary action selected for observation ``z``.  The model is

    * stationary balance for the uniformized kernel;
    * total occupation equal to one;
    * one selected action per observation with multiple feasible actions;
    * ``sum_{s: Z(s)=z} x[s,a] <= y[z,a]``.

    Observations with only one feasible action need no binary variable.  The
    aggregate linking inequality is equivalent to the usual statewise links
    for integral ``y`` and is substantially tighter under finite numerical
    feasibility tolerances.

    Since every stationary policy in this CTMDP has a unique recurrent class,
    the feasible occupation measure is the unique invariant distribution of
    the recovered deterministic policy.
    """

    if mip_relative_gap < 0.0:
        raise ValueError("mip_relative_gap must be nonnegative.")
    if mip_absolute_gap < 0.0:
        raise ValueError("mip_absolute_gap must be nonnegative.")
    if not 0.0 < mip_feasibility_tolerance <= 1.0e-4:
        raise ValueError(
            "mip_feasibility_tolerance must lie in (0, 1e-4]."
        )
    if not 1.0e-10 <= lp_feasibility_tolerance <= 1.0e-4:
        raise ValueError(
            "lp_feasibility_tolerance must lie in [1e-10, 1e-4]."
        )
    if certification_relative_gap < 0.0:
        raise ValueError("certification_relative_gap must be nonnegative.")
    if time_limit is not None and time_limit <= 0.0:
        raise ValueError("time_limit must be positive when specified.")
    if node_limit is not None and node_limit < 1:
        raise ValueError("node_limit must be positive when specified.")

    state_actions: tuple[tuple[int, int], ...] = tuple(
        (state_index, action)
        for state_index, state in enumerate(model.states)
        for action in model.feasible_actions(state)
    )
    x_index = {
        pair: variable_index for variable_index, pair in enumerate(state_actions)
    }
    number_of_x = len(state_actions)

    observation_actions: dict[Observation, tuple[int, ...]] = {}
    state_observations: list[Observation] = []
    for state in model.states:
        observation = backlog_feasibility_observation(model, state)
        actions = model.feasible_actions(state)
        previous = observation_actions.setdefault(observation, actions)
        if previous != actions:
            raise AssertionError(
                "States with the same Z observation must have the same "
                f"feasible actions: z={observation}, {previous} != {actions}."
            )
        state_observations.append(observation)

    ordered_observations = tuple(sorted(observation_actions))
    observation_state_indices: dict[Observation, list[int]] = {
        observation: [] for observation in ordered_observations
    }
    for state_index, observation in enumerate(state_observations):
        observation_state_indices[observation].append(state_index)

    decision_observations = tuple(
        observation
        for observation in ordered_observations
        if len(observation_actions[observation]) > 1
    )
    indicator_pairs: tuple[tuple[Observation, int], ...] = tuple(
        (observation, action)
        for observation in decision_observations
        for action in observation_actions[observation]
    )
    y_index = {
        pair: number_of_x + offset
        for offset, pair in enumerate(indicator_pairs)
    }
    number_of_y = len(indicator_pairs)
    number_of_variables = number_of_x + number_of_y

    objective = np.zeros(number_of_variables, dtype=float)
    for variable, (state_index, action) in enumerate(state_actions):
        objective[variable] = _cost_rate(model, state_index, action)

    integrality = np.zeros(number_of_variables, dtype=np.uint8)
    integrality[number_of_x:] = 1
    lower_bounds = np.zeros(number_of_variables, dtype=float)
    upper_bounds = np.ones(number_of_variables, dtype=float)

    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []
    current_row = 0

    # Stationary balance: sum_a x[s,a] - sum_{r,a} x[r,a] P^a(r,s) = 0.
    for _ in model.states:
        constraint_lower.append(0.0)
        constraint_upper.append(0.0)
        current_row += 1
    for variable, (state_index, action) in enumerate(state_actions):
        row_indices.append(state_index)
        column_indices.append(variable)
        coefficients.append(1.0)

        transition_row = model.uniformized_row(state_index, action)
        for successor_index in np.flatnonzero(transition_row):
            row_indices.append(int(successor_index))
            column_indices.append(variable)
            coefficients.append(-float(transition_row[successor_index]))

    # Normalization: sum_{s,a} x[s,a] = 1.
    normalization_row = current_row
    for variable in range(number_of_x):
        row_indices.append(normalization_row)
        column_indices.append(variable)
        coefficients.append(1.0)
    constraint_lower.append(1.0)
    constraint_upper.append(1.0)
    current_row += 1

    # One deterministic action is selected for every nontrivial observation.
    # Forced observations have no binary variables.
    for observation in decision_observations:
        for action in observation_actions[observation]:
            row_indices.append(current_row)
            column_indices.append(y_index[(observation, action)])
            coefficients.append(1.0)
        constraint_lower.append(1.0)
        constraint_upper.append(1.0)
        current_row += 1

    # Observation consistency.  Aggregating all occupation assigned to an
    # observation/action pair is stronger numerically than one link per state:
    # small feasibility violations cannot accumulate across the fibre Z^{-1}(z).
    for observation in decision_observations:
        for action in observation_actions[observation]:
            for state_index in observation_state_indices[observation]:
                row_indices.append(current_row)
                column_indices.append(x_index[(state_index, action)])
                coefficients.append(1.0)
            row_indices.append(current_row)
            column_indices.append(y_index[(observation, action)])
            coefficients.append(-1.0)
            constraint_lower.append(-math.inf)
            constraint_upper.append(0.0)
            current_row += 1

    constraint_matrix = sparse.coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(current_row, number_of_variables),
        dtype=float,
    ).tocsr()
    linear_constraint = LinearConstraint(
        constraint_matrix,
        np.asarray(constraint_lower, dtype=float),
        np.asarray(constraint_upper, dtype=float),
    )

    options: dict[str, bool | float | int] = {
        "disp": display_solver_output,
        "presolve": True,
        "mip_rel_gap": mip_relative_gap,
        "mip_abs_gap": mip_absolute_gap,
        "mip_feasibility_tolerance": mip_feasibility_tolerance,
        "primal_feasibility_tolerance": lp_feasibility_tolerance,
        "dual_feasibility_tolerance": lp_feasibility_tolerance,
    }
    if time_limit is not None:
        options["time_limit"] = time_limit
    if node_limit is not None:
        options["node_limit"] = node_limit

    started = time.perf_counter()
    with warnings.catch_warnings():
        # SciPy warns that native HiGHS options are not in its abbreviated
        # option list, then deliberately forwards them to HiGHS.  Suppress
        # only that expected forwarding warning.
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected:.*",
            category=RuntimeWarning,
        )
        solver_result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=linear_constraint,
            options=options,
        )
    solve_seconds = time.perf_counter() - started

    if solver_result.x is None:
        raise RuntimeError(
            "The D-Z MILP returned no feasible policy: "
            f"status={solver_result.status}, message={solver_result.message}"
        )
    if require_optimal and solver_result.status != 0:
        raise RuntimeError(
            "The D-Z MILP was not certified optimal: "
            f"status={solver_result.status}, message={solver_result.message}, "
            f"mip_gap={getattr(solver_result, 'mip_gap', None)}"
        )

    solution = np.asarray(solver_result.x, dtype=float)
    selected_actions: dict[Observation, int] = {}
    for observation in ordered_observations:
        actions = observation_actions[observation]
        if len(actions) == 1:
            selected_actions[observation] = actions[0]
            continue
        values = np.asarray(
            [solution[y_index[(observation, action)]] for action in actions]
        )
        selected_position = int(np.argmax(values))
        if values[selected_position] < 0.5:
            raise AssertionError(
                f"No integral action recovered for observation {observation}: "
                f"values={values}."
            )
        selected_actions[observation] = actions[selected_position]

    policy = np.asarray(
        [selected_actions[observation] for observation in state_observations],
        dtype=int,
    )
    model.validate_policy(policy)

    milp_occupation = np.zeros((model.number_of_states, 3), dtype=float)
    for variable, (state_index, action) in enumerate(state_actions):
        milp_occupation[state_index, action] = max(solution[variable], 0.0)
    milp_stationary = milp_occupation.sum(axis=1)
    milp_stationary /= milp_stationary.sum()

    milp_propagated = np.zeros(model.number_of_states, dtype=float)
    for (state_index, action), variable in x_index.items():
        milp_propagated += milp_occupation[state_index, action] * (
            model.uniformized_row(state_index, action)
        )
    milp_balance_residual = float(
        np.max(np.abs(milp_stationary - milp_propagated))
    )

    indicator_values = solution[number_of_x:]
    integrality_error = (
        0.0
        if indicator_values.size == 0
        else float(
            np.max(
                np.minimum(
                    np.abs(indicator_values),
                    np.abs(1.0 - indicator_values),
                )
            )
        )
    )
    independent = model.evaluate_policy(policy, "best_deterministic_z_policy")
    stationary = independent.stationary.copy()
    occupation = np.zeros((model.number_of_states, 3), dtype=float)
    occupation[np.arange(model.number_of_states), policy] = stationary
    propagated = np.zeros(model.number_of_states, dtype=float)
    for state_index, action in enumerate(policy):
        propagated += stationary[state_index] * model.uniformized_row(
            state_index, int(action)
        )
    balance_residual = float(np.max(np.abs(stationary - propagated)))

    milp_objective = float(solver_result.fun)
    dual_bound_value = getattr(solver_result, "mip_dual_bound", None)
    dual_bound = (
        None if dual_bound_value is None else float(dual_bound_value)
    )
    if dual_bound is None:
        certified_absolute_gap = math.inf
        certified_relative_gap = math.inf
    else:
        certified_absolute_gap = max(0.0, independent.gain - dual_bound)
        certified_relative_gap = certified_absolute_gap / max(
            abs(independent.gain), 1.0e-15
        )
    maximum_integrality_error = max(
        10.0 * mip_feasibility_tolerance, 1.0e-8
    )
    certified_to_requested_gap = bool(
        solver_result.status == 0
        and integrality_error <= maximum_integrality_error
        and certified_relative_gap <= certification_relative_gap
    )
    if require_optimal:
        if dual_bound is None:
            raise RuntimeError(
                "The D-Z MILP returned no global dual bound, so the recovered "
                "policy cannot be certified optimal."
            )
        if integrality_error > maximum_integrality_error:
            raise RuntimeError(
                "The D-Z MILP reported optimality but its binary solution is "
                f"not integral to the required tolerance: {integrality_error:.3e} "
                f"> {maximum_integrality_error:.3e}."
            )
        if certified_relative_gap > certification_relative_gap:
            raise RuntimeError(
                "The rounded D-Z policy is not certified to the requested "
                f"relative gap: {certified_relative_gap:.3e} > "
                f"{certification_relative_gap:.3e}.  Tighten solver feasibility "
                "or optimality tolerances before reporting this result. "
                f"Independent gain={independent.gain:.15g}, "
                f"MILP objective={milp_objective:.15g}, "
                f"dual bound={dual_bound:.15g}, "
                f"raw balance residual={milp_balance_residual:.3e}."
            )

    return DeterministicZPolicyResult(
        gain=independent.gain,
        milp_objective=milp_objective,
        independently_evaluated_gain=independent.gain,
        gain_evaluation_error=abs(milp_objective - independent.gain),
        certified_absolute_gap=certified_absolute_gap,
        certified_relative_gap=certified_relative_gap,
        certified_to_requested_gap=certified_to_requested_gap,
        requested_certification_relative_gap=certification_relative_gap,
        requested_mip_relative_gap=mip_relative_gap,
        requested_mip_absolute_gap=mip_absolute_gap,
        mip_feasibility_tolerance=mip_feasibility_tolerance,
        lp_feasibility_tolerance=lp_feasibility_tolerance,
        policy=policy,
        stationary=stationary,
        occupation=occupation,
        observation_actions=selected_actions,
        solver_status=int(solver_result.status),
        solver_message=str(solver_result.message),
        solver_success=bool(solver_result.success),
        solve_seconds=solve_seconds,
        mip_gap=(
            None
            if getattr(solver_result, "mip_gap", None) is None
            else float(solver_result.mip_gap)
        ),
        mip_dual_bound=dual_bound,
        mip_node_count=(
            None
            if getattr(solver_result, "mip_node_count", None) is None
            else int(solver_result.mip_node_count)
        ),
        number_of_states=model.number_of_states,
        number_of_observations=len(ordered_observations),
        number_of_ambiguous_observations=sum(
            len(observation_actions[observation]) > 1
            for observation in ordered_observations
        ),
        number_of_occupation_variables=number_of_x,
        number_of_binary_variables=number_of_y,
        balance_residual=balance_residual,
        milp_occupation_balance_residual=milp_balance_residual,
        normalization_error=abs(float(stationary.sum()) - 1.0),
        integrality_error=integrality_error,
        independent_generator_residual=independent.generator_residual,
    )
