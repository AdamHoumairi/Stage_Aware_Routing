"""Small, deterministic regression tests for the published CTMDP models."""

from __future__ import annotations

from itertools import product
import unittest

import numpy as np

from ctmdp_routing import (
    ModelParameters,
    N2AverageCostCTMDP,
    N3AverageCostCTMDP,
    N3ModelParameters,
    N2RandomizedZPolicyProblem,
    backlog_feasibility_observation,
    solve_deterministic_z_policy,
    solve_randomized_z_policy,
)


class N2InvariantTests(unittest.TestCase):
    def test_b1_feasible_destinations_have_empty_feeders(self) -> None:
        model = N2AverageCostCTMDP(
            ModelParameters(K=2, B=1, arrival_rate=1.0, mu_vm=1.0)
        )
        jsq = model.jsq_policy()
        feeder_first = model.stage_aware_lexicographic_policy()
        for index, state in enumerate(model.states):
            for action in model.feasible_actions(state):
                if action != 0:
                    self.assertEqual(model.local_state(state, action)[0], 0)
            self.assertEqual(jsq[index], feeder_first[index])

    def test_uniformized_rows_and_stationary_evaluation_agree(self) -> None:
        model = N2AverageCostCTMDP(
            ModelParameters(K=2, B=2, arrival_rate=1.0, mu_vm=1.0)
        )
        solution = model.relative_value_iteration()
        evaluation = model.evaluate_policy(solution.policy, "optimal")
        checks = model.run_invariant_tests(solution, evaluation)
        self.assertLess(checks["maximum_row_sum_error"], 1.0e-10)
        self.assertLess(checks["rvi_stationary_gain_error"], 1.0e-8)

    def test_uniform_tie_jsq_is_z_measurable_and_swap_equivariant(self) -> None:
        model = N2AverageCostCTMDP(
            ModelParameters(K=2, B=2, arrival_rate=1.0, mu_vm=1.0)
        )
        probabilities = model.randomized_tie_jsq_action_probabilities()
        model.validate_action_probabilities(probabilities)

        action_by_observation: dict[object, np.ndarray] = {}
        for state_index, state in enumerate(model.states):
            feasible = model.feasible_actions(state)
            if feasible == (0,):
                expected = {0: 1.0}
            else:
                totals = {
                    vm: sum(model.local_state(state, vm)) for vm in feasible
                }
                minimum = min(totals.values())
                minimizers = tuple(
                    vm for vm in feasible if totals[vm] == minimum
                )
                expected = {
                    vm: 1.0 / len(minimizers) for vm in minimizers
                }
            for action in (0, 1, 2):
                self.assertAlmostEqual(
                    probabilities[state_index, action],
                    expected.get(action, 0.0),
                )

            observation = backlog_feasibility_observation(model, state)
            previous = action_by_observation.setdefault(
                observation, probabilities[state_index]
            )
            np.testing.assert_allclose(previous, probabilities[state_index])

            swapped_index = model.index[model.swap_vms(state)]
            self.assertAlmostEqual(
                probabilities[state_index, 0], probabilities[swapped_index, 0]
            )
            self.assertAlmostEqual(
                probabilities[state_index, 1], probabilities[swapped_index, 2]
            )
            self.assertAlmostEqual(
                probabilities[state_index, 2], probabilities[swapped_index, 1]
            )

        evaluation = model.evaluate_action_probabilities(
            probabilities, "jsq_uniform_ties"
        )
        self.assertLess(evaluation.generator_residual, 1.0e-10)
        self.assertLess(evaluation.poisson_gain_error, 1.0e-10)
        self.assertLess(evaluation.cost_decomposition_error, 1.0e-10)

    def test_uniform_tie_jsq_is_optimal_in_b1_checkpoint(self) -> None:
        model = N2AverageCostCTMDP(
            ModelParameters(K=2, B=1, arrival_rate=1.0, mu_vm=1.0)
        )
        optimal = model.relative_value_iteration()
        optimal_evaluation = model.evaluate_policy(
            optimal.policy, "full_state_optimal"
        )
        uniform_evaluation = model.evaluate_action_probabilities(
            model.randomized_tie_jsq_action_probabilities(),
            "jsq_uniform_ties",
        )
        rz = solve_randomized_z_policy(
            model,
            random_starts=1,
            seed=19,
            global_lower_bound=optimal_evaluation.gain,
        )
        self.assertLess(
            abs(uniform_evaluation.gain - optimal_evaluation.gain), 1.0e-8
        )
        self.assertLess(abs(rz.gain - optimal_evaluation.gain), 1.0e-8)
        self.assertTrue(rz.globally_certified)

    def test_dz_milp_matches_full_state_optimum_when_b_is_one(self) -> None:
        model = N2AverageCostCTMDP(
            ModelParameters(K=2, B=1, arrival_rate=1.0, mu_vm=1.0)
        )
        full = model.relative_value_iteration()
        full_evaluation = model.evaluate_policy(full.policy, "full_state_optimal")
        dz = solve_deterministic_z_policy(model)

        self.assertLess(abs(dz.gain - full_evaluation.gain), 1.0e-8)
        self.assertLess(dz.gain_evaluation_error, 1.0e-8)
        self.assertLess(dz.balance_residual, 1.0e-8)
        self.assertLess(dz.independent_generator_residual, 1.0e-8)
        self.assertTrue(dz.certified_to_requested_gap)

    def test_dz_milp_matches_exhaustive_noninjective_toy_case(self) -> None:
        model = N2AverageCostCTMDP(
            ModelParameters(K=1, B=2, arrival_rate=0.8, mu_vm=1.0)
        )
        dz = solve_deterministic_z_policy(model)

        observation_actions: dict[object, tuple[int, ...]] = {}
        for state in model.states:
            observation = backlog_feasibility_observation(model, state)
            observation_actions.setdefault(
                observation, model.feasible_actions(state)
            )
        flexible = tuple(
            observation
            for observation, actions in observation_actions.items()
            if len(actions) > 1
        )
        fixed = {
            observation: actions[0]
            for observation, actions in observation_actions.items()
            if len(actions) == 1
        }

        exhaustive_gain = np.inf
        for choices in product((1, 2), repeat=len(flexible)):
            action_by_observation = dict(fixed)
            action_by_observation.update(zip(flexible, choices))
            policy = np.asarray(
                [
                    action_by_observation[
                        backlog_feasibility_observation(model, state)
                    ]
                    for state in model.states
                ],
                dtype=int,
            )
            gain = model.evaluate_policy(policy, "enumerated_z_policy").gain
            exhaustive_gain = min(exhaustive_gain, gain)

        self.assertLess(abs(dz.gain - exhaustive_gain), 1.0e-8)
        self.assertTrue(dz.certified_to_requested_gap)
        for left_index, left_state in enumerate(model.states):
            for right_index, right_state in enumerate(model.states):
                if backlog_feasibility_observation(
                    model, left_state
                ) == backlog_feasibility_observation(model, right_state):
                    self.assertEqual(dz.policy[left_index], dz.policy[right_index])

    def test_rz_analytic_gradient_matches_finite_differences(self) -> None:
        model = N2AverageCostCTMDP(
            ModelParameters(K=1, B=2, arrival_rate=0.8, mu_vm=1.0)
        )
        problem = N2RandomizedZPolicyProblem(model)
        parameters = np.linspace(0.2, 0.8, problem.number_of_variables)
        _, gradient = problem.objective_and_gradient(parameters)
        finite_difference = np.empty_like(gradient)
        step = 1.0e-6
        for index in range(problem.number_of_variables):
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] += step
            minus[index] -= step
            finite_difference[index] = (
                problem.objective(plus) - problem.objective(minus)
            ) / (2.0 * step)
        self.assertLess(
            float(np.max(np.abs(gradient - finite_difference))), 1.0e-7
        )

    def test_rz_multistart_improves_dz_toy_benchmark(self) -> None:
        model = N2AverageCostCTMDP(
            ModelParameters(K=1, B=2, arrival_rate=0.8, mu_vm=1.0)
        )
        dz = solve_deterministic_z_policy(model)
        rz = solve_randomized_z_policy(
            model,
            initial_action_probabilities={
                "deterministic_z_optimum": (
                    model.deterministic_action_probabilities(dz.policy)
                )
            },
            random_starts=4,
            seed=17,
        )
        uniform = model.evaluate_action_probabilities(
            model.randomized_tie_jsq_action_probabilities(),
            "jsq_uniform_ties",
        )
        self.assertLessEqual(rz.gain, dz.gain + 1.0e-9)
        self.assertLessEqual(rz.gain, uniform.gain + 1.0e-9)
        self.assertGreater(dz.gain - rz.gain, 1.0e-6)
        self.assertLess(rz.gain_evaluation_error, 1.0e-10)
        self.assertLess(rz.projected_gradient_residual, 1.0e-7)


class N3WitnessTests(unittest.TestCase):
    def test_n3_uniform_tie_jsq_probabilities_and_b1_optimality(self) -> None:
        model = N3AverageCostCTMDP(
            N3ModelParameters.homogeneous(
                K=1, B=1, arrival_rate=0.6, mu_vm=1.0
            )
        )
        probabilities = model.randomized_tie_jsq_action_probabilities()
        model.validate_action_probabilities(probabilities)
        for state_index, state in enumerate(model.states):
            feasible = model.feasible_actions(state)
            if feasible == (0,):
                expected = {0: 1.0}
            else:
                totals = {
                    vm: sum(model.local_state(state, vm)) for vm in feasible
                }
                minimum = min(totals.values())
                minimizers = tuple(
                    vm for vm in feasible if totals[vm] == minimum
                )
                expected = {
                    vm: 1.0 / len(minimizers) for vm in minimizers
                }
            for action in (0, 1, 2, 3):
                self.assertAlmostEqual(
                    probabilities[state_index, action],
                    expected.get(action, 0.0),
                )

        uniform = model.evaluate_action_probabilities(
            probabilities, "jsq_uniform_ties", poisson_check=True
        )
        optimal = model.relative_value_iteration()
        optimal_evaluation = model.evaluate_policy(
            optimal.policy, "full_state_optimal"
        )
        self.assertLess(abs(uniform.gain - optimal_evaluation.gain), 1.0e-8)
        self.assertLess(uniform.generator_residual, 1.0e-10)
        self.assertLess(float(uniform.poisson_residual), 1.0e-10)

    def test_known_composition_reversal_has_distinct_minimizers(self) -> None:
        parameters = N3ModelParameters.homogeneous(
            K=2, B=2, arrival_rate=1.8, mu_vm=1.0
        )
        model = N3AverageCostCTMDP(parameters)
        solution = model.relative_value_iteration()
        left = (0, 1, 1, 0, 1, 0)
        right = (1, 0, 0, 1, 1, 0)
        self.assertEqual(solution.optimal_action_sets[model.index[left]], (1,))
        self.assertEqual(solution.optimal_action_sets[model.index[right]], (2,))


if __name__ == "__main__":
    unittest.main()
