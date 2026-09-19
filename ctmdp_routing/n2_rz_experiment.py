"""Compare a numerically optimized randomized Z-policy with CTMDP baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .n2_solver import ModelParameters, N2AverageCostCTMDP, relative_gain_gap
from .z_policy_milp import solve_deterministic_z_policy
from .z_policy_randomized import (
    N2RandomizedZPolicyProblem,
    solve_randomized_z_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument("--mu-vm", type=float, default=1.0)
    parser.add_argument("--mu-container", type=float, default=1.0)
    parser.add_argument("--holding-vm", type=float, default=1.0)
    parser.add_argument("--holding-container", type=float, default=1.0)
    parser.add_argument("--blocking-cost", type=float, default=10.0)
    parser.add_argument("--random-starts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--optimizer-max-iterations", type=int, default=500)
    parser.add_argument(
        "--skip-dz",
        action="store_true",
        help="Skip the globally certified deterministic Z-policy benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/n2_rz_experiment"),
    )
    args = parser.parse_args()

    parameters = ModelParameters(
        K=args.K,
        B=args.B,
        arrival_rate=args.arrival_rate,
        mu_vm=args.mu_vm,
        mu_container=args.mu_container,
        holding_vm=args.holding_vm,
        holding_container=args.holding_container,
        blocking_cost=args.blocking_cost,
    )
    model = N2AverageCostCTMDP(parameters)
    full_solution = model.relative_value_iteration()
    full_evaluation = model.evaluate_policy(
        full_solution.policy, "full_state_optimal"
    )
    lowest_index = model.evaluate_policy(
        model.jsq_policy(), "jsq_lowest_index_ties"
    )
    uniform_probabilities = model.randomized_tie_jsq_action_probabilities()
    uniform_tie = model.evaluate_action_probabilities(
        uniform_probabilities, "jsq_uniform_ties"
    )

    dz = None
    extra_starts = {}
    if not args.skip_dz:
        dz = solve_deterministic_z_policy(model)
        extra_starts["deterministic_z_optimum"] = (
            model.deterministic_action_probabilities(dz.policy)
        )

    rz = solve_randomized_z_policy(
        model,
        initial_action_probabilities=extra_starts,
        random_starts=args.random_starts,
        seed=args.seed,
        maximum_iterations=args.optimizer_max_iterations,
        global_lower_bound=full_evaluation.gain,
    )

    tolerance = 1.0e-8 * max(1.0, abs(full_evaluation.gain))
    if rz.gain < full_evaluation.gain - tolerance:
        raise AssertionError("The R-Z candidate is below the full-state optimum.")
    if rz.gain > uniform_tie.gain + tolerance:
        raise AssertionError("R-Z optimization failed to recover uniform-tie JSQ.")
    if dz is not None and rz.gain > dz.gain + tolerance:
        raise AssertionError("R-Z optimization failed to recover the D-Z policy.")

    numerical_zero = 1.0e-12 * max(1.0, abs(full_evaluation.gain))
    observation_gap = rz.gain - full_evaluation.gain
    uniform_rule_gap = uniform_tie.gain - rz.gain
    if abs(observation_gap) <= numerical_zero:
        observation_gap = 0.0
    if abs(uniform_rule_gap) <= numerical_zero:
        uniform_rule_gap = 0.0
    summary: dict[str, object] = {
        "K": parameters.K,
        "B": parameters.B,
        "arrival_rate": parameters.arrival_rate,
        "rho": parameters.rho,
        "mu_vm": parameters.mu_vm,
        "mu_container": parameters.mu_container,
        "stage_capacity_ratio": parameters.stage_capacity_ratio,
        "holding_vm": parameters.holding_vm,
        "holding_container": parameters.holding_container,
        "blocking_cost": parameters.blocking_cost,
        "full_state_optimal_gain": full_evaluation.gain,
        "rz_candidate_gain": rz.gain,
        "rz_candidate_relative_gap": relative_gain_gap(
            rz.gain, full_evaluation.gain
        ),
        "dz_gain": None if dz is None else dz.gain,
        "dz_relative_gap": (
            None
            if dz is None
            else relative_gain_gap(dz.gain, full_evaluation.gain)
        ),
        "rz_improvement_over_dz": (
            None if dz is None else dz.gain - rz.gain
        ),
        "jsq_lowest_index_gain": lowest_index.gain,
        "jsq_lowest_index_relative_gap": relative_gain_gap(
            lowest_index.gain, full_evaluation.gain
        ),
        "jsq_uniform_tie_gain": uniform_tie.gain,
        "jsq_uniform_tie_relative_gap": relative_gain_gap(
            uniform_tie.gain, full_evaluation.gain
        ),
        "candidate_observation_gap": observation_gap,
        "candidate_uniform_rule_gap": uniform_rule_gap,
        "candidate_observation_gap_share": (
            observation_gap / (uniform_tie.gain - full_evaluation.gain)
            if uniform_tie.gain > full_evaluation.gain
            else 0.0
        ),
        "candidate_decomposition_error": abs(
            (uniform_tie.gain - full_evaluation.gain)
            - observation_gap
            - uniform_rule_gap
        ),
        **{f"rz_{key}": value for key, value in rz.summary_dict().items()},
        "rz_optimization_runs": rz.run_dicts(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    problem = N2RandomizedZPolicyProblem(model)
    lowest_probabilities = model.deterministic_action_probabilities(
        model.jsq_policy()
    )
    with (args.output_dir / "observation_policy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = (
            "l1",
            "f1",
            "l2",
            "f2",
            "feasible_actions",
            "rz_probability_block",
            "rz_probability_vm1",
            "rz_probability_vm2",
            "uniform_jsq_probability_vm1",
            "uniform_jsq_probability_vm2",
            "lowest_index_jsq_action",
            "dz_action",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for observation in problem.ordered_observations:
            state_index = problem.state_observations.index(observation)
            rz_row = rz.action_probabilities[state_index]
            uniform_row = uniform_probabilities[state_index]
            lowest_row = lowest_probabilities[state_index]
            writer.writerow(
                {
                    "l1": observation[0][0],
                    "f1": observation[0][1],
                    "l2": observation[1][0],
                    "f2": observation[1][1],
                    "feasible_actions": ";".join(
                        map(str, problem.observation_actions[observation])
                    ),
                    "rz_probability_block": rz_row[0],
                    "rz_probability_vm1": rz_row[1],
                    "rz_probability_vm2": rz_row[2],
                    "uniform_jsq_probability_vm1": uniform_row[1],
                    "uniform_jsq_probability_vm2": uniform_row[2],
                    "lowest_index_jsq_action": int(lowest_row.argmax()),
                    "dz_action": (
                        "" if dz is None else dz.observation_actions[observation]
                    ),
                }
            )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
