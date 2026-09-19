"""Run an exact deterministic Z-measurable policy comparison for N=2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .n2_solver import ModelParameters, N2AverageCostCTMDP, relative_gain_gap
from .z_policy_milp import (
    backlog_feasibility_observation,
    solve_deterministic_z_policy,
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
    parser.add_argument("--mip-relative-gap", type=float, default=1.0e-9)
    parser.add_argument("--mip-absolute-gap", type=float, default=1.0e-10)
    parser.add_argument("--mip-feasibility-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--lp-feasibility-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--certification-relative-gap", type=float, default=1.0e-8)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--node-limit", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/n2_dz_experiment"),
    )
    parser.add_argument("--solver-log", action="store_true")
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
    full_evaluation = model.evaluate_policy(full_solution.policy, "full_state_optimal")
    dz = solve_deterministic_z_policy(
        model,
        mip_relative_gap=args.mip_relative_gap,
        mip_absolute_gap=args.mip_absolute_gap,
        mip_feasibility_tolerance=args.mip_feasibility_tolerance,
        lp_feasibility_tolerance=args.lp_feasibility_tolerance,
        certification_relative_gap=args.certification_relative_gap,
        time_limit=args.time_limit,
        node_limit=args.node_limit,
        display_solver_output=args.solver_log,
    )
    jsq_deterministic = model.evaluate_policy(
        model.jsq_policy(), "jsq_lowest_index_ties"
    )
    jsq_uniform = model.evaluate_action_probabilities(
        model.randomized_tie_jsq_action_probabilities(),
        "jsq_uniform_ties",
    )

    summary = {
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
        "dz_deterministic_gain": dz.gain,
        "dz_deterministic_relative_gap": relative_gain_gap(
            dz.gain, full_evaluation.gain
        ),
        "jsq_lowest_index_gain": jsq_deterministic.gain,
        "jsq_lowest_index_relative_gap": relative_gain_gap(
            jsq_deterministic.gain, full_evaluation.gain
        ),
        "jsq_uniform_tie_gain": jsq_uniform.gain,
        "jsq_uniform_tie_relative_gap": relative_gain_gap(
            jsq_uniform.gain, full_evaluation.gain
        ),
        **{f"dz_{key}": value for key, value in dz.summary_dict().items()},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "policy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "state_index",
                "q1",
                "n1",
                "q2",
                "n2",
                "l1",
                "f1",
                "l2",
                "f2",
                "feasible_actions",
                "dz_action",
                "stationary_probability",
            ),
        )
        writer.writeheader()
        for state_index, state in enumerate(model.states):
            observation = backlog_feasibility_observation(model, state)
            writer.writerow(
                {
                    "state_index": state_index,
                    "q1": state[0],
                    "n1": state[1],
                    "q2": state[2],
                    "n2": state[3],
                    "l1": observation[0][0],
                    "f1": observation[0][1],
                    "l2": observation[1][0],
                    "f2": observation[1][1],
                    "feasible_actions": ";".join(
                        map(str, model.feasible_actions(state))
                    ),
                    "dz_action": int(dz.policy[state_index]),
                    "stationary_probability": dz.stationary[state_index],
                }
            )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
