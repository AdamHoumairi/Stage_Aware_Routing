"""Evaluate uniform-tie JSQ on the 160-instance homogeneous N=2 grid."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np

from .n2_solver import (
    ModelParameters,
    N2AverageCostCTMDP,
    PolicyEvaluation,
    action_set_signature,
    assert_finite_summary,
    numeric_token,
    relative_gain_gap,
    write_csv,
)
from .z_policy_milp import backlog_feasibility_observation


NUMERICAL_OPTIMALITY_TOLERANCE = 1.0e-8
RVI_TOLERANCE = 1.0e-11
TIE_TOLERANCE = 1.0e-10


def build_parameters() -> list[ModelParameters]:
    """Return the 160 homogeneous two-VM configurations used in the paper."""

    configurations: list[ModelParameters] = []
    for K, B, eta, rho in product(
        (2, 4),
        (1, 2, 3, 4),
        (0.25, 0.5, 1.0, 2.0),
        (0.25, 0.5, 0.75, 1.0, 1.5),
    ):
        mu_container = 1.0
        mu_vm = eta * K * mu_container
        arrival_rate = rho * 2.0 * min(mu_vm, K * mu_container)
        configurations.append(
            ModelParameters(
                K=K,
                B=B,
                arrival_rate=arrival_rate,
                mu_vm=mu_vm,
                mu_container=mu_container,
                holding_vm=1.0,
                holding_container=1.0,
                blocking_cost=10.0,
            )
        )
    return configurations


def _evaluation_fields(
    prefix: str, evaluation: PolicyEvaluation
) -> dict[str, object]:
    """Return policy-performance and numerical-audit fields."""

    return {
        f"{prefix}_gain": evaluation.gain,
        f"{prefix}_blocking_probability": evaluation.blocking_probability,
        f"{prefix}_mean_vm_occupancy": evaluation.mean_vm_occupancy,
        f"{prefix}_mean_container_occupancy": (
            evaluation.mean_container_occupancy
        ),
        f"{prefix}_mean_total_occupancy": evaluation.mean_total_occupancy,
        f"{prefix}_mean_response_time": evaluation.mean_response_time,
        f"{prefix}_generator_residual": evaluation.generator_residual,
        f"{prefix}_poisson_gain_error": evaluation.poisson_gain_error,
        f"{prefix}_flow_residual_arrival_promotion": (
            evaluation.flow_residual_arrival_promotion
        ),
        f"{prefix}_flow_residual_promotion_departure": (
            evaluation.flow_residual_promotion_departure
        ),
        f"{prefix}_cost_decomposition_error": (
            evaluation.cost_decomposition_error
        ),
    }


def _z_measurability_error(
    model: N2AverageCostCTMDP, probabilities: np.ndarray
) -> float:
    """Return the largest policy difference within an observation fibre."""

    action_by_observation: dict[object, np.ndarray] = {}
    error = 0.0
    for state_index, state in enumerate(model.states):
        observation = backlog_feasibility_observation(model, state)
        previous = action_by_observation.setdefault(
            observation, probabilities[state_index]
        )
        error = max(
            error,
            float(np.max(np.abs(previous - probabilities[state_index]))),
        )
    return error


def _vm_swap_equivariance_error(
    model: N2AverageCostCTMDP, probabilities: np.ndarray
) -> float:
    """Return the largest error after swapping the two VM labels."""

    error = 0.0
    for state_index, state in enumerate(model.states):
        swapped_index = model.index[model.swap_vms(state)]
        swapped_probabilities = probabilities[swapped_index]
        error = max(
            error,
            abs(float(probabilities[state_index, 0] - swapped_probabilities[0])),
            abs(float(probabilities[state_index, 1] - swapped_probabilities[2])),
            abs(float(probabilities[state_index, 2] - swapped_probabilities[1])),
        )
    return error


def run_configuration(parameters: ModelParameters) -> dict[str, object]:
    """Solve and evaluate one full-state and two JSQ policies."""

    model = N2AverageCostCTMDP(parameters)
    optimal = model.relative_value_iteration(
        tolerance=RVI_TOLERANCE,
        tie_tolerance=TIE_TOLERANCE,
    )
    optimal_evaluation = model.evaluate_policy(
        optimal.policy, "full_state_optimal"
    )
    checks = model.run_invariant_tests(optimal, optimal_evaluation)

    lowest_index = model.evaluate_policy(
        model.jsq_policy(), "jsq_lowest_index_ties"
    )
    uniform_probabilities = model.randomized_tie_jsq_action_probabilities()
    uniform_tie = model.evaluate_action_probabilities(
        uniform_probabilities, "jsq_uniform_ties"
    )

    tie_mask = (
        (uniform_probabilities[:, 1] > 0.0)
        & (uniform_probabilities[:, 2] > 0.0)
    )
    uniform_gap = relative_gain_gap(
        uniform_tie.gain, optimal_evaluation.gain
    )
    lowest_index_gap = relative_gain_gap(
        lowest_index.gain, optimal_evaluation.gain
    )

    summary = {
        "scenario_id": (
            f"K{parameters.K}_B{parameters.B}_"
            f"lambda{numeric_token(parameters.arrival_rate)}_"
            f"rho{numeric_token(parameters.rho)}_"
            f"muvm{numeric_token(parameters.mu_vm)}_"
            f"muc{numeric_token(parameters.mu_container)}_"
            f"hvm{numeric_token(parameters.holding_vm)}_"
            f"hc{numeric_token(parameters.holding_container)}_"
            f"cb{numeric_token(parameters.blocking_cost)}"
        ),
        "K": parameters.K,
        "B": parameters.B,
        "lambda": parameters.arrival_rate,
        "rho": parameters.rho,
        "mu_vm": parameters.mu_vm,
        "mu_container": parameters.mu_container,
        "stage_capacity_ratio": parameters.stage_capacity_ratio,
        "holding_vm": parameters.holding_vm,
        "holding_container": parameters.holding_container,
        "blocking_cost": parameters.blocking_cost,
        "states": model.number_of_states,
        "uniformization_rate": parameters.uniformization_rate,
        "rvi_iterations": optimal.iterations,
        "rvi_span_residual": optimal.span_residual,
        "bellman_residual": optimal.bellman_residual,
        "optimal_action_set_signature": action_set_signature(
            optimal.optimal_action_sets
        ),
        "jsq_uniform_tie_relative_gain_gap": uniform_gap,
        "jsq_uniform_tie_absolute_gain_gap": (
            uniform_tie.gain - optimal_evaluation.gain
        ),
        "jsq_lowest_index_relative_gain_gap": lowest_index_gap,
        "jsq_lowest_index_absolute_gain_gap": (
            lowest_index.gain - optimal_evaluation.gain
        ),
        "uniform_minus_lowest_index_gain": (
            uniform_tie.gain - lowest_index.gain
        ),
        "uniform_tie_state_count": int(np.sum(tie_mask)),
        "uniform_tie_stationary_probability": float(
            uniform_tie.stationary @ tie_mask.astype(float)
        ),
        "uniform_tie_z_measurability_error": _z_measurability_error(
            model, uniform_probabilities
        ),
        "uniform_tie_vm_swap_equivariance_error": (
            _vm_swap_equivariance_error(model, uniform_probabilities)
        ),
        **_evaluation_fields("optimal", optimal_evaluation),
        **_evaluation_fields("jsq_lowest_index", lowest_index),
        **_evaluation_fields("jsq_uniform_tie", uniform_tie),
        **{f"check_{name}": value for name, value in checks.items()},
    }
    assert_finite_summary(summary)
    if summary["uniform_tie_z_measurability_error"] > 1.0e-12:
        raise AssertionError("Uniform-tie JSQ is not Z-measurable.")
    if summary["uniform_tie_vm_swap_equivariance_error"] > 1.0e-12:
        raise AssertionError("Uniform-tie JSQ is not VM-swap equivariant.")
    numerical_errors = [
        float(summary[f"{prefix}_{suffix}"])
        for prefix in ("optimal", "jsq_lowest_index", "jsq_uniform_tie")
        for suffix in (
            "generator_residual",
            "poisson_gain_error",
            "flow_residual_arrival_promotion",
            "flow_residual_promotion_departure",
            "cost_decomposition_error",
        )
    ]
    if max(numerical_errors) > 1.0e-8:
        raise AssertionError(
            "A stationary-policy evaluation residual exceeds 1e-8."
        )
    return summary


def aggregate_rows(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate the policy gaps by feeder capacity and over B >= 2."""

    def summarize(
        label: str, selected: Sequence[dict[str, object]]
    ) -> dict[str, object]:
        uniform_gaps = np.asarray(
            [float(row["jsq_uniform_tie_relative_gain_gap"]) for row in selected]
        )
        deterministic_gaps = np.asarray(
            [
                float(row["jsq_lowest_index_relative_gain_gap"])
                for row in selected
            ]
        )
        maximum_position = int(np.argmax(uniform_gaps))
        maximum_row = selected[maximum_position]
        return {
            "group": label,
            "cases": len(selected),
            "uniform_tie_numerically_optimal": int(
                np.sum(uniform_gaps <= NUMERICAL_OPTIMALITY_TOLERANCE)
            ),
            "uniform_tie_mean_gap_percent": 100.0 * float(uniform_gaps.mean()),
            "uniform_tie_max_gap_percent": 100.0 * float(uniform_gaps.max()),
            "lowest_index_mean_gap_percent": (
                100.0 * float(deterministic_gaps.mean())
            ),
            "lowest_index_max_gap_percent": (
                100.0 * float(deterministic_gaps.max())
            ),
            "uniform_tie_max_scenario_id": maximum_row["scenario_id"],
            "uniform_tie_max_K": maximum_row["K"],
            "uniform_tie_max_B": maximum_row["B"],
            "uniform_tie_max_eta": maximum_row["stage_capacity_ratio"],
            "uniform_tie_max_rho": maximum_row["rho"],
        }

    aggregates = []
    for feeder_capacity in sorted({int(row["B"]) for row in rows}):
        selected = [row for row in rows if int(row["B"]) == feeder_capacity]
        aggregates.append(summarize(f"B={feeder_capacity}", selected))
    positive_capacity = [row for row in rows if int(row["B"]) >= 2]
    if positive_capacity:
        aggregates.append(summarize("B>=2", positive_capacity))
    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/n2_jsq_grid"),
    )
    parser.add_argument(
        "--limit", type=int, help="Run only the first N configurations."
    )
    args = parser.parse_args()

    configurations = build_parameters()
    if args.limit is not None:
        configurations = configurations[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for number, parameters in enumerate(configurations, start=1):
        row = run_configuration(parameters)
        rows.append(row)
        print(
            f"[{number:>3d}/{len(configurations)}] K={parameters.K}, "
            f"B={parameters.B}, eta={parameters.stage_capacity_ratio:g}, "
            f"rho={parameters.rho:g}, uniform-JSQ gap="
            f"{100.0 * float(row['jsq_uniform_tie_relative_gain_gap']):.3f}%"
        )

    aggregates = aggregate_rows(rows)
    write_csv(args.output_dir / "scenario_summary.csv", rows)
    write_csv(args.output_dir / "aggregate_summary.csv", aggregates)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "study": "uniform_tie_JSQ_N2_grid",
                "scenario_count": len(rows),
                "jsq_tie_breaking": "uniform_over_feasible_minimum-backlog_VMs",
                "numerical_optimality_tolerance": (
                    NUMERICAL_OPTIMALITY_TOLERANCE
                ),
                "rvi_tolerance": RVI_TOLERANCE,
                "tie_tolerance": TIE_TOLERANCE,
                "aggregate_summary": aggregates,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
