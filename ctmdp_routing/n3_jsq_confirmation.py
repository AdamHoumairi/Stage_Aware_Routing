"""Evaluate uniform-tie JSQ on the 18 homogeneous N=3 confirmations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .n3_solver import (
    N3AverageCostCTMDP,
    N3ModelParameters,
    relative_gain_gap,
)


NUMERICAL_OPTIMALITY_TOLERANCE = 1.0e-8
RVI_TOLERANCE = 1.0e-11
TIE_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class Scenario:
    """One homogeneous three-VM confirmation configuration."""

    scenario_id: str
    family: str
    parameters: N3ModelParameters


def _homogeneous_scenario(
    scenario_id: str, K: int, B: int, eta: float, rho: float
) -> Scenario:
    mu_container = 1.0
    mu_vm = eta * K * mu_container
    arrival_rate = rho * 3.0 * min(mu_vm, K * mu_container)
    return Scenario(
        scenario_id=scenario_id,
        family=f"homogeneous_core_K{K}",
        parameters=N3ModelParameters.homogeneous(
            K=K,
            B=B,
            arrival_rate=arrival_rate,
            mu_vm=mu_vm,
            mu_container=mu_container,
            holding_vm=1.0,
            holding_container=1.0,
            blocking_cost=10.0,
        ),
    )


def build_scenarios() -> list[Scenario]:
    """Return the 18 homogeneous configurations in manuscript order."""

    scenarios: list[Scenario] = []
    number = 1
    for B in (1, 2):
        for eta in (0.5, 1.0, 2.0):
            for rho in (0.6, 0.9):
                scenarios.append(
                    _homogeneous_scenario(
                        f"H{number:02d}", 2, B, eta, rho
                    )
                )
                number += 1
    for B in (1, 2):
        for eta in (0.5, 1.0, 2.0):
            scenarios.append(
                _homogeneous_scenario(
                    f"H{number:02d}", 4, B, eta, 0.75
                )
            )
            number += 1
    if len(scenarios) != 18:
        raise AssertionError("The N=3 confirmation design must contain 18 cases.")
    return scenarios


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _observation(
    model: N3AverageCostCTMDP, state: tuple[int, ...]
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            sum(model.local_state(state, vm)),
            int(model.local_state(state, vm)[0] < model.p.B[vm - 1]),
        )
        for vm in (1, 2, 3)
    )


def _z_measurability_error(
    model: N3AverageCostCTMDP, probabilities: np.ndarray
) -> float:
    action_by_observation: dict[object, np.ndarray] = {}
    error = 0.0
    for state_index, state in enumerate(model.states):
        observation = _observation(model, state)
        previous = action_by_observation.setdefault(
            observation, probabilities[state_index]
        )
        error = max(
            error,
            float(np.max(np.abs(previous - probabilities[state_index]))),
        )
    return error


def run_scenario(scenario: Scenario) -> dict[str, object]:
    model = N3AverageCostCTMDP(scenario.parameters)
    optimal = model.relative_value_iteration(
        tolerance=RVI_TOLERANCE,
        tie_tolerance=TIE_TOLERANCE,
    )
    optimal_evaluation = model.evaluate_policy(
        optimal.policy, "full_state_optimal", poisson_check=True
    )
    lowest_index_evaluation = model.evaluate_policy(
        model.jsq_policy(), "jsq_lowest_index_ties"
    )
    uniform_probabilities = model.randomized_tie_jsq_action_probabilities()
    uniform_evaluation = model.evaluate_action_probabilities(
        uniform_probabilities,
        "jsq_uniform_ties",
        poisson_check=True,
    )
    checks = model.invariant_checks(optimal, optimal_evaluation)

    tie_action_count = np.sum(uniform_probabilities[:, 1:] > 0.0, axis=1)
    p = scenario.parameters
    row: dict[str, object] = {
        "scenario_id": scenario.scenario_id,
        "family": scenario.family,
        "N": 3,
        "homogeneous_K": p.K[0],
        "homogeneous_B": p.B[0],
        "rho": p.rho,
        "lambda": p.arrival_rate,
        "homogeneous_stage_capacity_ratio": p.stage_capacity_ratios[0],
        "states": model.number_of_states,
        "uniformization_rate": p.uniformization_rate,
        "optimal_gain": optimal_evaluation.gain,
        "jsq_lowest_index_gain": lowest_index_evaluation.gain,
        "jsq_lowest_index_relative_gain_gap": relative_gain_gap(
            lowest_index_evaluation.gain, optimal_evaluation.gain
        ),
        "jsq_uniform_tie_gain": uniform_evaluation.gain,
        "jsq_uniform_tie_absolute_gain_gap": (
            uniform_evaluation.gain - optimal_evaluation.gain
        ),
        "jsq_uniform_tie_relative_gain_gap": relative_gain_gap(
            uniform_evaluation.gain, optimal_evaluation.gain
        ),
        "uniform_minus_lowest_index_gain": (
            uniform_evaluation.gain - lowest_index_evaluation.gain
        ),
        "uniform_tie_state_count": int(np.sum(tie_action_count >= 2)),
        "uniform_tie_stationary_probability": float(
            uniform_evaluation.stationary @ (tie_action_count >= 2).astype(float)
        ),
        "uniform_tie_z_measurability_error": _z_measurability_error(
            model, uniform_probabilities
        ),
        "jsq_uniform_tie_blocking_probability": (
            uniform_evaluation.blocking_probability
        ),
        "jsq_uniform_tie_mean_vm_occupancy": (
            uniform_evaluation.mean_vm_occupancy
        ),
        "jsq_uniform_tie_mean_container_occupancy": (
            uniform_evaluation.mean_container_occupancy
        ),
        "jsq_uniform_tie_mean_total_occupancy": (
            uniform_evaluation.mean_total_occupancy
        ),
        "jsq_uniform_tie_mean_response_time": (
            uniform_evaluation.mean_response_time
        ),
        "jsq_uniform_tie_generator_residual": (
            uniform_evaluation.generator_residual
        ),
        "jsq_uniform_tie_poisson_residual": (
            uniform_evaluation.poisson_residual
        ),
        "jsq_uniform_tie_poisson_gain_error": (
            uniform_evaluation.poisson_gain_error
        ),
        "jsq_uniform_tie_cost_decomposition_error": (
            uniform_evaluation.cost_decomposition_error
        ),
        "rvi_iterations": optimal.iterations,
        "rvi_span_residual": optimal.span_residual,
        **{f"check_{name}": value for name, value in checks.items()},
    }

    for name, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise AssertionError(f"Non-finite result {name}={value}.")
    if float(row["uniform_tie_z_measurability_error"]) > 1.0e-12:
        raise AssertionError("Uniform-tie JSQ is not Z-measurable.")
    numerical_errors = (
        float(row["jsq_uniform_tie_generator_residual"]),
        float(row["jsq_uniform_tie_poisson_residual"]),
        float(row["jsq_uniform_tie_poisson_gain_error"]),
        float(row["jsq_uniform_tie_cost_decomposition_error"]),
    )
    if max(numerical_errors) > 1.0e-8:
        raise AssertionError("An N=3 uniform-tie evaluation residual exceeds 1e-8.")
    return row


def aggregate_rows(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    result = []
    for feeder_capacity in sorted({int(row["homogeneous_B"]) for row in rows}):
        selected = [
            row for row in rows if int(row["homogeneous_B"]) == feeder_capacity
        ]
        gaps = np.asarray(
            [float(row["jsq_uniform_tie_relative_gain_gap"]) for row in selected]
        )
        maximum_position = int(np.argmax(gaps))
        maximum_row = selected[maximum_position]
        result.append(
            {
                "group": f"B={feeder_capacity}",
                "cases": len(selected),
                "uniform_tie_numerically_optimal": int(
                    np.sum(gaps <= NUMERICAL_OPTIMALITY_TOLERANCE)
                ),
                "uniform_tie_mean_gap_percent": 100.0 * float(gaps.mean()),
                "uniform_tie_max_gap_percent": 100.0 * float(gaps.max()),
                "uniform_tie_max_scenario_id": maximum_row["scenario_id"],
                "uniform_tie_max_K": maximum_row["homogeneous_K"],
                "uniform_tie_max_eta": maximum_row[
                    "homogeneous_stage_capacity_ratio"
                ],
                "uniform_tie_max_rho": maximum_row["rho"],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/n3_jsq_confirmation"),
    )
    parser.add_argument("--scenario-ids", nargs="+")
    args = parser.parse_args()

    scenarios = build_scenarios()
    if args.scenario_ids:
        requested = set(args.scenario_ids)
        scenarios = [item for item in scenarios if item.scenario_id in requested]
        if len(scenarios) != len(requested):
            raise ValueError("Unknown scenario identifier requested.")

    rows = []
    for number, scenario in enumerate(scenarios, start=1):
        row = run_scenario(scenario)
        rows.append(row)
        print(
            f"[{number:>2d}/{len(scenarios)}] {scenario.scenario_id}: "
            f"uniform-JSQ gap="
            f"{100.0 * float(row['jsq_uniform_tie_relative_gain_gap']):.3f}%"
        )

    aggregates = aggregate_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "scenario_summary.csv", rows)
    write_csv(args.output_dir / "aggregate_summary.csv", aggregates)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "study": "uniform_tie_JSQ_N3_homogeneous_confirmation",
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
