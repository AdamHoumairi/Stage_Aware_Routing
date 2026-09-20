#!/usr/bin/env python3
"""Regenerate the manuscript's numerical Figure 2 from frozen results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_maximum_gaps(path: Path) -> dict[tuple[int, int, float], float]:
    maxima: dict[tuple[int, int, float], float] = {}
    load_counts: dict[tuple[int, int, float], set[float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["K"]),
                int(row["B"]),
                float(row["stage_capacity_ratio"]),
            )
            gap_percent = 100.0 * float(row["uniform_jsq_relative_gap"])
            maxima[key] = max(maxima.get(key, 0.0), gap_percent)
            load_counts.setdefault(key, set()).add(float(row["rho"]))
    if len(maxima) != 32 or any(len(loads) != 5 for loads in load_counts.values()):
        raise ValueError("Expected a complete 2 x 4 x 4 grid with five loads.")
    return maxima


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/paper/n2_policy_grid.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/figures")
    )
    args = parser.parse_args()

    maxima = read_maximum_gaps(args.input)
    capacities = (1, 2, 3, 4)
    ratios = (0.25, 0.5, 1.0, 2.0)
    maximum = max(maxima.values())
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.35, 3.22), sharey=True)
    image = None
    for axis, K in zip(axes, (2, 4)):
        values = np.asarray(
            [[maxima[(K, B, eta)] for eta in ratios] for B in capacities]
        )
        image = axis.imshow(
            values, cmap="Blues", vmin=0.0, vmax=maximum, aspect="auto"
        )
        axis.set_title(f"K={K} containers per VM")
        axis.set_xticks(range(len(ratios)), ["0.25", "0.50", "1.00", "2.00"])
        axis.set_yticks(range(len(capacities)), [str(B) for B in capacities])
        axis.set_xlabel(r"Stage-capacity ratio $\eta$")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                label = "0" if abs(value) < 5.0e-7 else f"{value:.2f}"
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    color="white" if value > 0.56 * maximum else "black",
                    fontsize=8,
                )
    axes[0].set_ylabel("Feeder capacity B")
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, fraction=0.035, pad=0.035)
    colorbar.set_label("Maximum uniform-tie JSQ gap over load (%)")
    figure.subplots_adjust(
        left=0.09, right=0.88, bottom=0.19, top=0.88, wspace=0.10
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output_dir / "fig2_uniform_tie_jsq_gap_heatmap.pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        args.output_dir / "fig2_uniform_tie_jsq_gap_heatmap.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()

