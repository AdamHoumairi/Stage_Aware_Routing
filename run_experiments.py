#!/usr/bin/env python3
"""Single entry point for the reorganized CTMDP experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys


MODULES = {
    "n2-grid": "ctmdp_routing.n2_jsq_grid",
    "n2-dz": "ctmdp_routing.n2_dz_experiment",
    "n2-rz": "ctmdp_routing.n2_rz_experiment",
    "n3": "ctmdp_routing.n3_jsq_confirmation",
}


def _run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study",
        choices=(*MODULES, "test", "check"),
        help="Experiment or verification task to run.",
    )
    args, forwarded = parser.parse_known_args()

    if args.study == "test":
        _run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-v",
            ]
        )
        return

    if args.study == "check":
        if forwarded:
            parser.error("The check task does not accept additional arguments.")
        _run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-v",
            ]
        )
        _run(
            [
                sys.executable,
                "-m",
                MODULES["n2-grid"],
                "--limit",
                "1",
                "--output-dir",
                "results/check_n2",
            ]
        )
        _run(
            [
                sys.executable,
                "-m",
                MODULES["n3"],
                "--scenario-ids",
                "H01",
                "--output-dir",
                "results/check_n3",
            ]
        )
        return

    _run([sys.executable, "-m", MODULES[args.study], *forwarded])


if __name__ == "__main__":
    main()
