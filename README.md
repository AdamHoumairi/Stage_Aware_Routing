# Stage-Aware Routing: Reorganized CTMDP Code

This compact package contains compatible implementations of the finite
average-cost CTMDP experiments used to study stage-aware routing in two-stage
VM-container systems.

The supplied standalone files have been reorganized as a Python package so
that relative imports resolve correctly. The incompatible file
`n2_average_cost_ctmdp(1).py` has been replaced by the compatible
`ctmdp_routing/n2_solver.py`, and public experiment files no longer use the
provisional `pilot` suffix.

## Contents

- `ctmdp_routing/n2_solver.py`: two-VM CTMDP, RVI and exact policy evaluation.
- `ctmdp_routing/n3_solver.py`: three-VM CTMDP and composition-reversal audit.
- `ctmdp_routing/z_policy_milp.py`: globally certified deterministic
  observation-constrained D-Z policy.
- `ctmdp_routing/z_policy_randomized.py`: multistart randomized
  observation-constrained R-Z optimization.
- `ctmdp_routing/n2_jsq_grid.py`: 160-case uniform-tie JSQ grid.
- `ctmdp_routing/n2_dz_experiment.py`: one D-Z comparison.
- `ctmdp_routing/n2_rz_experiment.py`: one R-Z comparison.
- `ctmdp_routing/n3_jsq_confirmation.py`: 18-case uniform-tie JSQ confirmation.
- `tests/test_ctmdp.py`: deterministic regression tests.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Verification

Run the complete regression suite:

```bash
python run_experiments.py test
```

Run the tests plus one representative N=2 case and one N=3 case:

```bash
python run_experiments.py check
```

## Experiments

```bash
python run_experiments.py n2-grid
python run_experiments.py n2-dz --K 2 --B 2 --arrival-rate 1 --mu-vm 1
python run_experiments.py n2-rz --K 2 --B 2 --arrival-rate 1 --mu-vm 1
python run_experiments.py n3
```

Arguments written after the study name are passed to that experiment. Use,
for example, `python run_experiments.py n2-grid --limit 2` for a short run.
All outputs are written under `results/` unless another `--output-dir` is
specified.

## Scope

This package reorganizes and repairs the files supplied for the uniform-tie
JSQ and D-Z/R-Z analyses. The weighted feeder-first and cost-sensitivity
runners should be added separately before freezing the final repository
release if those manuscript results are to be reproduced from the same
snapshot.
