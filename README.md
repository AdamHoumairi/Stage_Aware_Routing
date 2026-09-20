# Stage-Aware Routing in Finite Two-Stage Cloud Service Systems

This repository is the compact reproducibility package for the manuscript
**“Stage-Aware Routing in Finite Two-Stage Cloud Service Systems: A CTMDP
Study.”** It contains the finite average-cost CTMDP solvers, policy-class
optimizers, frozen manuscript results, and deterministic numerical audits.

## Installation

Python **3.11 or newer** is required by the pinned NumPy and SciPy versions.

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

## Quick verification

```bash
python run_experiments.py test
python run_experiments.py check
python run_experiments.py audit
```

`test` runs the deterministic regression suite. `check` additionally solves
representative N=2 and N=3 cases. `audit` verifies the frozen CSV files and the
headline values reported in the manuscript.

## Manuscript experiments

| Manuscript result | Command | Main generated file |
|---|---|---|
| 160-case UJSQ grid | `python run_experiments.py n2-grid` | `scenario_summary.csv` |
| UFF and optimized WFF | `python run_experiments.py n2-wff-grid --workers 4` | `scenario_summary.csv` |
| 108-case cost sensitivity | `python run_experiments.py cost-sensitivity --workers 4` | `scenario_summary.csv` |
| Complete R-Z grid | `python run_experiments.py n2-rz-grid --workers 4` | `scenario_summary.csv` |
| D-Z/R-Z representative cases | `python run_experiments.py observation-table` | `table4_observation_policies.csv` |
| 18-case N=3 confirmation | `python run_experiments.py n3` | `scenario_summary.csv` |

Run every study with:

```bash
python run_experiments.py paper --workers 4
```

This is the full calculation and can take several hours, principally because
of the multistart R-Z grid. Add `--limit 2` for a short end-to-end smoke run.
Scenario checkpoints allow interrupted WFF, cost-sensitivity, and R-Z runs to
resume without recomputing completed cases.

## Policy classes

- **UJSQ:** uniform randomization among feasible VMs with minimum total
  backlog.
- **UFF:** uniform randomization among feasible VMs with lexicographically
  minimum feeder/container occupancy `(q_i, n_i)`.
- **WFF:** uniform randomization among minimizers of `r*q_i + n_i`, with
  `r >= 1`. The optimizer evaluates every rational critical weight and one
  representative of every open interval, giving a global optimum within this
  one-parameter class.
- **D-Z:** deterministic policies measurable with respect to total backlog and
  admission feasibility. The MILP solution is globally certified.
- **R-Z:** stationary randomized policies under the same observation. The
  analytic-gradient reflected multistart search returns a feasible candidate;
  global optimality is claimed only when it reaches the full-state lower bound.

WFF is optimized separately in each scenario and is therefore an
expressiveness benchmark, not a deployable adaptive tuning rule.

## Frozen results

The reviewer-facing files are under `results/paper/`:

- `n2_policy_grid.csv`: full-state, UJSQ, UFF, and optimized WFF results;
- `n2_rz_grid.csv`: complete observation-constrained randomized-policy grid;
- `cost_sensitivity_grid.csv`: the 108-case WFF cost sweep;
- `n3_confirmation.csv`: uniform-tie JSQ gaps and recomputed composition
  reversals;
- the corresponding compact aggregate CSV files.

The N=3 reversal counts are generated directly by the current solver rather
than copied from the earlier deterministic-tie study.

## Numerical protocol

- Constant-rate uniformization with positive slack.
- Relative value iteration starts from the zero bias, normalizes at the empty
  reference state, and stops when the span of the bias increment is at most
  `1e-11`.
- Bellman action ties use tolerance `1e-10`; numerical policy optimality uses
  relative tolerance `1e-8`.
- Stationary costs are computed from invariant distributions and checked with
  an independently solved Poisson equation, flow balances, and cost
  decomposition.
- D-Z uses SciPy's HiGHS-backed MILP solver and reports the solver gap.
- R-Z uses an analytic Poisson-equation gradient, seed `20260917`, eight random
  vectors and their VM-reflected counterparts, canonical starts, and
  L-BFGS-B with at most 500 iterations per initial start.

## Figure 2

The data heatmap can be regenerated separately:

```bash
python -m pip install -r requirements-figures.txt
python generate_figures.py
```

## Main files

```text
ctmdp_routing/
  n2_solver.py                  # N=2 CTMDP and exact evaluation
  n3_solver.py                  # N=3 sparse CTMDP and reversal audit
  feeder_first.py               # UFF/WFF definitions and exact class search
  z_policy_milp.py              # certified D-Z optimization
  z_policy_randomized.py        # reflected multistart R-Z optimization
  n2_jsq_grid.py                # 160-case UJSQ experiment
  n2_feeder_first_grid.py       # UFF/WFF experiment
  n2_feeder_first_cost_sensitivity.py
  n2_rz_grid.py
  n2_observation_table.py
  n3_jsq_confirmation.py
  audit_results.py
tests/test_ctmdp.py
results/paper/
run_experiments.py
```

The code is released under the MIT License.

