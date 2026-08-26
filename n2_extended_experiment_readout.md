# N=2 CTMDP: extended experiment readout

Date: 31 July 2026  
Solver: `n2_average_cost_ctmdp.py`, version 2.0

## Experimental design

All results were computed by exact average-cost CTMDP solution and exact
stationary-policy evaluation. Simulation was not used to estimate the policy
gaps in these sweeps.

The service-balance sweep contains 160 scenarios:

- $K\in\{2,4\}$;
- $B\in\{1,2,3,4\}$;
- $\mu_{\mathrm{vm}}/(K\mu_c)\in\{0.25,0.5,1,2\}$;
- $\rho\in\{0.25,0.5,0.75,1,1.5\}$;
- $h_q=h_c=1$ and $C_B=10$.

The targeted cost sweep contains 108 scenarios with $K=B=4$:

- the same four service-capacity ratios;
- $\rho\in\{0.5,0.75,1\}$;
- $h_q/h_c\in\{0.5,1,2\}$ with $h_c=1$;
- $C_B\in\{1,10,50\}$.

The corrected script additionally reports randomized-tie JSQ, separates JSQ
errors at total-load ties from errors under strict total-load order, derives
the feasible interval for a fixed score $w_q q_i+w_n n_i$, reports feeder and
container occupancies separately, and compares optimal action sets across
adjacent parameter values.

## Numerical verification

All 268 scenarios completed successfully. Across the 160-scenario main grid:

- maximum uniformized-row sum error: $2.22\times10^{-16}$;
- maximum RVI versus stationary gain discrepancy: $3.15\times10^{-10}$;
- maximum stationary-generator residual: $6.53\times10^{-16}$;
- maximum flow-balance error: $2.67\times10^{-15}$.

The targeted cost sweep had the same qualitative numerical accuracy.

## Main findings

### 1. Feeder capacity creates an exact structural boundary

For $B=1$, JSQ was exactly optimal in all 40 service-balance scenarios, and no
composition reversal was possible at a genuine routing state.

For every tested scenario with $B\ge2$ (120 of 120), at least one aggregate
total-load class contained strict opposite optimal actions. Internal stage
composition is therefore decision-relevant as soon as more than one job can
occupy the VM-side stage.

### 2. JSQ's loss is real and is not a tie-breaking artifact

In the main grid, the largest deterministic-JSQ gap was 4.322%, at
$K=4$, $B=3$, $\mu_{\mathrm{vm}}/(K\mu_c)=0.5$, and $\rho=0.5$.

Across the grid, deterministic JSQ made 1,486 wrong decisions in states with a
strict total-load ordering, compared with 1,000 wrong decisions at total-load
ties. Equal randomization at JSQ ties did not cure the problem: it was worse
than deterministic tie-breaking in 90 scenarios, better in 30, and identical
up to numerical tolerance in the 40 $B=1$ scenarios.

Under the targeted cost sweep, the maximum JSQ gap increased to 6.561%, at
$K=B=4$, service-capacity ratio 0.5, $\rho=0.5$, $h_q/h_c=2$, and $C_B=50$.
Randomized-tie JSQ reached 6.962% in the same scenario.

### 3. The feeder-first rule is regime-dependent

The lexicographic rule that minimizes $(q_i,n_i)$ was exact throughout the
strong feeder-bottleneck regime, but it was not universal.

| Service-capacity ratio | Maximum gap, main grid | Maximum gap, cost sweep |
|---:|---:|---:|
| 0.25 | 0% | 0% |
| 0.50 | 0.0029% | 0.0036% |
| 1.00 | 0.1032% | 0.1907% |
| 2.00 | 0.3587% | 0.5599% |

Thus, feeder-first is an exact or almost exact bottleneck-regime policy and a
strong approximation elsewhere, rather than a universal optimal rule.

### 4. A fixed linear local score is sometimes insufficient

In the main grid, some positive ratio $w_q/w_n$ reproduced every optimal
action in 124 of 160 scenarios. After excluding the structurally trivial
$B=1$ cases, feasibility was:

- 30/30 scenarios at service-capacity ratio 0.25;
- 24/30 at ratio 0.5;
- 15/30 at ratio 1;
- 15/30 at ratio 2.

For $K=B=4$ in the cost sweep, a fixed linear score was feasible in all 27
ratio-0.25 cases, but in only 1 of 27 ratio-0.5 cases and none of the ratio-1
or ratio-2 cases. A service-aware or state-dependent correction is therefore
needed outside the feeder-bottleneck regime.

### 5. Load changes policy structure less than service balance

In the main grid, only 12 of 128 adjacent-$\rho$ comparisons changed any
optimal routing action set, whereas 47 of 120 adjacent service-ratio
comparisons did. In the targeted cost grid, service balance changed the policy
in 80 of 81 comparisons; blocking cost did so in 10 of 72, feeder holding cost
in only 1 of 72, and load in 22 of 72.

Performance gaps were largest at moderate load. In the main grid, the maximum
JSQ gap was 4.322% at $\rho=0.5$ and 4.254% at $\rho=0.75$, falling to 0.829%
at $\rho=1.5$.

## Interpretation for the reconstructed paper

The strongest defensible narrative is now:

> Feeder capacity determines whether aggregate-load dispatching is sufficient,
> while the relative capacities of the feeder and container stages determine
> which composition-aware structure is appropriate. JSQ is exact for a
> one-place feeder, becomes materially suboptimal for larger feeders, and a
> feeder-first correction is exact in the feeder-bottleneck regime but requires
> refinement as the downstream containers become limiting.

These results are exact finite-system evidence, not yet a theorem. The next
mathematical task is to prove the $B=1$ reduction and characterize sufficient
conditions for feeder-first optimality. The next computational task is to
validate a small number of representative cases by continuous-time simulation,
then test a rate-aware or state-dependent score on held-out loads.

## Reproduction commands

```powershell
python .\n2_average_cost_ctmdp.py --K 2 4 --B 1 2 3 4 `
  --stage-capacity-ratios 0.25 0.5 1 2 `
  --rhos 0.25 0.5 0.75 1 1.5 `
  --output-dir n2_service_balance_sensitivity
```

```powershell
python .\n2_average_cost_ctmdp.py --K 4 --B 4 `
  --stage-capacity-ratios 0.25 0.5 1 2 `
  --rhos 0.5 0.75 1 `
  --holding-vm 0.5 1 2 --holding-container 1 `
  --blocking-cost 1 10 50 `
  --output-dir n2_cost_sensitivity_K4B4
```
