# SFE Optimal Trajectory Project Memory

## Project Overview
CasADi/IPOPT-based optimal trajectory optimisation for SFE (Supercritical Fluid Extraction) model discrimination. Uses MATLAB + CasADi with CVODES integrator.

## Key Files
- `main_multi_pressure.m` — main script; parfor multistart with pressure as decision variable
- `modelSFE.m` — ODE RHS; Power_model and Linear_model variants
- `buildIntegrator.m` — wraps CasADi CVODES integrator
- `reconstruct_T_polynomial_approximation.m` — T = f(log(-ENTHALPY_RHO), PRESSURE)
- `Parameters.csv` — model parameters (k1 at rows 44-47, k2 at rows 48-53)

## State Vector Layout (Nx = 3*nstages + 2 = 152)
- [1 : nstages]            FLUID concentrations   ~0.001-0.22
- [nstages+1 : 2*nstages]  SOLID concentrations   ~7 kg/m³
- [2*nstages+1 : 3*nstages] ENTHALPY_RHO  stored SCALED /1e4  (~-15 after scaling)
- [3*nstages+1]             PRESSURE               ~100-200 bar
- [3*nstages+2]             cumulative YIELD        ~0-2

## Critical: ENTHALPY_SCALE = 1e4
ENTHALPY_RHO physical value is ~-150,000 kJ/m³. It is stored divided by 1e4 to
normalise the ODE. Two places must be consistent:
1. `modelSFE.m`: `ENTHALPY_RHO = x(...) * 1e4;` and `xdot_enthalpy / 1e4`
2. `main_multi_pressure.m`: `x0 = [...; (enthalpy_rho / 1e4) * ones(nstages,1); ...]`

## CVODES Settings (buildIntegrator.m)
- abstol = 1e-6, reltol = 1e-4
- max_num_steps = 50000   ← needed to prevent CV_TOO_MUCH_WORK in adjoint pass
- sensitivity_method = 'staggered'  ← more stable for stiff systems

## Decision Variables (normalised to [-1,1])
- zFeedTemp: 1 x N_Time (60 steps, every 10 min)
- zFeedFlow: 1 x N_Time
- zFeedPressKnots: 1 x N_P_knots (10 knots, ZOH every 60 min)

## Known Issues / Debugging Notes
- `CV_TOO_MUCH_WORK` in backward adjoint: fixed by max_num_steps=50000 + state scaling
- Pressure ~120-125 bar is near-critical CO2 regime (Tc=304.1K, Pc=73.8bar);
  EOS derivatives are largest there, causing stiff adjoint ODEs
- LHS-based pressure initial guess should be used (not hardcoded 125 bar)
- Only Y_cum (yield, state Nx) is used from the trajectory — enthalpy not extracted post-sim
