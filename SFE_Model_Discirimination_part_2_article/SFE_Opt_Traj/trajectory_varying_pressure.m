function out = trajectory_varying_pressure(feedTemp, feedFlow, feedPress, N_mc)
%TRAJECTORY_VARYING_PRESSURE  Forward-simulate both SFE models (thermal lag)
%  with time-varying pressure and compute uncertainty via Monte Carlo.
%
%  OUT = TRAJECTORY_VARYING_PRESSURE(feedTemp, feedFlow, feedPress, N_mc)
%
%  Inputs:
%    feedTemp  — 1×N_Time feed temperature trajectory [K]
%    feedFlow  — 1×N_Time feed flow rate trajectory   [m³/s]
%    feedPress — 1×N_Time feed pressure trajectory     [bar]
%    N_mc      — number of Monte Carlo samples (default 500)
%
%  Output struct fields:
%    out.Time       — 1×(N_Time+1) time vector [min]
%    out.Y_P_nom    — 1×(N_Time+1) Power  model nominal yield [g]
%    out.Y_L_nom    — 1×(N_Time+1) Linear model nominal yield [g]
%    out.Y_P_mc     — N_mc×(N_Time+1) MC yield samples, Power  model
%    out.Y_L_mc     — N_mc×(N_Time+1) MC yield samples, Linear model
%    out.Y_P_obs_mc — N_mc×(N_Time+1) MC observed samples (model + noise)
%    out.Y_L_obs_mc — N_mc×(N_Time+1) MC observed samples (model + noise)

if nargin < 4 || isempty(N_mc)
    N_mc = 500;
end

import casadi.*

feedTemp  = reshape(feedTemp,  1, []);
feedFlow  = reshape(feedFlow,  1, []);
feedPress = reshape(feedPress, 1, []);
N_Time    = numel(feedTemp);
finalTime = 600;
timeStep  = finalTime / N_Time;

%% Load static data
config.timeStep       = timeStep;
config.finalTime      = finalTime;
config.T_min          = 303;      config.T_max = 313;
config.F_min          = 3.3e-5;   config.F_max = 6.7e-5;
config.feedPressValue = feedPress(1);
config.sample_tol     = 1e-3;
static = load_opt_traj_static_data(config);

%% Symbolic inputs
fT  = MX.sym('fT', 1, N_Time);
fF  = MX.sym('fF', 1, N_Time);
fP  = MX.sym('fP', 1, N_Time);
k1s = MX.sym('k1', 4, 1);
k2s = MX.sym('k2', 6, 1);
ks  = [k1s; k2s];

%% Initial state (depends on first T and P)
T_sym        = fT(1);
P_sym        = fP(1);
Z            = Compressibility(T_sym, P_sym, static.Parameters);
rho          = rhoPB_Comp(T_sym, P_sym, Z, static.Parameters);
enthalpy_rho = rho * SpecificEnthalpy(T_sym, P_sym, Z, rho, static.Parameters);
x0 = [static.C0fluid';
      static.C0solid * static.bed_mask;
      (enthalpy_rho / 1e4) * ones(static.nstages, 1);
      P_sym; 0];

%% Control and parameter matrix
uu             = [fT', fP', fF'];
Parameters_sym = MX(cell2mat(static.Parameters));
Parameters_sym(static.which_k) = ks(1:numel(static.which_k));
U_base = [uu'; repmat(Parameters_sym, 1, N_Time)];

%% Build integrators (thermal lag model)
f_power  = @(x, u) modelSFE_thermal_lag(x, u, static.bed_mask, ...
    static.timeStep_in_sec, 'Power_model', ...
    static.epsi_mask, static.one_minus_epsi_mask);
f_linear = @(x, u) modelSFE_thermal_lag(x, u, static.bed_mask, ...
    static.timeStep_in_sec, 'Linear_model', ...
    static.epsi_mask, static.one_minus_epsi_mask);

F_power        = buildIntegrator(f_power,  [static.Nx, static.Nu], static.timeStep_in_sec, 'cvodes');
F_linear       = buildIntegrator(f_linear, [static.Nx, static.Nu], static.timeStep_in_sec, 'cvodes');
F_accum_power  = F_power.mapaccum( 'F_accum_power',  N_Time);
F_accum_linear = F_linear.mapaccum('F_accum_linear', N_Time);

X_power  = F_accum_power( x0, U_base);
X_linear = F_accum_linear(x0, U_base);

%% Full yield trajectory (prepend t=0)
Y_full_P = [MX(0), X_power( static.Nx, :)];
Y_full_L = [MX(0), X_linear(static.Nx, :)];

%% Compile CasADi function (built once, called N_mc times)
eval_fn = Function('traj_eval', ...
    {fT, fF, fP, k1s, k2s}, ...
    {Y_full_P, Y_full_L});

%% Nominal evaluation
[Y_P_nom_cas, Y_L_nom_cas] = eval_fn( ...
    feedTemp, feedFlow, feedPress, ...
    static.k1_val(:), static.k2_val(:));
Y_P_nom = full(Y_P_nom_cas);
Y_L_nom = full(Y_L_nom_cas);

%% Monte Carlo: sample parameters from their posterior distributions
rng(42);
k1_samples = mvnrnd(static.k1_val(:)', static.Cov_power_cum,  N_mc);   % N_mc × 4
k2_samples = mvnrnd(static.k2_val(:)', static.Cov_linear_cum, N_mc);   % N_mc × 6

sigma2     = static.sigma2_cases(1);
sigma_meas = sqrt(sigma2);

N_out    = N_Time + 1;
Y_P_mc     = zeros(N_mc, N_out);
Y_L_mc     = zeros(N_mc, N_out);
Y_P_obs_mc = zeros(N_mc, N_out);
Y_L_obs_mc = zeros(N_mc, N_out);

fprintf('Running %d Monte Carlo samples...\n', N_mc);
for m = 1:N_mc
    [yP, yL] = eval_fn(feedTemp, feedFlow, feedPress, ...
        k1_samples(m,:)', k2_samples(m,:)');
    Y_P_mc(m, :) = full(yP);
    Y_L_mc(m, :) = full(yL);

    % Add measurement noise
    Y_P_obs_mc(m, :) = Y_P_mc(m, :) + sigma_meas * randn(1, N_out);
    Y_L_obs_mc(m, :) = Y_L_mc(m, :) + sigma_meas * randn(1, N_out);

    if mod(m, 100) == 0
        fprintf('  %d / %d done\n', m, N_mc);
    end
end
fprintf('Monte Carlo complete.\n');

%% Output
out.Time       = static.Time;
out.Y_P_nom    = Y_P_nom;
out.Y_L_nom    = Y_L_nom;
out.Y_P_mc     = Y_P_mc;
out.Y_L_mc     = Y_L_mc;
out.Y_P_obs_mc = Y_P_obs_mc;
out.Y_L_obs_mc = Y_L_obs_mc;
end
