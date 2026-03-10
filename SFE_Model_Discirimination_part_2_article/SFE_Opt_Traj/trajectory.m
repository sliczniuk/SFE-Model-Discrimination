function out = trajectory(feedTemp, feedFlow, P_bar, casadi_path)
%TRAJECTORY  Forward-simulate both SFE models and compute 95% confidence intervals.
%
%  OUT = TRAJECTORY(feedTemp, feedFlow, P_bar) evaluates the Power and
%  Linear kinetic models at the given input trajectory and returns cumulative
%  yield predictions with 95% confidence interval bands.
%
%  Inputs:
%    feedTemp    — 1×N_Time feed temperature trajectory [K]
%    feedFlow    — 1×N_Time feed flow rate trajectory [m³/s]
%    P_bar       — scalar constant extraction pressure [bar]
%    casadi_path — (optional) path to CasADi install folder;
%                  omit if startup.m has already added CasADi to the path
%
%  Output struct fields:
%    out.Time     — 1×(N_Time+1) time vector [min], starts at 0
%    out.Y_P      — 1×(N_Time+1) Power model cumulative yield [g]
%    out.Y_L      — 1×(N_Time+1) Linear model cumulative yield [g]
%    out.CI_P     — 1×(N_Time+1) 95% CI half-width, Power model [g]
%    out.CI_L     — 1×(N_Time+1) 95% CI half-width, Linear model [g]
%    out.feedTemp — echo of input [K]
%    out.feedFlow — echo of input [m³/s]
%    out.P_bar    — echo of input [bar]
%
%  The confidence intervals account for both parameter estimation uncertainty
%  (propagated via first-order sensitivity) and measurement noise:
%    Var(y(t)) = diag( J(t) * Cov_theta * J(t)' ) + sigma2
%    CI(t)     = 1.96 * sqrt( Var(y(t)) )

%% CasADi path
if nargin >= 4 && ~isempty(casadi_path)
    addpath(casadi_path);
end
import casadi.*

%% Time grid (inferred from trajectory length)
feedTemp  = reshape(feedTemp, 1, []);
feedFlow  = reshape(feedFlow, 1, []);
N_Time    = numel(feedTemp);
finalTime = 600;                      % fixed extraction duration [min]
timeStep  = finalTime / N_Time;       % e.g. 15 if N_Time=40

%% Load static problem data (Parameters.csv, covariances, bed geometry)
config.timeStep       = timeStep;
config.finalTime      = finalTime;
config.T_min          = 303;      config.T_max = 313;      % [K]
config.F_min          = 3.3e-5;   config.F_max = 6.7e-5;  % [m³/s]
config.feedPressValue = P_bar;
config.sample_tol     = 1e-3;
static = load_opt_traj_static_data(config);

%% Symbolic CasADi inputs
fT  = MX.sym('fT',  1, N_Time);   % feed temperature [K]
fF  = MX.sym('fF',  1, N_Time);   % feed flow rate [m³/s]
k1s = MX.sym('k1',  4, 1);        % Power model parameters
k2s = MX.sym('k2',  6, 1);        % Linear model parameters
ks  = [k1s; k2s];

%% Symbolic initial state (follows build_opt_traj_problem.m pattern)
T_sym        = fT(1);
P_num        = P_bar;
Z            = Compressibility(T_sym, P_num, static.Parameters);
rho          = rhoPB_Comp(T_sym, P_num, Z, static.Parameters);
enthalpy_rho = rho * SpecificEnthalpy(T_sym, P_num, Z, rho, static.Parameters);
x0 = [static.C0fluid';
      static.C0solid * static.bed_mask;
      enthalpy_rho * ones(static.nstages, 1);
      P_num; 0];

%% Control and parameter matrix
feedPress      = P_bar * ones(1, N_Time);              % constant pressure [bar]
uu             = [fT', feedPress', fF'];               % N_Time × 3
Parameters_sym = MX(cell2mat(static.Parameters));
Parameters_sym(static.which_k) = ks(1:numel(static.which_k));
U_base = [uu'; repmat(Parameters_sym, 1, N_Time)];    % Nu × N_Time

%% Build integrators and roll out full trajectory via mapaccum
f_power  = @(x,u) modelSFE(x, u, static.bed_mask, static.timeStep_in_sec, ...
    'Power_model',  static.epsi_mask, static.one_minus_epsi_mask);
f_linear = @(x,u) modelSFE(x, u, static.bed_mask, static.timeStep_in_sec, ...
    'Linear_model', static.epsi_mask, static.one_minus_epsi_mask);

F_power        = buildIntegrator(f_power,  [static.Nx, static.Nu], static.timeStep_in_sec, 'cvodes');
F_linear       = buildIntegrator(f_linear, [static.Nx, static.Nu], static.timeStep_in_sec, 'cvodes');
F_accum_power  = F_power.mapaccum( 'F_accum_power',  N_Time);
F_accum_linear = F_linear.mapaccum('F_accum_linear', N_Time);

X_power  = F_accum_power( x0, U_base);   % Nx × N_Time
X_linear = F_accum_linear(x0, U_base);   % Nx × N_Time

% Prepend t=0 (yield=0) to get N_Time+1 points
Y_full_P = [MX(0), X_power( static.Nx, :)];   % 1 × (N_Time+1)
Y_full_L = [MX(0), X_linear(static.Nx, :)];   % 1 × (N_Time+1)

%% Jacobians of full yield trajectory w.r.t. model parameters
J_full_P = jacobian(Y_full_P, k1s);   % (N_Time+1) × 4
J_full_L = jacobian(Y_full_L, k2s);   % (N_Time+1) × 6

%% Compile evaluation Function and evaluate numerically
eval_fn = Function('traj_eval', ...
    {fT, fF, k1s, k2s}, ...
    {Y_full_P, Y_full_L, J_full_P, J_full_L});

[Y_P_cas, Y_L_cas, JJ_P_cas, JJ_L_cas] = eval_fn( ...
    feedTemp, feedFlow, static.k1_val(:), static.k2_val(:));

Y_P  = full(Y_P_cas);    % 1 × (N_Time+1)
Y_L  = full(Y_L_cas);
JJ_P = full(JJ_P_cas);   % (N_Time+1) × 4
JJ_L = full(JJ_L_cas);   % (N_Time+1) × 6

%% 95% confidence intervals
% Var(y(t)) = diag( J(t)*Cov*J(t)' ) + sigma2
% Efficient diagonal extraction: sum((J*Cov).*J, 2)
sigma2 = static.sigma2_cases(1);   % cumulative yield measurement variance
z_95   = 1.96;

var_P = sum((JJ_P * static.Cov_power_cum)  .* JJ_P, 2)' + sigma2;   % 1 × (N_Time+1)
var_L = sum((JJ_L * static.Cov_linear_cum) .* JJ_L, 2)' + sigma2;

CI_P = z_95 * sqrt(max(var_P, 0));   % 1 × (N_Time+1)
CI_L = z_95 * sqrt(max(var_L, 0));

%% Assemble output
out.Time     = static.Time;    % 1 × (N_Time+1) [min]
out.Y_P      = Y_P;
out.Y_L      = Y_L;
out.CI_P     = CI_P;
out.CI_L     = CI_L;
out.feedTemp = feedTemp;
out.feedFlow = feedFlow;
out.P_bar    = P_bar;
end
