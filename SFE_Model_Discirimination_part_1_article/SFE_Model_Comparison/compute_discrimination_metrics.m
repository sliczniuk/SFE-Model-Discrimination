function [max_KS, integrated_JS, Time_ks_max, results] = compute_discrimination_metrics(T, P, F, timeStep, finalTime, varargin)
% COMPUTE_DISCRIMINATION_METRICS Compute model discrimination metrics via Monte Carlo
%
% Evaluates discrimination between Power and Linear SFE extraction models
% at a fixed operating point (T, P, F) while propagating kinetic parameter
% uncertainty through Monte Carlo sampling.
%
% Two paired MC loops are run per call:
%   Loop 1 (cumulative) : parameters sampled with Cov_XXX_cum
%                         → Y_power_valid / Y_linear_valid [g]
%                         → empirical noise added (sigma_empirical_XXX_cum)
%                         → cumulative yield metrics on noise-inflated samples
%   Loop 2 (differential): parameters sampled with Cov_XXX_diff
%                         → Rate_power_valid / Rate_linear_valid [g/min]
%                         → empirical noise added (sigma_empirical_XXX_diff)
%                         → extraction rate metrics on noise-inflated samples
%
% Empirical sigmas are hardcoded from parameter estimation residuals:
%   Cumulative fit: Power  sigma = 0.1609 [g],  Linear sigma = 0.1521 [g]
%   Differential fit: Power sigma = 0.03932 [g/min], Linear sigma = 0.03503 [g/min]
%
% Physical bounds are enforced after noise inflation:
%   Y(t)   clamped to [0, m_total=3g]
%   dY/dt  clamped to [0, Inf)
%
% Syntax:
%   [max_KS, integrated_JS] = compute_discrimination_metrics(T, P, F, timeStep, finalTime)
%   [max_KS, integrated_JS, Time_ks_max, results] = compute_discrimination_metrics(...)
%   [...] = compute_discrimination_metrics(..., 'Name', Value, ...)
%
% Inputs:
%   T         - Temperature [K]
%   P         - Pressure [bar]
%   F         - Flow rate [m3/s]
%   timeStep  - Time step [minutes]
%   finalTime - Total extraction time [minutes]
%
% Optional Name-Value Pairs:
%   'N_MC'    - Number of Monte Carlo samples per loop (default: 500)
%   'Seed'    - Random seed for reproducibility (default: 42)
%   'Verbose' - Print progress messages (default: true)
%   'UseMap'  - Use CasADi map over samples (default: true)
%   'SigmaCumPower'   - Measurement noise std, Power cumulative [g] (default: 0.1609)
%   'SigmaCumLinear'  - Measurement noise std, Linear cumulative [g] (default: 0.1521)
%   'SigmaDiffPower'  - Measurement noise std, Power differential [g/min] (default: 0.03932)
%   'SigmaDiffLinear' - Measurement noise std, Linear differential [g/min] (default: 0.03503)
%                       Defaults are empirical sigmas from parameter estimation residuals.
%                       Set any to 0 to disable noise inflation for that signal/model.
%   'RateSamplingTime'- Sampling window for rate computation [min] (default: timeStep).
%                       Rate = ΔY(dt_rate) / dt_rate [g/min]. Must be a positive multiple
%                       of timeStep. Larger dt_rate improves SNR by growing the signal
%                       (mass increment) while the noise floor (sigma_diff) stays fixed.
%                       Example: timeStep=5, RateSamplingTime=25 → stride=5, rates over
%                       25-min cumulative increments with 25-min temporal resolution.
%   'N_noiseCI'       - Number of independent noise draws for metric CI bands (default: 30).
%                       Each draw re-samples N(0,sigma^2) noise on the fixed clean MC
%                       trajectories (Y_power_valid / Rate_power_valid) and recomputes all
%                       four discrimination metrics (JS, KS, AUC, KL). The 2.5th/97.5th
%                       percentiles across draws give [2 x n_time] CI fields in metrics.
%                       Compute cost ≈ N_noiseCI × n_time × AUC_cost ≈ 1–3 min at N_MC=1500.
%                       Set to 0 to skip CI computation entirely (ci95_* fields absent).
%
% Outputs:
%   max_KS        - Maximum KS statistic over cumulative yield trajectory
%   integrated_JS - JS divergence integrated over cumulative yield trajectory
%   Time_ks_max   - Time at which max KS occurs [min]
%   results       - Struct with full results (see fields below)
%
% results fields:
%   .sigma_empirical_power_cum   - Empirical noise, Power cumulative [g]
%   .sigma_empirical_linear_cum  - Empirical noise, Linear cumulative [g]
%   .sigma_empirical_power_diff  - Empirical noise, Power differential [g/min]
%   .sigma_empirical_linear_diff - Empirical noise, Linear differential [g/min]
%   .Time                 - Time vector for cumulative signal [min]
%   .Y_power_valid        - Power cumulative trajectories, clean [n_valid_cum x N_Time+1]
%   .Y_linear_valid       - Linear cumulative trajectories, clean
%   .Y_power_nom          - Nominal Power cumulative trajectory
%   .Y_linear_nom         - Nominal Linear cumulative trajectory
%   .dt_rate              - Effective rate sampling window [min] (= stride * timeStep)
%   .stride               - Subsampling stride used for rate computation
%   .Time_rate            - Time vector for rate signal [min] (midpoints of dt_rate windows)
%   .Rate_power_valid     - Power rate trajectories, clean [n_valid_diff x n_time_rate]
%   .Rate_linear_valid    - Linear rate trajectories, clean
%   .Rate_power_nom       - Nominal Power rate trajectory
%   .Rate_linear_nom      - Nominal Linear rate trajectory
%   .metrics              - Struct with all time-wise and integrated metrics
%                           (computed on noise-inflated samples)
%                           When N_noiseCI > 0, also contains CI fields:
%                             .ci95_js_cum / .ci95_ks_cum / .ci95_auc_cum   [2 x n_time_cum]
%                             .ci95_kl_pl_cum / .ci95_kl_lp_cum             [2 x n_time_cum]
%                             .ci95_js_rate / .ci95_ks_rate / .ci95_auc_rate [2 x n_time_rate]
%                             .ci95_kl_pl_rate / .ci95_kl_lp_rate           [2 x n_time_rate]
%                           Row 1 = 2.5th percentile, Row 2 = 97.5th percentile
%                           across N_noiseCI independent noise draws.
%   .n_valid_cum          - Valid samples from cumulative loop
%   .n_valid_diff         - Valid samples from differential loop

%% Parse inputs
p = inputParser;
addRequired(p, 'T', @isnumeric);
addRequired(p, 'P', @isnumeric);
addRequired(p, 'F', @isnumeric);
addRequired(p, 'timeStep', @isnumeric);
addRequired(p, 'finalTime', @isnumeric);
addParameter(p, 'N_MC', 500, @isnumeric);
addParameter(p, 'Seed', 42, @isnumeric);
addParameter(p, 'Verbose', true, @islogical);
addParameter(p, 'UseMap', true, @islogical);
addParameter(p, 'SigmaCumPower',   1.6090e-01, @isnumeric);  % [g]
addParameter(p, 'SigmaCumLinear',  1.5209e-01, @isnumeric);  % [g]
addParameter(p, 'SigmaDiffPower',  3.9319e-02, @isnumeric);  % [g/min]
addParameter(p, 'SigmaDiffLinear', 3.5033e-02, @isnumeric);  % [g/min]
addParameter(p, 'RateSamplingTime', [], @(x) isempty(x) || (isnumeric(x) && x > 0));
addParameter(p, 'N_noiseCI', 30, @(x) isnumeric(x) && x >= 0 && x == round(x));
parse(p, T, P, F, timeStep, finalTime, varargin{:});

N_MC    = p.Results.N_MC;
verbose = p.Results.Verbose;
use_map = p.Results.UseMap;

rng(p.Results.Seed);

import casadi.*

if verbose
    fprintf('Computing discrimination metrics at T=%.0fK, P=%.0f bar, F=%.2e m3/s\n', T, P, F);
    fprintf('  MC samples per loop: %d\n', N_MC);
end

%% Measurement noise sigmas (from inputParser — default = empirical residuals)
% Override via 'SigmaCumPower', 'SigmaCumLinear', 'SigmaDiffPower', 'SigmaDiffLinear'
% Set to 0 to use pure parameter uncertainty without any noise inflation.
sigma_empirical_power_cum   = p.Results.SigmaCumPower;    % [g]
sigma_empirical_linear_cum  = p.Results.SigmaCumLinear;   % [g]
sigma_empirical_power_diff  = p.Results.SigmaDiffPower;   % [g/min]
sigma_empirical_linear_diff = p.Results.SigmaDiffLinear;  % [g/min]

if verbose
    fprintf('  sigma_empirical cumulative:   Power=%.4f [g],  Linear=%.4f [g]\n', ...
        sigma_empirical_power_cum, sigma_empirical_linear_cum);
    fprintf('  sigma_empirical differential: Power=%.4f [g/min], Linear=%.4f [g/min]\n', ...
        sigma_empirical_power_diff, sigma_empirical_linear_diff);
end

%% Covariance matrices
% Nominal parameter values
theta_power  = [1.222524; 4.308414; 0.972739; 3.428618];         % [k_w0, a_w, b_w, n_k]
theta_linear = [0.19; -8.188; 0.62; 3.158; 11.922; -0.6868];    % [D_i(0), D_i(Re), D_i(F), Ups(0), Ups(Re), Ups(F)]

% Power model (4x4) — cumulative fit.  Empirical sigma: 0.1609 [g]
Cov_power_cum = [
    1.0035e-02,  1.1795e-02,  1.8268e-03,  2.5611e-02;
    1.1795e-02,  5.6469e-02,  3.1182e-03,  2.8266e-02;
    1.8268e-03,  3.1182e-03,  5.7241e-03,  6.4459e-03;
    2.5611e-02,  2.8266e-02,  6.4459e-03,  7.1744e-02
];

% Power model (4x4) — differential fit.  Empirical sigma: 0.03932 [g/min]
Cov_power_diff = [
    3.2963e-03,  1.2094e-03, -2.5042e-03,  6.8414e-03;
    1.2094e-03,  1.0981e-01, -5.7125e-04,  2.3381e-03;
   -2.5042e-03, -5.7125e-04,  1.3915e-02, -2.7301e-04;
    6.8414e-03,  2.3381e-03, -2.7301e-04,  3.8686e-02
];

% Linear model (6x6) — cumulative fit.  Empirical sigma: 0.1521 [g]
Cov_linear_cum = [
    2.7801e-02,  3.5096e-02, -6.9596e-03,  7.1573e-02,  1.0992e-02, -1.2661e-02;
    3.5096e-02,  6.8482e-01, -5.0531e-02, -4.8187e-02,  3.9209e-01, -2.6206e-02;
   -6.9596e-03, -5.0531e-02,  4.5693e-03, -7.7054e-03, -1.5915e-02,  3.3012e-03;
    7.1573e-02, -4.8187e-02, -7.7054e-03,  2.9254e-01,  6.5758e-02, -4.6300e-02;
    1.0992e-02,  3.9209e-01, -1.5915e-02,  6.5758e-02,  2.9506e+00, -1.3133e-01;
   -1.2661e-02, -2.6206e-02,  3.3012e-03, -4.6300e-02, -1.3133e-01,  1.2975e-02
];

% Linear model (6x6) — differential fit.  Empirical sigma: 0.03503 [g/min]
Cov_linear_diff = [
    2.2178e-02,  1.0828e-02, -4.3832e-03,  4.3992e-02,  4.4695e-03, -7.4634e-03;
    1.0828e-02,  4.3513e-01, -2.4832e-02,  3.0423e-03,  6.7633e-01, -3.3289e-02;
   -4.3832e-03, -2.4832e-02,  2.1282e-03, -7.3766e-03, -3.2298e-02,  2.8884e-03;
    4.3992e-02,  3.0423e-03, -7.3766e-03,  3.1429e-01, -6.2258e-02, -4.7085e-02;
    4.4695e-03,  6.7633e-01, -3.2298e-02, -6.2258e-02,  6.1032e+00, -2.4975e-01;
   -7.4634e-03, -3.3289e-02,  2.8884e-03, -4.7085e-02, -2.4975e-01,  1.8474e-02
];

%% Setup simulation infrastructure
Parameters_table = readtable('Parameters.csv');
Parameters = num2cell(Parameters_table{:,3});

m_total = 3.0;  % [g] maximum possible cumulative yield
before  = 0.04;
bed     = 0.92;

Time_in_sec     = (timeStep:timeStep:finalTime) * 60;
Time            = [0, Time_in_sec/60];
N_Time          = length(Time_in_sec);
timeStep_in_sec = timeStep * 60;
dt_min          = timeStep;

% Rate sampling window: dt_rate [min], stride = dt_rate / timeStep
if isempty(p.Results.RateSamplingTime)
    dt_rate = dt_min;   % default: same as timeStep (stride = 1, unchanged behaviour)
else
    dt_rate = p.Results.RateSamplingTime;
end
stride  = max(1, round(dt_rate / dt_min));
dt_rate = stride * dt_min;   % snap to nearest exact multiple of timeStep

nstages = Parameters{1};
r       = Parameters{3};
epsi    = Parameters{4};
L       = Parameters{6};

nstagesbefore = 1:floor(before * nstages);
nstagesbed    = nstagesbefore(end)+1 : nstagesbefore(end) + floor(bed * nstages);
nstagesafter  = nstagesbed(end)+1 : nstages;

bed_mask = nan(nstages, 1);
bed_mask(nstagesbefore) = 0;
bed_mask(nstagesbed)    = 1;
bed_mask(nstagesafter)  = 0;

V_slice = (L/nstages) * pi * r^2;
V_bed   = V_slice * numel(nstagesbed);

V_before_fluid = repmat(V_slice * numel(nstagesbefore) / numel(nstagesbefore), numel(nstagesbefore), 1);
V_bed_fluid    = repmat(V_bed * (1-epsi) / numel(nstagesbed), numel(nstagesbed), 1);
V_after_fluid  = repmat(V_slice * numel(nstagesafter) / numel(nstagesafter), numel(nstagesafter), 1);
V_fluid        = [V_before_fluid; V_bed_fluid; V_after_fluid];

C0solid        = m_total * 1e-3 / (V_bed * epsi);
Parameters{2}  = C0solid;

m_fluid  = zeros(1, nstages);
C0fluid  = m_fluid * 1e-3 ./ V_fluid';

epsi_mask           = epsi .* bed_mask;
one_minus_epsi_mask = 1 - epsi_mask;

Nx = 3 * nstages + 2;
Nu = 3 + numel(Parameters);

% Fluid properties and initial state
Z            = Compressibility(T, P, Parameters);
rho          = rhoPB_Comp(T, P, Z, Parameters);
enthalpy_rho = rho * SpecificEnthalpy(T, P, Z, rho, Parameters);

x0 = [C0fluid'; C0solid * bed_mask; enthalpy_rho * ones(nstages, 1); P; 0];

feedTemp  = T * ones(1, N_Time);
feedPress = P * ones(1, N_Time);
feedFlow  = F * ones(1, N_Time);
uu        = [feedTemp', feedPress', feedFlow'];
U_base    = [uu'; repmat(cell2mat(Parameters), 1, N_Time)];

%% Nominal simulations
if verbose
    fprintf('  Running nominal simulations...\n');
end

f_power_nom = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, ...
    'Power_model', epsi_mask, one_minus_epsi_mask);
F_power_nom = buildIntegrator(f_power_nom, [Nx, Nu], timeStep_in_sec, 'cvodes');
F_accum_power_nom = F_power_nom.mapaccum('F_accum_power', N_Time);

f_linear_nom = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, ...
    'Linear_model', epsi_mask, one_minus_epsi_mask);
F_linear_nom = buildIntegrator(f_linear_nom, [Nx, Nu], timeStep_in_sec, 'cvodes');
F_accum_linear_nom = F_linear_nom.mapaccum('F_accum_linear', N_Time);

X_power_nom  = F_accum_power_nom(x0, U_base);
X_linear_nom = F_accum_linear_nom(x0, U_base);

Y_power_nom  = full([0, X_power_nom(Nx, :)]);
Y_linear_nom = full([0, X_linear_nom(Nx, :)]);

% Nominal rates — subsampled to coarse dt_rate window
Y_power_nom_coarse  = Y_power_nom(1:stride:end);
Y_linear_nom_coarse = Y_linear_nom(1:stride:end);
Rate_power_nom      = diff(Y_power_nom_coarse);    % mass increment [g] over dt_rate window
Rate_linear_nom     = diff(Y_linear_nom_coarse);
Time_rate           = (0:stride:N_Time-stride) * dt_min + dt_rate/2;
n_time_rate         = length(Time_rate);

%% Build MC integrators (shared by both loops)
n_param   = numel(Parameters);
Nu_power  = 3 + n_param + 4;
Nu_linear = 3 + n_param + 6;

f_power_mc = @(x, u) modelSFE_power_with_params(x, u, bed_mask, timeStep_in_sec, ...
    epsi_mask, one_minus_epsi_mask, n_param);
f_linear_mc = @(x, u) modelSFE_linear_with_params(x, u, bed_mask, timeStep_in_sec, ...
    epsi_mask, one_minus_epsi_mask, n_param);

F_power_mc  = buildIntegrator(f_power_mc,  [Nx, Nu_power],  timeStep_in_sec, 'cvodes');
F_linear_mc = buildIntegrator(f_linear_mc, [Nx, Nu_linear], timeStep_in_sec, 'cvodes');

if use_map
    F_power_map  = F_power_mc.map(N_MC);
    F_linear_map = F_linear_mc.map(N_MC);
end

%% Helper: run one MC loop given parameter samples
% Returns cell {Y_p, Y_l} each [N_MC x N_Time+1], NaN for failed trajectories
    function Y_MC = run_mc_loop(theta_power_s, theta_linear_s, label)
        if verbose
            fprintf('  Running MC loop (%s)...\n', label);
        end
        Y_p = NaN(N_MC, N_Time + 1);
        Y_l = NaN(N_MC, N_Time + 1);
        Y_p(:, 1) = 0;
        Y_l(:, 1) = 0;

        Xp = repmat(x0, 1, N_MC);
        Xl = repmat(x0, 1, N_MC);

        tic;
        if use_map
            for t = 1:N_Time
                u_t    = U_base(:, t);
                Up_t   = [repmat(u_t, 1, N_MC); theta_power_s];
                Ul_t   = [repmat(u_t, 1, N_MC); theta_linear_s];
                Xp     = F_power_map(Xp, Up_t);
                Xl     = F_linear_map(Xl, Ul_t);
                Y_p(:, t+1) = full(Xp(Nx, :)).';
                Y_l(:, t+1) = full(Xl(Nx, :)).';
            end
        else
            vp = true(N_MC, 1);
            vl = true(N_MC, 1);
            for t = 1:N_Time
                u_t = U_base(:, t);
                for i = 1:N_MC
                    if vp(i)
                        try
                            Xp(:,i) = full(F_power_mc(Xp(:,i), [u_t; theta_power_s(:,i)]));
                        catch
                            Xp(:,i) = NaN; vp(i) = false;
                        end
                    end
                    if vl(i)
                        try
                            Xl(:,i) = full(F_linear_mc(Xl(:,i), [u_t; theta_linear_s(:,i)]));
                        catch
                            Xl(:,i) = NaN; vl(i) = false;
                        end
                    end
                end
                Y_p(:, t+1) = Xp(Nx, :).';
                Y_l(:, t+1) = Xl(Nx, :).';
            end
        end
        if verbose
            fprintf('  MC loop (%s) completed in %.1f s\n', label, toc);
        end
        Y_MC = {Y_p, Y_l};
    end

%% Loop 1: cumulative covariance → Y_power_valid / Y_linear_valid
rng(p.Results.Seed);
L_pow_cum = chol(Cov_power_cum  + 1e-10*eye(4), 'lower');
L_lin_cum = chol(Cov_linear_cum + 1e-10*eye(6), 'lower');

theta_power_cum  = repmat(theta_power,  1, N_MC) + L_pow_cum * randn(4, N_MC);
theta_linear_cum = repmat(theta_linear, 1, N_MC) + L_lin_cum * randn(6, N_MC);

out_cum = run_mc_loop(theta_power_cum, theta_linear_cum, 'cumulative');
Y_power_MC_cum  = out_cum{1};
Y_linear_MC_cum = out_cum{2};

valid_p_cum  = all(isfinite(Y_power_MC_cum)  & Y_power_MC_cum  >= 0 & Y_power_MC_cum  <= 10, 2);
valid_l_cum  = all(isfinite(Y_linear_MC_cum) & Y_linear_MC_cum >= 0 & Y_linear_MC_cum <= 10, 2);
valid_cum    = valid_p_cum & valid_l_cum;
Y_power_valid  = Y_power_MC_cum(valid_cum, :);   % clean trajectories [n_valid_cum x N_Time+1]
Y_linear_valid = Y_linear_MC_cum(valid_cum, :);
n_valid_cum    = sum(valid_cum);

if verbose
    fprintf('  Valid samples (cumulative): %d/%d\n', n_valid_cum, N_MC);
end

% Add empirical noise to cumulative trajectories for metric computation
% Y_obs = Y_model(theta) + epsilon, epsilon ~ N(0, sigma^2)
% Lower bound: clamp at 0 (balance cannot read negative mass)
% Upper bound: unbounded — measurement noise can push reading above m_total
rng(p.Results.Seed + 10);  % separate seed for noise realisation
Y_power_obs  = max(0, Y_power_valid  + sigma_empirical_power_cum  * randn(size(Y_power_valid)));
Y_linear_obs = max(0, Y_linear_valid + sigma_empirical_linear_cum * randn(size(Y_linear_valid)));

%% Loop 2: differential covariance → Rate_power_valid / Rate_linear_valid
rng(p.Results.Seed + 1);   % different seed so samples are independent of loop 1
L_pow_diff = chol(Cov_power_diff  + 1e-10*eye(4), 'lower');
L_lin_diff = chol(Cov_linear_diff + 1e-10*eye(6), 'lower');

theta_power_diff  = repmat(theta_power,  1, N_MC) + L_pow_diff * randn(4, N_MC);
theta_linear_diff = repmat(theta_linear, 1, N_MC) + L_lin_diff * randn(6, N_MC);

out_diff = run_mc_loop(theta_power_diff, theta_linear_diff, 'differential');
Y_power_MC_diff  = out_diff{1};
Y_linear_MC_diff = out_diff{2};

valid_p_diff = all(isfinite(Y_power_MC_diff)  & Y_power_MC_diff  >= 0 & Y_power_MC_diff  <= 10, 2);
valid_l_diff = all(isfinite(Y_linear_MC_diff) & Y_linear_MC_diff >= 0 & Y_linear_MC_diff <= 10, 2);
valid_diff   = valid_p_diff & valid_l_diff;

% Clean rate trajectories from diff-covariance trajectories — coarse dt_rate window
Y_p_diff_valid = Y_power_MC_diff(valid_diff, :);    % [n_valid_diff x N_Time+1]
Y_l_diff_valid = Y_linear_MC_diff(valid_diff, :);
% Subsample cumulative trajectory at every stride-th column (keep t=0)
Y_p_coarse = Y_p_diff_valid(:, 1:stride:end);       % [n_valid_diff x N_coarse+1]
Y_l_coarse = Y_l_diff_valid(:, 1:stride:end);
Rate_power_valid  = diff(Y_p_coarse, 1, 2);    % mass increment [g] over dt_rate window
Rate_linear_valid = diff(Y_l_coarse, 1, 2);
n_valid_diff      = sum(valid_diff);

if verbose
    fprintf('  Valid samples (differential): %d/%d\n', n_valid_diff, N_MC);
end

% Add empirical measurement noise to mass increment trajectories for metric computation
% sigma_empirical_XXX_diff is in [g] (balance reading noise, independent of dt_rate)
% Clamp to [0, Inf) — mass increment cannot be negative
rng(p.Results.Seed + 11);  % separate seed for noise realisation
Rate_power_obs  = max(0, Rate_power_valid  + sigma_empirical_power_diff  * randn(size(Rate_power_valid)));
Rate_linear_obs = max(0, Rate_linear_valid + sigma_empirical_linear_diff * randn(size(Rate_linear_valid)));

%% Cumulative yield metrics  (noise-inflated Y_obs, skip t=0)
if verbose
    fprintf('  Computing cumulative yield metrics...\n');
end

n_time_cum = length(Time);
metrics = struct();
metrics.Time                        = Time;
metrics.T0                          = T;
metrics.P0                          = P;
metrics.F0                          = F;
metrics.sigma_empirical_power_cum   = sigma_empirical_power_cum;
metrics.sigma_empirical_linear_cum  = sigma_empirical_linear_cum;
metrics.sigma_empirical_power_diff  = sigma_empirical_power_diff;
metrics.sigma_empirical_linear_diff = sigma_empirical_linear_diff;

% Preallocate cumulative metrics
metrics.kl_power_linear = zeros(1, n_time_cum);
metrics.kl_linear_power = zeros(1, n_time_cum);
metrics.js_divergence   = zeros(1, n_time_cum);
metrics.ks_stat         = zeros(1, n_time_cum);
metrics.ks_pval         = zeros(1, n_time_cum);
metrics.mean_diff       = zeros(1, n_time_cum);
metrics.overlap_coef    = zeros(1, n_time_cum);
metrics.auc             = zeros(1, n_time_cum);
metrics.mean_power      = zeros(1, n_time_cum);
metrics.mean_linear     = zeros(1, n_time_cum);
metrics.std_power       = zeros(1, n_time_cum);
metrics.std_linear      = zeros(1, n_time_cum);
metrics.ci95_power      = zeros(2, n_time_cum);
metrics.ci95_linear     = zeros(2, n_time_cum);

for i_t = 1:n_time_cum
    y_p = Y_power_obs(:, i_t);   % noise-inflated
    y_l = Y_linear_obs(:, i_t);

    % Mean and CI from noise-inflated samples
    metrics.mean_power(i_t)     = mean(y_p);
    metrics.mean_linear(i_t)    = mean(y_l);
    metrics.std_power(i_t)      = std(y_p);
    metrics.std_linear(i_t)     = std(y_l);
    metrics.ci95_power(:, i_t)  = [prctile(y_p, 2.5); prctile(y_p, 97.5)];
    metrics.ci95_linear(:, i_t) = [prctile(y_l, 2.5); prctile(y_l, 97.5)];
    metrics.mean_diff(i_t)      = mean(y_p) - mean(y_l);

    [~, p_ks, ks_val] = kstest2(y_p, y_l);
    metrics.ks_stat(i_t) = ks_val;
    metrics.ks_pval(i_t) = p_ks;

    if std(y_p) < 1e-10 && std(y_l) < 1e-10
        continue;
    end

    [kl_pl, kl_lp, js, ov] = compute_divergences_local(y_p, y_l);
    metrics.kl_power_linear(i_t) = kl_pl;
    metrics.kl_linear_power(i_t) = kl_lp;
    metrics.js_divergence(i_t)   = js;
    metrics.overlap_coef(i_t)    = ov;
    metrics.auc(i_t)             = compute_auc_local(y_p, y_l);
end

% prob_power_greater on clean trajectories (pure model comparison, no noise)
metrics.prob_power_greater = mean(Y_power_valid > Y_linear_valid, 1);
% final_diff statistics on noise-inflated trajectories — consistent with the
% histogram in the final_yield plot (parameter uncertainty + measurement noise)
Y_final_diff_obs = Y_power_obs(:,end) - Y_linear_obs(:,end);
metrics.final_diff_mean = mean(Y_final_diff_obs);
metrics.final_diff_std  = std( Y_final_diff_obs);
metrics.final_diff_ci95 = [prctile(Y_final_diff_obs, 2.5), ...
                            prctile(Y_final_diff_obs, 97.5)];

% Integrated / max (skip index 1: t=0, Y=0 trivially)
Time_int    = Time(2:end);
metrics.kl_integrated  = trapz(Time_int, metrics.kl_power_linear(2:end));
metrics.js_integrated  = trapz(Time_int, metrics.js_divergence(2:end));
metrics.ks_integrated  = trapz(Time_int, metrics.ks_stat(2:end));
metrics.auc_integrated = trapz(Time_int, metrics.auc(2:end));

[metrics.js_max,  idx_js]  = max(metrics.js_divergence);  metrics.js_max_time  = Time(idx_js);
[metrics.kl_max,  idx_kl]  = max(metrics.kl_power_linear); metrics.kl_max_time = Time(idx_kl);
[metrics.ks_max,  idx_ks]  = max(metrics.ks_stat);         metrics.ks_max_time = Time(idx_ks);
[metrics.auc_max, idx_auc] = max(metrics.auc);             metrics.auc_max_time = Time(idx_auc);
Time_ks_max = Time(idx_ks);

%% Rate metrics  (noise-inflated Rate_obs, all time points meaningful)
if verbose
    fprintf('  Computing extraction rate metrics...\n');
end

metrics.Time_rate            = Time_rate;
metrics.js_rate              = zeros(1, n_time_rate);
metrics.kl_rate_power_linear = zeros(1, n_time_rate);
metrics.kl_rate_linear_power = zeros(1, n_time_rate);
metrics.ks_rate              = zeros(1, n_time_rate);
metrics.ks_rate_pval         = zeros(1, n_time_rate);
metrics.auc_rate             = zeros(1, n_time_rate);
metrics.overlap_rate         = zeros(1, n_time_rate);
metrics.mean_rate_power      = zeros(1, n_time_rate);
metrics.mean_rate_linear     = zeros(1, n_time_rate);
metrics.std_rate_power       = zeros(1, n_time_rate);
metrics.std_rate_linear      = zeros(1, n_time_rate);
metrics.ci95_rate_power      = zeros(2, n_time_rate);
metrics.ci95_rate_linear     = zeros(2, n_time_rate);

for i_t = 1:n_time_rate
    r_p = Rate_power_obs(:, i_t);    % noise-inflated
    r_l = Rate_linear_obs(:, i_t);

    metrics.mean_rate_power(i_t)      = mean(r_p);
    metrics.mean_rate_linear(i_t)     = mean(r_l);
    metrics.std_rate_power(i_t)       = std(r_p);
    metrics.std_rate_linear(i_t)      = std(r_l);
    metrics.ci95_rate_power(:, i_t)   = [prctile(r_p, 2.5); prctile(r_p, 97.5)];
    metrics.ci95_rate_linear(:, i_t)  = [prctile(r_l, 2.5); prctile(r_l, 97.5)];

    [~, p_ks, ks_val] = kstest2(r_p, r_l);
    metrics.ks_rate(i_t)      = ks_val;
    metrics.ks_rate_pval(i_t) = p_ks;

    if std(r_p) < 1e-12 && std(r_l) < 1e-12
        continue;
    end

    [kl_pl, kl_lp, js, ov] = compute_divergences_local(r_p, r_l);
    metrics.kl_rate_power_linear(i_t) = kl_pl;
    metrics.kl_rate_linear_power(i_t) = kl_lp;
    metrics.js_rate(i_t)              = js;
    metrics.overlap_rate(i_t)         = ov;
    metrics.auc_rate(i_t)             = compute_auc_local(r_p, r_l);
end

% Integrate over all rate time points (no trivial zero at t=0 for rates)
metrics.js_rate_integrated  = trapz(Time_rate, metrics.js_rate);
metrics.ks_rate_integrated  = trapz(Time_rate, metrics.ks_rate);
metrics.auc_rate_integrated = trapz(Time_rate, metrics.auc_rate);

[metrics.js_rate_max,  idx_jsr]  = max(metrics.js_rate);   metrics.js_rate_max_time  = Time_rate(idx_jsr);
[metrics.ks_rate_max,  idx_ksr]  = max(metrics.ks_rate);   metrics.ks_rate_max_time  = Time_rate(idx_ksr);
[metrics.auc_rate_max, idx_aucr] = max(metrics.auc_rate);  metrics.auc_rate_max_time = Time_rate(idx_aucr);

%% Metric CI via N_noiseCI independent noise draws on clean trajectories
% Re-draws N(0,sigma^2) noise on the fixed clean MC ensembles (Y_power_valid,
% Rate_power_valid) without re-running the ODE integrators.  Recomputes JS,
% KS, AUC and KL at every time point for each draw, then takes 2.5/97.5
% percentiles to produce [2 x n_time] CI bands.
% Seeds: Seed+100+b (cumulative), Seed+200+b (rate) — avoids collision with
%        Seed+10 and Seed+11 used for the main noise inflation above.
N_noiseCI = p.Results.N_noiseCI;
if N_noiseCI > 0
    if verbose
        fprintf('  Computing metric CI (%d noise draws)...\n', N_noiseCI);
    end

    % Preallocate [N_noiseCI x n_time] boot arrays
    boot_js_cum     = zeros(N_noiseCI, n_time_cum);
    boot_ks_cum     = zeros(N_noiseCI, n_time_cum);
    boot_auc_cum    = zeros(N_noiseCI, n_time_cum);
    boot_kl_pl_cum  = zeros(N_noiseCI, n_time_cum);
    boot_kl_lp_cum  = zeros(N_noiseCI, n_time_cum);
    boot_js_rate    = zeros(N_noiseCI, n_time_rate);
    boot_ks_rate    = zeros(N_noiseCI, n_time_rate);
    boot_auc_rate   = zeros(N_noiseCI, n_time_rate);
    boot_kl_pl_rate = zeros(N_noiseCI, n_time_rate);
    boot_kl_lp_rate = zeros(N_noiseCI, n_time_rate);

    for b = 1:N_noiseCI
        % --- Cumulative noise draw ---
        rng(p.Results.Seed + 100 + b);
        Y_p_b = max(0, Y_power_valid  + sigma_empirical_power_cum  * randn(size(Y_power_valid)));
        Y_l_b = max(0, Y_linear_valid + sigma_empirical_linear_cum * randn(size(Y_linear_valid)));

        for i_t = 1:n_time_cum
            y_p_b = Y_p_b(:, i_t);
            y_l_b = Y_l_b(:, i_t);
            if std(y_p_b) < 1e-10 && std(y_l_b) < 1e-10
                continue;
            end
            [~, ~, ks_b] = kstest2(y_p_b, y_l_b);
            boot_ks_cum(b, i_t) = ks_b;
            [kl_pl_b, kl_lp_b, js_b, ~] = compute_divergences_local(y_p_b, y_l_b);
            boot_js_cum(b, i_t)    = js_b;
            boot_kl_pl_cum(b, i_t) = kl_pl_b;
            boot_kl_lp_cum(b, i_t) = kl_lp_b;
            boot_auc_cum(b, i_t)   = compute_auc_local(y_p_b, y_l_b);
        end

        % --- Rate (increment) noise draw ---
        rng(p.Results.Seed + 200 + b);
        R_p_b = max(0, Rate_power_valid  + sigma_empirical_power_diff  * randn(size(Rate_power_valid)));
        R_l_b = max(0, Rate_linear_valid + sigma_empirical_linear_diff * randn(size(Rate_linear_valid)));

        for i_t = 1:n_time_rate
            r_p_b = R_p_b(:, i_t);
            r_l_b = R_l_b(:, i_t);
            if std(r_p_b) < 1e-12 && std(r_l_b) < 1e-12
                continue;
            end
            [~, ~, ks_b] = kstest2(r_p_b, r_l_b);
            boot_ks_rate(b, i_t) = ks_b;
            [kl_pl_b, kl_lp_b, js_b, ~] = compute_divergences_local(r_p_b, r_l_b);
            boot_js_rate(b, i_t)    = js_b;
            boot_kl_pl_rate(b, i_t) = kl_pl_b;
            boot_kl_lp_rate(b, i_t) = kl_lp_b;
            boot_auc_rate(b, i_t)   = compute_auc_local(r_p_b, r_l_b);
        end
    end

    % Store CI fields: [2 x n_time], row 1 = 2.5th pct, row 2 = 97.5th pct
    metrics.ci95_js_cum     = prctile(boot_js_cum,     [2.5, 97.5], 1);
    metrics.ci95_ks_cum     = prctile(boot_ks_cum,     [2.5, 97.5], 1);
    metrics.ci95_auc_cum    = prctile(boot_auc_cum,    [2.5, 97.5], 1);
    metrics.ci95_kl_pl_cum  = prctile(boot_kl_pl_cum,  [2.5, 97.5], 1);
    metrics.ci95_kl_lp_cum  = prctile(boot_kl_lp_cum,  [2.5, 97.5], 1);

    metrics.ci95_js_rate    = prctile(boot_js_rate,    [2.5, 97.5], 1);
    metrics.ci95_ks_rate    = prctile(boot_ks_rate,    [2.5, 97.5], 1);
    metrics.ci95_auc_rate   = prctile(boot_auc_rate,   [2.5, 97.5], 1);
    metrics.ci95_kl_pl_rate = prctile(boot_kl_pl_rate, [2.5, 97.5], 1);
    metrics.ci95_kl_lp_rate = prctile(boot_kl_lp_rate, [2.5, 97.5], 1);

    % Mean across noise draws — smooth representative for rate_comparison_ci main curves.
    % Averaging N_noiseCI realisations cancels noise variance, leaving only the
    % parameter-uncertainty signal (no ODE re-runs required).
    metrics.mean_ci_js_cum     = mean(boot_js_cum,     1);  % [1 x n_time_cum]
    metrics.mean_ci_ks_cum     = mean(boot_ks_cum,     1);
    metrics.mean_ci_auc_cum    = mean(boot_auc_cum,    1);
    metrics.mean_ci_kl_pl_cum  = mean(boot_kl_pl_cum,  1);
    metrics.mean_ci_kl_lp_cum  = mean(boot_kl_lp_cum,  1);

    metrics.mean_ci_js_rate    = mean(boot_js_rate,    1);  % [1 x n_time_rate]
    metrics.mean_ci_ks_rate    = mean(boot_ks_rate,    1);
    metrics.mean_ci_auc_rate   = mean(boot_auc_rate,   1);
    metrics.mean_ci_kl_pl_rate = mean(boot_kl_pl_rate, 1);
    metrics.mean_ci_kl_lp_rate = mean(boot_kl_lp_rate, 1);

    if verbose
        fprintf('  Metric CI done.\n');
    end
end

%% Primary outputs
max_KS        = metrics.ks_max;
integrated_JS = metrics.js_integrated;

%% Pack results
if nargout >= 3
    results = struct();
    % Operating point
    results.T0             = T;
    results.P0             = P;
    results.F0             = F;
    results.ExtractionTime = finalTime;
    results.N_MC           = N_MC;
    results.Seed           = p.Results.Seed;   % for reproducible CI draws in plot
    results.N_noiseCI      = N_noiseCI;        % number of noise draws used for CI
    results.dt_rate        = dt_rate;
    results.stride         = stride;
    results.n_valid_cum    = n_valid_cum;
    results.n_valid_diff   = n_valid_diff;
    % Empirical sigmas (for reference)
    results.sigma_empirical_power_cum   = sigma_empirical_power_cum;
    results.sigma_empirical_linear_cum  = sigma_empirical_linear_cum;
    results.sigma_empirical_power_diff  = sigma_empirical_power_diff;
    results.sigma_empirical_linear_diff = sigma_empirical_linear_diff;
    % Cumulative yield trajectories — clean (no noise), from Cov_XXX_cum loop
    results.Time           = Time;
    results.Y_power_valid  = Y_power_valid;
    results.Y_linear_valid = Y_linear_valid;
    results.Y_power_nom    = Y_power_nom;
    results.Y_linear_nom   = Y_linear_nom;
    % Cumulative yield trajectories — noise-inflated (for distribution diagnostics)
    results.Y_power_obs    = Y_power_obs;
    results.Y_linear_obs   = Y_linear_obs;
    % Rate trajectories — clean (no noise), from Cov_XXX_diff loop
    results.Time_rate          = Time_rate;
    results.Rate_power_valid   = Rate_power_valid;
    results.Rate_linear_valid  = Rate_linear_valid;
    results.Rate_power_nom     = Rate_power_nom;
    results.Rate_linear_nom    = Rate_linear_nom;
    % Rate trajectories — noise-inflated (for rate distribution diagnostics)
    results.Rate_power_obs  = Rate_power_obs;
    results.Rate_linear_obs = Rate_linear_obs;
    % Parameter samples (both loops)
    results.theta_power_cum_samples   = theta_power_cum;
    results.theta_linear_cum_samples  = theta_linear_cum;
    results.theta_power_diff_samples  = theta_power_diff;
    results.theta_linear_diff_samples = theta_linear_diff;
    % Covariance matrices
    results.Cov_power_cum   = Cov_power_cum;
    results.Cov_linear_cum  = Cov_linear_cum;
    results.Cov_power_diff  = Cov_power_diff;
    results.Cov_linear_diff = Cov_linear_diff;
    % All metrics (computed on noise-inflated samples)
    results.metrics = metrics;
end

if verbose
    fprintf('  --- Cumulative yield metrics (with empirical noise) ---\n');
    fprintf('  Max KS:           %.4f at t=%.1f min\n', max_KS, metrics.ks_max_time);
    fprintf('  Integrated JS:    %.4f nats*min\n',      integrated_JS);
    fprintf('  Max JS:           %.4f at t=%.1f min\n', metrics.js_max, metrics.js_max_time);
    fprintf('  --- Rate metrics (with empirical noise) ---\n');
    fprintf('  Max JS (rate):    %.4f at t=%.1f min\n', metrics.js_rate_max, metrics.js_rate_max_time);
    fprintf('  Integrated JS (rate): %.4f nats*min\n',  metrics.js_rate_integrated);
end

end

%% Local function: KL / JS / overlap via kernel density
function [kl_pl, kl_lp, js, overlap] = compute_divergences_local(y_p, y_l)
    y_all = [y_p; y_l];
    y_min = min(y_all);
    y_max = max(y_all);

    if y_max <= y_min || numel(y_all) < 5
        kl_pl = 0; kl_lp = 0; js = 0; overlap = 1;
        return;
    end

    n_grid = 200;
    y_grid = linspace(y_min - 0.1*(y_max-y_min), y_max + 0.1*(y_max-y_min), n_grid);

    f_p = ksdensity(y_p, y_grid, 'Function', 'pdf');
    f_l = ksdensity(y_l, y_grid, 'Function', 'pdf');

    dx  = y_grid(2) - y_grid(1);
    f_p = f_p / (sum(f_p) * dx);
    f_l = f_l / (sum(f_l) * dx);

    eps_val = 1e-12;
    f_p = max(f_p, eps_val);
    f_l = max(f_l, eps_val);

    kl_pl = trapz(y_grid, f_p .* log(f_p ./ f_l));
    kl_lp = trapz(y_grid, f_l .* log(f_l ./ f_p));

    f_m = 0.5 * (f_p + f_l);
    js  = 0.5 * trapz(y_grid, f_p .* log(f_p ./ f_m)) + ...
          0.5 * trapz(y_grid, f_l .* log(f_l ./ f_m));

    overlap = trapz(y_grid, min(f_p, f_l));
end

%% Local function: AUC via Mann-Whitney U
function auc = compute_auc_local(y_p, y_l)
    n_p = numel(y_p);
    n_l = numel(y_l);

    if n_p < 2 || n_l < 2
        auc = 0.5;
        return;
    end

    count_greater = 0;
    count_equal   = 0;
    for i = 1:n_p
        count_greater = count_greater + sum(y_p(i) > y_l);
        count_equal   = count_equal   + sum(y_p(i) == y_l);
    end

    auc = (count_greater + 0.5 * count_equal) / (n_p * n_l);
    % Directional AUC: P(y_p > y_l) in [0,1].
    % 0.5 = no discrimination; >0.5 = Power dominates; <0.5 = Linear dominates.
    % Do NOT fold with max(auc, 1-auc) — that discards directionality and
    % silently inflates CI lower bounds when noise draws cross the 0.5 boundary.
end
