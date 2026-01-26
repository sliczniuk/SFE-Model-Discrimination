function [max_KS, integrated_JS, results] = compute_discrimination_metrics(T, P, F, timeStep, finalTime, varargin)
% COMPUTE_DISCRIMINATION_METRICS Compute model discrimination metrics via Monte Carlo
%
% Evaluates discrimination between Power and Linear SFE extraction models
% at a fixed operating point (T, P, F) while propagating kinetic parameter
% uncertainty through Monte Carlo sampling.
%
% Syntax:
%   [max_KS, integrated_JS] = compute_discrimination_metrics(T, P, F, timeStep, finalTime)
%   [max_KS, integrated_JS, results] = compute_discrimination_metrics(T, P, F, timeStep, finalTime)
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
%   'N_MC'    - Number of Monte Carlo samples (default: 500)
%   'Seed'    - Random seed for reproducibility (default: 42)
%   'Verbose' - Print progress messages (default: true)
%
% Outputs:
%   max_KS       - Maximum KS statistic over trajectory
%   integrated_JS - JS divergence integrated over time [nats*min]
%   results      - Struct containing full results:
%                  .Time           - Time vector [min]
%                  .Y_power_valid  - Valid Power model trajectories
%                  .Y_linear_valid - Valid Linear model trajectories
%                  .Y_power_nom    - Nominal Power trajectory
%                  .Y_linear_nom   - Nominal Linear trajectory
%                  .metrics        - Time-pointwise metrics struct
%                  .n_valid        - Number of valid MC samples
%
% Example:
%   [max_KS, int_JS] = compute_discrimination_metrics(303, 150, 5e-5, 5, 600);
%   fprintf('Max KS: %.4f, Integrated JS: %.4f\n', max_KS, int_JS);

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
parse(p, T, P, F, timeStep, finalTime, varargin{:});

N_MC = p.Results.N_MC;
verbose = p.Results.Verbose;

rng(p.Results.Seed);

import casadi.*

%% Start parallel pool if not already running
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local');
end
n_workers = pool.NumWorkers;

if verbose
    fprintf('Computing discrimination metrics at T=%.0fK, P=%.0f bar, F=%.2e m3/s\n', T, P, F);
    fprintf('  MC samples: %d, Workers: %d\n', N_MC, n_workers);
end

%% Load parameters and covariance matrices
Parameters_table = readtable('Parameters.csv');
Parameters = num2cell(Parameters_table{:,3});

% Power model covariance (4x4): [k_w0, a_w, b_w, n_k]
Cov_power = [
    0.0029,  0.0066,  0.0000,  0.0054;
    0.0066,  0.0772,  0.0020,  0.0009;
    0.0000,  0.0020,  0.0082, -0.0004;
    0.0054,  0.0009, -0.0004,  0.0313
];

% Linear model FULL covariance (6x6): [D_i(0), D_i(Re), D_i(F), Ups(0), Ups(Re), Ups(F)]
Cov_linear_full = [
    0.0127,  0.0286, -0.0040,  0.0263,  0.0193, -0.0058;
    0.0286,  0.4420, -0.0316,  0.0224,  0.6514, -0.0401;
   -0.0040, -0.0316,  0.0027, -0.0060, -0.0382,  0.0033;
    0.0263,  0.0224, -0.0060,  0.2018, -0.0717, -0.0313;
    0.0193,  0.6514, -0.0382, -0.0717,  5.9260, -0.2668;
   -0.0058, -0.0401,  0.0033, -0.0313, -0.2668,  0.0186
];

% Nominal parameter values
theta_power = [1.222524; 4.308414; 0.972739; 3.428618];
theta_Di = [0.19; -8.188; 0.62];
theta_Upsilon = [3.158; 11.922; -0.6868];
theta_linear = [theta_Di; theta_Upsilon];

%% Setup simulation infrastructure
m_total = 3.0;
before = 0.04;
bed = 0.92;

Time_in_sec = (timeStep:timeStep:finalTime) * 60;
Time = [0, Time_in_sec/60];
N_Time = length(Time_in_sec);
timeStep_in_sec = timeStep * 60;

nstages = Parameters{1};
r = Parameters{3};
epsi = Parameters{4};
L = Parameters{6};

nstagesbefore = 1:floor(before * nstages);
nstagesbed = nstagesbefore(end)+1 : nstagesbefore(end) + floor(bed * nstages);
nstagesafter = nstagesbed(end)+1 : nstages;

bed_mask = nan(nstages, 1);
bed_mask(nstagesbefore) = 0;
bed_mask(nstagesbed) = 1;
bed_mask(nstagesafter) = 0;

V_slice = (L/nstages) * pi * r^2;
V_bed = V_slice * numel(nstagesbed);

V_before_fluid = repmat(V_slice * numel(nstagesbefore) / numel(nstagesbefore), numel(nstagesbefore), 1);
V_bed_fluid = repmat(V_bed * (1-epsi) / numel(nstagesbed), numel(nstagesbed), 1);
V_after_fluid = repmat(V_slice * numel(nstagesafter) / numel(nstagesafter), numel(nstagesafter), 1);
V_fluid = [V_before_fluid; V_bed_fluid; V_after_fluid];

C0solid = m_total * 1e-3 / (V_bed * epsi);
Parameters{2} = C0solid;

m_fluid = zeros(1, nstages);
C0fluid = m_fluid * 1e-3 ./ V_fluid';

epsi_mask = epsi .* bed_mask;
one_minus_epsi_mask = 1 - epsi_mask;

Nx = 3 * nstages + 2;
Nu = 3 + numel(Parameters);

% Compute fluid properties
Z = Compressibility(T, P, Parameters);
rho = rhoPB_Comp(T, P, Z, Parameters);
enthalpy_rho = rho * SpecificEnthalpy(T, P, Z, rho, Parameters);

% Initial state
x0 = [C0fluid'; C0solid * bed_mask; enthalpy_rho * ones(nstages, 1); P; 0];

% Input vectors
feedTemp = T * ones(1, N_Time);
feedPress = P * ones(1, N_Time);
feedFlow = F * ones(1, N_Time);
uu = [feedTemp', feedPress', feedFlow'];
U_base = [uu'; repmat(cell2mat(Parameters), 1, N_Time)];

%% Nominal simulation
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

X_power_nom = F_accum_power_nom(x0, U_base);
X_linear_nom = F_accum_linear_nom(x0, U_base);

Y_power_nom = full([0, X_power_nom(Nx, :)]);
Y_linear_nom = full([0, X_linear_nom(Nx, :)]);

%% Monte Carlo sampling
if verbose
    fprintf('  Running MC simulations...\n');
end

% Cholesky decomposition
try
    L_power_chol = chol(Cov_power + 1e-10*eye(4), 'lower');
    L_linear_chol = chol(Cov_linear_full + 1e-10*eye(6), 'lower');
catch
    warning('Cholesky failed, using diagonal approximation');
    L_power_chol = diag(sqrt(diag(Cov_power)));
    L_linear_chol = diag(sqrt(diag(Cov_linear_full)));
end

% Generate parameter samples
Z_power_samples = randn(4, N_MC);
Z_linear_samples = randn(6, N_MC);

theta_power_samples = repmat(theta_power, 1, N_MC) + L_power_chol * Z_power_samples;
theta_linear_samples = repmat(theta_linear, 1, N_MC) + L_linear_chol * Z_linear_samples;

% Storage
Y_power_MC_cell = cell(N_MC, 1);
Y_linear_MC_cell = cell(N_MC, 1);
valid_power = true(N_MC, 1);
valid_linear = true(N_MC, 1);

% Local variables for parfor
bed_mask_local = bed_mask;
epsi_mask_local = epsi_mask;
one_minus_epsi_mask_local = one_minus_epsi_mask;
timeStep_in_sec_local = timeStep_in_sec;
Nx_local = Nx;
Nu_local = Nu;
N_Time_local = N_Time;
x0_local = x0;
U_base_local = U_base;

tic;
parfor i = 1:N_MC
    % Power model
    theta_p = theta_power_samples(:, i);
    try
        f_power_i = @(x, u) modelSFE_power_with_params(x, u, bed_mask_local, timeStep_in_sec_local, ...
            epsi_mask_local, one_minus_epsi_mask_local, theta_p(1), theta_p(2), theta_p(3), theta_p(4));

        F_power_i = buildIntegrator(f_power_i, [Nx_local, Nu_local], timeStep_in_sec_local, 'cvodes');
        F_accum_power_i = F_power_i.mapaccum('F_accum', N_Time_local);

        X_power_i = F_accum_power_i(x0_local, U_base_local);
        Y_power_i = full([0, X_power_i(Nx_local, :)]);

        if any(isnan(Y_power_i)) || any(Y_power_i < 0) || any(Y_power_i > 10)
            valid_power(i) = false;
            Y_power_MC_cell{i} = NaN(1, N_Time_local + 1);
        else
            Y_power_MC_cell{i} = Y_power_i;
        end
    catch
        valid_power(i) = false;
        Y_power_MC_cell{i} = NaN(1, N_Time_local + 1);
    end

    % Linear model
    theta_l = theta_linear_samples(:, i);
    try
        f_linear_i = @(x, u) modelSFE_linear_with_params(x, u, bed_mask_local, timeStep_in_sec_local, ...
            epsi_mask_local, one_minus_epsi_mask_local, theta_l(1), theta_l(2), theta_l(3), ...
            theta_l(4), theta_l(5), theta_l(6));

        F_linear_i = buildIntegrator(f_linear_i, [Nx_local, Nu_local], timeStep_in_sec_local, 'cvodes');
        F_accum_linear_i = F_linear_i.mapaccum('F_accum', N_Time_local);

        X_linear_i = F_accum_linear_i(x0_local, U_base_local);
        Y_linear_i = full([0, X_linear_i(Nx_local, :)]);

        if any(isnan(Y_linear_i)) || any(Y_linear_i < 0) || any(Y_linear_i > 10)
            valid_linear(i) = false;
            Y_linear_MC_cell{i} = NaN(1, N_Time_local + 1);
        else
            Y_linear_MC_cell{i} = Y_linear_i;
        end
    catch
        valid_linear(i) = false;
        Y_linear_MC_cell{i} = NaN(1, N_Time_local + 1);
    end
end
mc_time = toc;

if verbose
    fprintf('  MC completed in %.1f seconds\n', mc_time);
end

% Convert and filter
Y_power_MC = cell2mat(Y_power_MC_cell);
Y_linear_MC = cell2mat(Y_linear_MC_cell);

valid_both = valid_power & valid_linear;
Y_power_valid = Y_power_MC(valid_both, :);
Y_linear_valid = Y_linear_MC(valid_both, :);
n_valid = sum(valid_both);

if verbose
    fprintf('  Valid samples: %d/%d\n', n_valid, N_MC);
end

%% Compute divergence metrics
Time_full = Time;
n_time_full = length(Time_full);

metrics = struct();
metrics.Time = Time_full;
metrics.T0 = T;
metrics.P0 = P;
metrics.F0 = F;

% Preallocate
metrics.kl_power_linear = zeros(1, n_time_full);
metrics.kl_linear_power = zeros(1, n_time_full);
metrics.js_divergence = zeros(1, n_time_full);
metrics.ks_stat = zeros(1, n_time_full);
metrics.ks_pval = zeros(1, n_time_full);
metrics.mean_diff = zeros(1, n_time_full);
metrics.overlap_coef = zeros(1, n_time_full);

metrics.mean_power = zeros(1, n_time_full);
metrics.mean_linear = zeros(1, n_time_full);
metrics.std_power = zeros(1, n_time_full);
metrics.std_linear = zeros(1, n_time_full);
metrics.ci95_power = zeros(2, n_time_full);
metrics.ci95_linear = zeros(2, n_time_full);

for i_t = 1:n_time_full
    y_p = Y_power_valid(:, i_t);
    y_l = Y_linear_valid(:, i_t);

    metrics.mean_power(i_t) = mean(y_p);
    metrics.mean_linear(i_t) = mean(y_l);
    metrics.std_power(i_t) = std(y_p);
    metrics.std_linear(i_t) = std(y_l);
    metrics.ci95_power(:, i_t) = [prctile(y_p, 2.5); prctile(y_p, 97.5)];
    metrics.ci95_linear(:, i_t) = [prctile(y_l, 2.5); prctile(y_l, 97.5)];
    metrics.mean_diff(i_t) = mean(y_p) - mean(y_l);

    [~, p_ks, ks_stat] = kstest2(y_p, y_l);
    metrics.ks_stat(i_t) = ks_stat;
    metrics.ks_pval(i_t) = p_ks;

    if i_t == 1 || (std(y_p) < 1e-10 && std(y_l) < 1e-10)
        continue;
    end

    [kl_pl, kl_lp, js, overlap] = compute_divergences_local(y_p, y_l);
    metrics.kl_power_linear(i_t) = kl_pl;
    metrics.kl_linear_power(i_t) = kl_lp;
    metrics.js_divergence(i_t) = js;
    metrics.overlap_coef(i_t) = overlap;
end

%% Compute integrated and max metrics
Time_int = Time_full(2:end);

metrics.kl_integrated = trapz(Time_int, metrics.kl_power_linear(2:end));
metrics.js_integrated = trapz(Time_int, metrics.js_divergence(2:end));
metrics.ks_integrated = trapz(Time_int, metrics.ks_stat(2:end));

[metrics.js_max, idx_js_max] = max(metrics.js_divergence);
metrics.js_max_time = Time_full(idx_js_max);

[metrics.kl_max, idx_kl_max] = max(metrics.kl_power_linear);
metrics.kl_max_time = Time_full(idx_kl_max);

[metrics.ks_max, idx_ks_max] = max(metrics.ks_stat);
metrics.ks_max_time = Time_full(idx_ks_max);

metrics.prob_power_greater = mean(Y_power_valid > Y_linear_valid, 1);

metrics.final_diff_mean = mean(Y_power_valid(:, end) - Y_linear_valid(:, end));
metrics.final_diff_std = std(Y_power_valid(:, end) - Y_linear_valid(:, end));
metrics.final_diff_ci95 = [prctile(Y_power_valid(:, end) - Y_linear_valid(:, end), 2.5), ...
                           prctile(Y_power_valid(:, end) - Y_linear_valid(:, end), 97.5)];

%% Outputs
max_KS = metrics.ks_max;
integrated_JS = metrics.js_integrated;

if nargout >= 3
    results = struct();
    results.T0 = T;
    results.P0 = P;
    results.F0 = F;
    results.ExtractionTime = finalTime;
    results.N_MC = N_MC;
    results.n_valid = n_valid;
    results.Time = Time_full;
    results.Y_power_valid = Y_power_valid;
    results.Y_linear_valid = Y_linear_valid;
    results.Y_power_nom = Y_power_nom;
    results.Y_linear_nom = Y_linear_nom;
    results.metrics = metrics;
    results.theta_power_samples = theta_power_samples;
    results.theta_linear_samples = theta_linear_samples;
    results.Cov_power = Cov_power;
    results.Cov_linear = Cov_linear_full;
end

if verbose
    fprintf('  Max KS: %.4f at t=%.0f min\n', max_KS, metrics.ks_max_time);
    fprintf('  Integrated JS: %.4f nats*min\n', integrated_JS);
end

end

%% Local function: compute_divergences
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

    dx = y_grid(2) - y_grid(1);
    f_p = f_p / (sum(f_p) * dx);
    f_l = f_l / (sum(f_l) * dx);

    eps_val = 1e-12;
    f_p = max(f_p, eps_val);
    f_l = max(f_l, eps_val);

    kl_pl = trapz(y_grid, f_p .* log(f_p ./ f_l));
    kl_lp = trapz(y_grid, f_l .* log(f_l ./ f_p));

    f_m = 0.5 * (f_p + f_l);
    js = 0.5 * trapz(y_grid, f_p .* log(f_p ./ f_m)) + ...
         0.5 * trapz(y_grid, f_l .* log(f_l ./ f_m));

    overlap = trapz(y_grid, min(f_p, f_l));
end
