function [] = compute_discrimination_trjactory(T, P, F, timeStep, finalTime, varargin)
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
%   'UseMap'  - Use CasADi map over samples (default: true)
%
% Outputs:
%   max_KS       - Maximum KS statistic over trajectory (yield-based)
%   integrated_JS - JS divergence integrated over time [nats*min] (yield-based)
%   Time_ks_max  - Time at which max KS occurs [min]
%   results      - Struct containing full results:
%                  .Time              - Time vector for yields [min]
%                  .Y_power_valid     - Valid Power model yield trajectories
%                  .Y_linear_valid    - Valid Linear model yield trajectories
%                  .Y_power_nom       - Nominal Power yield trajectory
%                  .Y_linear_nom      - Nominal Linear yield trajectory
%                  .Time_rate         - Time vector for rates [min] (midpoints)
%                  .Rate_power_valid  - Valid Power model rate trajectories
%                  .Rate_linear_valid - Valid Linear model rate trajectories
%                  .Rate_power_nom    - Nominal Power rate trajectory
%                  .Rate_linear_nom   - Nominal Linear rate trajectory
%                  .metrics           - Time-pointwise metrics struct:
%                      Yield-based: js_divergence, ks_stat, auc, kl_*, overlap_coef
%                      Rate-based:  js_rate, ks_rate, auc_rate, kl_rate_*, overlap_rate
%                      Integrated:  js_integrated, js_rate_integrated, etc.
%                      Max values:  js_max, js_rate_max, with _time suffix
%                  .n_valid           - Number of valid MC samples
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
addParameter(p, 'UseMap', true, @islogical);
parse(p, T, P, F, timeStep, finalTime, varargin{:});

N_MC = p.Results.N_MC;
verbose = p.Results.Verbose;
use_map = p.Results.UseMap;

rng(p.Results.Seed);

import casadi.*

if verbose
    fprintf('Computing discrimination metrics at T=%.0fK, P=%.0f bar, F=%.2e m3/s\n', T, P, F);
    fprintf('  MC samples: %d\n', N_MC);
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
theta_linear = [0.19; -8.188; 0.62; 3.158; 11.922; -0.6868];  % [D_i; Upsilon]

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
Nu = 3 + numel(Parameters);  % For nominal simulations

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
Y_power_MC = NaN(N_MC, N_Time + 1);
Y_linear_MC = NaN(N_MC, N_Time + 1);
Y_power_MC(:, 1) = 0;
Y_linear_MC(:, 1) = 0;

% Build integrators once and map over samples
n_param = numel(Parameters);
Nu_power = 3 + n_param + 4;
Nu_linear = 3 + n_param + 6;

f_power_mc = @(x, u) modelSFE_power_with_params(x, u, bed_mask, timeStep_in_sec, ...
    epsi_mask, one_minus_epsi_mask, n_param);
f_linear_mc = @(x, u) modelSFE_linear_with_params(x, u, bed_mask, timeStep_in_sec, ...
    epsi_mask, one_minus_epsi_mask, n_param);

F_power_mc = buildIntegrator(f_power_mc, [Nx, Nu_power], timeStep_in_sec, 'cvodes');
F_linear_mc = buildIntegrator(f_linear_mc, [Nx, Nu_linear], timeStep_in_sec, 'cvodes');

X_power = repmat(x0, 1, N_MC);
X_linear = repmat(x0, 1, N_MC);

tic;
if use_map
    F_power_map = F_power_mc.map(N_MC);
    F_linear_map = F_linear_mc.map(N_MC);

    for t = 1:N_Time
        u_base_t = U_base(:, t);
        U_power_t = [repmat(u_base_t, 1, N_MC); theta_power_samples];
        U_linear_t = [repmat(u_base_t, 1, N_MC); theta_linear_samples];

        X_power = F_power_map(X_power, U_power_t);
        X_linear = F_linear_map(X_linear, U_linear_t);

        Y_power_MC(:, t + 1) = full(X_power(Nx, :)).';
        Y_linear_MC(:, t + 1) = full(X_linear(Nx, :)).';
    end
else
    valid_power_step = true(N_MC, 1);
    valid_linear_step = true(N_MC, 1);
    for t = 1:N_Time
        u_base_t = U_base(:, t);
        for i = 1:N_MC
            if valid_power_step(i)
                u_power_i = [u_base_t; theta_power_samples(:, i)];
                try
                    X_power(:, i) = full(F_power_mc(X_power(:, i), u_power_i));
                catch
                    X_power(:, i) = NaN;
                    valid_power_step(i) = false;
                end
            end
            if valid_linear_step(i)
                u_linear_i = [u_base_t; theta_linear_samples(:, i)];
                try
                    X_linear(:, i) = full(F_linear_mc(X_linear(:, i), u_linear_i));
                catch
                    X_linear(:, i) = NaN;
                    valid_linear_step(i) = false;
                end
            end
        end

        Y_power_MC(:, t + 1) = X_power(Nx, :).';
        Y_linear_MC(:, t + 1) = X_linear(Nx, :).';
    end
end
mc_time = toc;

if verbose
    fprintf('  MC completed in %.1f seconds\n', mc_time);
end

valid_power = all(isfinite(Y_power_MC) & Y_power_MC >= 0 & Y_power_MC <= 10, 2);
valid_linear = all(isfinite(Y_linear_MC) & Y_linear_MC >= 0 & Y_linear_MC <= 10, 2);
Y_power_MC(~valid_power, :) = NaN;
Y_linear_MC(~valid_linear, :) = NaN;

valid_both = valid_power & valid_linear;
Y_power_valid = Y_power_MC(valid_both, :);
Y_linear_valid = Y_linear_MC(valid_both, :);
n_valid = sum(valid_both);

if verbose
    fprintf('  Valid samples: %d/%d\n', n_valid, N_MC);
end

%% Compute rate trajectories via forward difference
% Rate(t) = (Y(t+1) - Y(t)) / dt
dt_min = timeStep;  % Time step in minutes
Rate_power_valid = diff(Y_power_valid, 1, 2) / dt_min;  % [n_valid x (n_time-1)]
Rate_linear_valid = diff(Y_linear_valid, 1, 2) / dt_min;
Time_rate = Time(1:end-1) + dt_min;  % Midpoint times for rates
n_time_rate = length(Time_rate);

% Nominal rates
Rate_power_nom = diff(Y_power_nom) / dt_min;
Rate_linear_nom = diff(Y_linear_nom) / dt_min;

Rate_power_nom_matrix = Rate_power_valid * 1000;
Rate_linear_nom_matrix = Rate_linear_valid * 1000;

if verbose
    fprintf('  Computing rate-based metrics...\n');
end

display('MC finished');

%% Compute divergence metrics
%{
for ii = 1:numel(Time_rate)
    fprintf('\ntime: %.2f min\n\n', Time_rate(ii));

    r_p = Rate_power_valid(:, ii);
    r_l = Rate_linear_valid(:, ii);

    [sigmaPoewr, muPower] = robustcov(r_p);
    [sigmaLinear, muLinear] = robustcov(r_l);

    [js_mean, js_diag] = js_gaussian_mc_reps(muPower, sigmaPoewr, muLinear, sigmaLinear, 1000, 3);
    fprintf('JS (Gaussian MC) = %.4f nats | n=%d | reps=%d | std=%.4f | rel=%.2f%% | 95%% CI [%.4f, %.4f]\n', ...
        js_mean, js_diag.n_per_rep, js_diag.reps, js_diag.std, 100*js_diag.rel, js_diag.ci95(1), js_diag.ci95(2));
end


for ii = 4:numel(Time_rate)
    fprintf('\ntime span: %.2f - %.2f min \n\n', Time_rate(3), Time_rate(ii));
    [sigmaPoewr, muPower] = robustcov(Rate_power_nom_matrix(:,3:ii))
    [sigmaLinear, muLinear] = robustcov(Rate_linear_nom_matrix(:,3:ii))

    d = size(Rate_power_nom_matrix(:,2:ii), 2);
    js_n = max(1000, 200 * d);
    js_reps = 3;

    [js_mean, js_diag] = js_gaussian_mc_reps(muPower, sigmaPoewr, muLinear, sigmaLinear, js_n, js_reps);
    fprintf('JS (Gaussian MC) = %.4f nats | n=%d | reps=%d | std=%.4f | rel=%.2f%% | 95%% CI [%.4f, %.4f]\n', ...
        js_mean, js_diag.n_per_rep, js_diag.reps, js_diag.std, 100*js_diag.rel, js_diag.ci95(1), js_diag.ci95(2));
    if js_diag.invalid_p > 0 || js_diag.invalid_q > 0
        fprintf('  Warning: non-finite log terms (P:%d, Q:%d)\n', js_diag.invalid_p, js_diag.invalid_q);
    end
end
%}
for ii = 2:numel(Time_rate)
    Xp = Rate_power_nom_matrix(:,1:ii);
    Xl = Rate_linear_nom_matrix(:,1:ii);

    x_t_p = Xp(:,end);
    x_t_l = Xl(:,end);
    X_prev_p = Xp(:,1:end-1);
    X_prev_l = Xl(:,1:end-1);

    % Regularized LS in case of collinearity
    lambda = 1e-6;
    beta_p = (X_prev_p.'*X_prev_p + lambda*eye(size(X_prev_p,2))) \ (X_prev_p.'*x_t_p);
    beta_l = (X_prev_l.'*X_prev_l + lambda*eye(size(X_prev_l,2))) \ (X_prev_l.'*x_t_l);

    res_p = x_t_p - X_prev_p*beta_p;
    res_l = x_t_l - X_prev_l*beta_l;

    [sigmaP, muP] = robustcov(res_p);
    [sigmaL, muL] = robustcov(res_l);

    [js_mean, js_diag] = js_gaussian_mc_reps(muP, sigmaP, muL, sigmaL, 1000, 3);
    fprintf('Incremental JS = %.4f nats at t=%.2f min\n', js_mean, Time_rate(ii));
end



display('Analysis finished');

%%
function [js, diag] = js_gaussian_mc(muP, sigP, muL, sigL, n)
    Xp = mvnrnd(muP, sigP, n);
    Xq = mvnrnd(muL, sigL, n);

    logp_xp = logmvnpdf(Xp, muP, sigP);
    logq_xp = logmvnpdf(Xp, muL, sigL);
    logm_xp = logsumexp([logp_xp, logq_xp], 2) - log(2);

    logp_xq = logmvnpdf(Xq, muP, sigP);
    logq_xq = logmvnpdf(Xq, muL, sigL);
    logm_xq = logsumexp([logp_xq, logq_xq], 2) - log(2);

    a = logp_xp - logm_xp;
    b = logq_xq - logm_xq;

    mask_p = isfinite(a);
    mask_q = isfinite(b);
    a = a(mask_p);
    b = b(mask_q);

    n_p = numel(a);
    n_q = numel(b);
    if n_p == 0 || n_q == 0
        js = NaN;
        diag = struct('se', NaN, 'ci95', [NaN, NaN], ...
            'n_p', n_p, 'n_q', n_q, ...
            'invalid_p', n - n_p, 'invalid_q', n - n_q);
        return;
    end

    mean_a = mean(a);
    mean_b = mean(b);
    var_a = var(a, 1);
    var_b = var(b, 1);

    js = 0.5 * (mean_a + mean_b);

    se = 0.5 * sqrt(var_a / n_p + var_b / n_q);
    ci95 = js + 1.96 * se * [-1, 1];

    diag = struct('se', se, 'ci95', ci95, ...
        'n_p', n_p, 'n_q', n_q, ...
        'invalid_p', n - n_p, 'invalid_q', n - n_q);
end

function [js_mean, diag] = js_gaussian_mc_reps(muP, sigP, muL, sigL, n, reps)
    js_vals = zeros(reps, 1);
    invalid_p = 0;
    invalid_q = 0;
    n_p_total = 0;
    n_q_total = 0;
    for k = 1:reps
        [js_vals(k), d] = js_gaussian_mc(muP, sigP, muL, sigL, n);
        invalid_p = invalid_p + d.invalid_p;
        invalid_q = invalid_q + d.invalid_q;
        n_p_total = n_p_total + d.n_p;
        n_q_total = n_q_total + d.n_q;
    end

    js_mean = mean(js_vals);
    js_std = std(js_vals, 1);
    js_se = js_std / sqrt(reps);
    ci95 = js_mean + 1.96 * js_se * [-1, 1];
    rel = js_std / max(abs(js_mean), eps);

    diag = struct( ...
        'std', js_std, ...
        'se', js_se, ...
        'ci95', ci95, ...
        'rel', rel, ...
        'reps', reps, ...
        'n_per_rep', n, ...
        'n_p_total', n_p_total, ...
        'n_q_total', n_q_total, ...
        'invalid_p', invalid_p, ...
        'invalid_q', invalid_q);
end

function y = logmvnpdf(X, mu, Sigma)
    % X: n x d, mu: 1 x d or d x 1, Sigma: d x d
    if isrow(mu), mu = mu(:); end
    X0 = X - mu.';
    d = size(X0, 2);

    % Cholesky with jitter fallback
    [L, p] = chol(Sigma, 'lower');
    if p > 0
        jitter = 1e-10 * trace(Sigma) / d;
        [L, p] = chol(Sigma + jitter * eye(d), 'lower');
        if p > 0
            error('logmvnpdf:SigmaNotPD', 'Sigma not positive definite even after jitter.');
        end
    end

    alpha = L \ X0.';                 % d x n
    quad = sum(alpha.^2, 1).';        % n x 1
    logdet = 2 * sum(log(diag(L)));

    y = -0.5 * (d*log(2*pi) + logdet + quad);
end

function y = logsumexp(A, dim)
    if nargin < 2, dim = 2; end
    m = max(A, [], dim);
    y = m + log(sum(exp(A - m), dim));
end

end
