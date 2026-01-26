%% Single Point Model Discrimination Analysis with Full Model Uncertainty Propagation
% This script computes discrimination metrics between Power and Linear models
% at a single operating point (T, P, F) using three uncertainty propagation methods:
%   1. Delta method (first-order Taylor expansion) - uses analytical approximations
%   2. Monte Carlo sampling - uses FULL CasADi models with parameterized functions
%   3. Sigma-point (unscented) transform - uses FULL CasADi models
%
% IMPORTANT: This version uses the original SFE models (not approximations)
% by creating parameterized versions of the kinetic functions.

%% Initialization
startup;

addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
import casadi.*

rng(42);  % Set seed for reproducibility

fprintf('=============================================================================\n');
fprintf('   SINGLE POINT MODEL DISCRIMINATION - FULL MODEL UNCERTAINTY PROPAGATION   \n');
fprintf('=============================================================================\n\n');

%% ========================================================================
%  USER-CONFIGURABLE OPERATING CONDITIONS
%  ========================================================================
T0 = 308;      % Temperature [K] (35°C)
P0 = 200;      % Pressure [bar]
F0 = 5e-5;     % Flow rate [m³/s]

ExtractionTime = 120;  % Extraction time [minutes]
timeStep = 5;          % Time step [minutes]

% Monte Carlo configuration
N_MC = 500;            % Number of MC samples (reduced due to computational cost)

% Sigma-point configuration
% NOTE: alpha should be chosen so that (n + lambda) > 0
% For n=4: lambda = alpha^2*(n+kappa) - n, need lambda > -n, so alpha^2*(n+kappa) > 0
% Using alpha=1 with kappa=0 gives lambda=0, so (n+lambda)=n > 0
alpha_UT = 1;          % Spread parameter (1 is standard, 1e-3 causes numerical issues)
beta_UT = 2;           % Prior distribution (2 for Gaussian)
kappa_UT = 0;          % Secondary scaling (some use 3-n for Gaussian)

fprintf('Operating conditions:\n');
fprintf('  Temperature: %.1f K (%.1f °C)\n', T0, T0-273.15);
fprintf('  Pressure: %.1f bar\n', P0);
fprintf('  Flow rate: %.2e m³/s\n', F0);
fprintf('  Extraction time: %.0f min\n', ExtractionTime);
fprintf('  Monte Carlo samples: %d\n', N_MC);
fprintf('\n');

%% ========================================================================
%  LOAD PARAMETERS AND COVARIANCE MATRICES
%  ========================================================================
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
theta_power = [1.222524; 4.308414; 0.972739; 3.428618];  % [k_w0, a_w, b_w, n_k]
theta_Di = [0.19; -8.188; 0.62];                          % [a, b, c] for D_i
theta_Upsilon = [3.158; 11.922; -0.6868];                 % [a, b, c] for Upsilon
theta_linear = [theta_Di; theta_Upsilon];                 % Combined 6x1 vector

fprintf('Nominal parameters:\n');
fprintf('Power model: k_w0=%.4f, a_w=%.4f, b_w=%.4f, n_k=%.4f\n', theta_power);
fprintf('Linear model: D_i=[%.4f, %.4f, %.4f], Y=[%.4f, %.4f, %.4f]\n', theta_linear);
fprintf('\n');

%% ========================================================================
%  SETUP SIMULATION INFRASTRUCTURE
%  ========================================================================
fprintf('Setting up simulation infrastructure...\n');

% Physical parameters
m_total = 3.0;  % Total mass [g]
before = 0.04;
bed = 0.92;

% Time configuration
Time_in_sec = (timeStep:timeStep:ExtractionTime) * 60;
Time = [0 Time_in_sec/60];
N_Time = length(Time_in_sec);
timeStep_in_sec = timeStep * 60;

% Extractor geometry
nstages = Parameters{1};
r = Parameters{3};
epsi = Parameters{4};
L = Parameters{6};
dp = Parameters{5};

nstagesbefore = 1:floor(before * nstages);
nstagesbed = nstagesbefore(end)+1 : nstagesbefore(end) + floor(bed * nstages);
nstagesafter = nstagesbed(end)+1 : nstages;

bed_mask = nan(nstages, 1);
bed_mask(nstagesbefore) = 0;
bed_mask(nstagesbed) = 1;
bed_mask(nstagesafter) = 0;

% Volume calculations
V_slice = (L/nstages) * pi * r^2;
V_bed = V_slice * numel(nstagesbed);

V_before_fluid = repmat(V_slice * numel(nstagesbefore) / numel(nstagesbefore), numel(nstagesbefore), 1);
V_bed_fluid = repmat(V_bed * (1-epsi) / numel(nstagesbed), numel(nstagesbed), 1);
V_after_fluid = repmat(V_slice * numel(nstagesafter) / numel(nstagesafter), numel(nstagesafter), 1);
V_fluid = [V_before_fluid; V_bed_fluid; V_after_fluid];

L_bed_after_nstages = linspace(0, L, nstages);
L_bed_after_nstages = L_bed_after_nstages(nstagesbed(1):end);
L_bed_after_nstages = L_bed_after_nstages - L_bed_after_nstages(1);

% Initial conditions
C0solid = m_total * 1e-3 / (V_bed * epsi);
Parameters{2} = C0solid;

m_fluid = zeros(1, nstages);
C0fluid = m_fluid * 1e-3 ./ V_fluid';

epsi_mask = epsi .* bed_mask;
one_minus_epsi_mask = 1 - epsi_mask;

Nx = 3 * nstages + 2;
Nu = 3 + numel(Parameters);

% Compute fluid properties
Z = Compressibility(T0, P0, Parameters);
rho = rhoPB_Comp(T0, P0, Z, Parameters);
enthalpy_rho = rho * SpecificEnthalpy(T0, P0, Z, rho, Parameters);
MU = Viscosity(T0, rho);

% Reynolds number
V_superficial = F0 / (pi * r^2);
Re = dp * rho * V_superficial / MU * 1.3;

% Initial state
x0 = [C0fluid'; C0solid * bed_mask; enthalpy_rho * ones(nstages, 1); P0; 0];

% Input vectors
feedTemp = T0 * ones(1, N_Time);
feedPress = P0 * ones(1, N_Time);
feedFlow = F0 * ones(1, N_Time);
uu = [feedTemp', feedPress', feedFlow'];
U_base = [uu'; repmat(cell2mat(Parameters), 1, N_Time)];

fprintf('  Fluid density: %.2f kg/m³\n', rho);
fprintf('  Reynolds number: %.4f\n', Re);
fprintf('\n');

%% ========================================================================
%  NOMINAL SIMULATION (using original hardcoded models)
%  ========================================================================
fprintf('Running nominal simulations with original models...\n');

% Build nominal integrators (using hardcoded parameters in original files)
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

Y_power_trajectory = full([0, X_power_nom(Nx, :)]);
Y_linear_trajectory = full([0, X_linear_nom(Nx, :)]);

Y_power_final = Y_power_trajectory(end);
Y_linear_final = Y_linear_trajectory(end);

fprintf('Nominal results:\n');
fprintf('  Power model final yield:  %.6f g\n', Y_power_final);
fprintf('  Linear model final yield: %.6f g\n', Y_linear_final);
fprintf('  Nominal difference: %.6f g\n', Y_power_final - Y_linear_final);
fprintf('\n');

%% ========================================================================
%  METHOD 1: DELTA METHOD (Analytical Approximation)
%  ========================================================================
fprintf('=== METHOD 1: DELTA METHOD ===\n\n');

% Power Model Jacobian - Numerical finite difference
fprintf('  Computing Power model Jacobian via finite differences...\n');

delta_fd = 1e-4;  % Relative perturbation size
J_power = zeros(1, 4);
theta_pow_nom = theta_power;

for j = 1:4
    % Perturb parameter j
    theta_plus = theta_pow_nom;
    theta_minus = theta_pow_nom;
    h = max(abs(theta_pow_nom(j)) * delta_fd, 1e-8);
    theta_plus(j) = theta_pow_nom(j) + h;
    theta_minus(j) = theta_pow_nom(j) - h;

    % Evaluate model at perturbed parameters
    try
        f_plus = @(x, u) modelSFE_power_with_params(x, u, bed_mask, timeStep_in_sec, ...
            epsi_mask, one_minus_epsi_mask, theta_plus(1), theta_plus(2), theta_plus(3), theta_plus(4));
        F_plus = buildIntegrator(f_plus, [Nx, Nu], timeStep_in_sec, 'cvodes');
        F_accum_plus = F_plus.mapaccum('F_accum_plus', N_Time);
        X_plus = F_accum_plus(x0, U_base);
        Y_plus = full(X_plus(Nx, end));

        f_minus = @(x, u) modelSFE_power_with_params(x, u, bed_mask, timeStep_in_sec, ...
            epsi_mask, one_minus_epsi_mask, theta_minus(1), theta_minus(2), theta_minus(3), theta_minus(4));
        F_minus = buildIntegrator(f_minus, [Nx, Nu], timeStep_in_sec, 'cvodes');
        F_accum_minus = F_minus.mapaccum('F_accum_minus', N_Time);
        X_minus = F_accum_minus(x0, U_base);
        Y_minus = full(X_minus(Nx, end));

        J_power(j) = (Y_plus - Y_minus) / (2 * h);
    catch
        J_power(j) = 0;  % If evaluation fails, assume zero sensitivity
    end
end

Var_Y_power_delta = J_power * Cov_power * J_power';
Std_Y_power_delta = sqrt(max(Var_Y_power_delta, 0));

fprintf('  Power Jacobian: [%.4f, %.4f, %.4f, %.4f]\n', J_power);
fprintf('Power model (Delta method):\n');
fprintf('  Std(Y) = %.6f g, CV = %.2f%%\n', Std_Y_power_delta, 100*Std_Y_power_delta/Y_power_final);

% Linear Model Jacobian - Numerical finite difference
% Compute sensitivities using small perturbations of the nominal model
fprintf('  Computing Linear model Jacobian via finite differences...\n');

delta_fd = 1e-4;  % Relative perturbation size
J_linear = zeros(1, 6);
theta_lin_nom = theta_linear;

for j = 1:6
    % Perturb parameter j
    theta_plus = theta_lin_nom;
    theta_minus = theta_lin_nom;
    h = max(abs(theta_lin_nom(j)) * delta_fd, 1e-8);
    theta_plus(j) = theta_lin_nom(j) + h;
    theta_minus(j) = theta_lin_nom(j) - h;

    % Evaluate model at perturbed parameters
    try
        f_plus = @(x, u) modelSFE_linear_with_params(x, u, bed_mask, timeStep_in_sec, ...
            epsi_mask, one_minus_epsi_mask, theta_plus(1), theta_plus(2), theta_plus(3), ...
            theta_plus(4), theta_plus(5), theta_plus(6));
        F_plus = buildIntegrator(f_plus, [Nx, Nu], timeStep_in_sec, 'cvodes');
        F_accum_plus = F_plus.mapaccum('F_accum_plus', N_Time);
        X_plus = F_accum_plus(x0, U_base);
        Y_plus = full(X_plus(Nx, end));

        f_minus = @(x, u) modelSFE_linear_with_params(x, u, bed_mask, timeStep_in_sec, ...
            epsi_mask, one_minus_epsi_mask, theta_minus(1), theta_minus(2), theta_minus(3), ...
            theta_minus(4), theta_minus(5), theta_minus(6));
        F_minus = buildIntegrator(f_minus, [Nx, Nu], timeStep_in_sec, 'cvodes');
        F_accum_minus = F_minus.mapaccum('F_accum_minus', N_Time);
        X_minus = F_accum_minus(x0, U_base);
        Y_minus = full(X_minus(Nx, end));

        J_linear(j) = (Y_plus - Y_minus) / (2 * h);
    catch
        J_linear(j) = 0;  % If evaluation fails, assume zero sensitivity
    end
end

Var_Y_linear_delta = J_linear * Cov_linear_full * J_linear';
Std_Y_linear_delta = sqrt(max(Var_Y_linear_delta, 0));

fprintf('  Linear Jacobian: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n', J_linear);
fprintf('Linear model (Delta method):\n');
fprintf('  Std(Y) = %.6f g, CV = %.2f%%\n', Std_Y_linear_delta, 100*Std_Y_linear_delta/Y_linear_final);

% Discrimination
Std_pooled_delta = sqrt(Var_Y_power_delta + Var_Y_linear_delta);
t_stat_delta = abs(Y_power_final - Y_linear_final) / Std_pooled_delta;
CI_diff_delta = [Y_power_final - Y_linear_final - 1.96*Std_pooled_delta, ...
                 Y_power_final - Y_linear_final + 1.96*Std_pooled_delta];

fprintf('Discrimination: t = %.4f, 95%% CI = [%.6f, %.6f]\n\n', t_stat_delta, CI_diff_delta);

%% ========================================================================
%  METHOD 2: MONTE CARLO WITH FULL MODELS
%  ========================================================================
fprintf('=== METHOD 2: MONTE CARLO (Full Models) ===\n\n');
fprintf('Running %d Monte Carlo samples with FULL CasADi models...\n', N_MC);
fprintf('(This may take several minutes)\n\n');

% Cholesky decomposition
L_power_chol = chol(Cov_power + 1e-10*eye(4), 'lower');
L_linear_chol = chol(Cov_linear_full + 1e-10*eye(6), 'lower');

% Generate parameter samples
Z_power_samples = randn(4, N_MC);
Z_linear_samples = randn(6, N_MC);

theta_power_samples = repmat(theta_power, 1, N_MC) + L_power_chol * Z_power_samples;
theta_linear_samples = repmat(theta_linear, 1, N_MC) + L_linear_chol * Z_linear_samples;

% Storage
Y_power_MC = zeros(1, N_MC);
Y_linear_MC = zeros(1, N_MC);
valid_power = true(1, N_MC);
valid_linear = true(1, N_MC);

% Progress tracking
fprintf('Progress: ');
progress_step = max(1, floor(N_MC/20));

tic;
for i = 1:N_MC
    % Progress indicator
    if mod(i, progress_step) == 0
        fprintf('.');
    end

    % --- Power model with sampled parameters ---
    theta_p = theta_power_samples(:, i);
    try
        % Build model with these specific parameters
        f_power_i = @(x, u) modelSFE_power_with_params(x, u, bed_mask, timeStep_in_sec, ...
            epsi_mask, one_minus_epsi_mask, theta_p(1), theta_p(2), theta_p(3), theta_p(4));

        F_power_i = buildIntegrator(f_power_i, [Nx, Nu], timeStep_in_sec, 'cvodes');
        F_accum_power_i = F_power_i.mapaccum('F_accum', N_Time);

        X_power_i = F_accum_power_i(x0, U_base);
        Y_power_MC(i) = full(X_power_i(Nx, end));

        if isnan(Y_power_MC(i)) || Y_power_MC(i) < 0 || Y_power_MC(i) > 10
            valid_power(i) = false;
        end
    catch
        valid_power(i) = false;
        Y_power_MC(i) = NaN;
    end

    % --- Linear model with sampled parameters ---
    theta_l = theta_linear_samples(:, i);
    try
        f_linear_i = @(x, u) modelSFE_linear_with_params(x, u, bed_mask, timeStep_in_sec, ...
            epsi_mask, one_minus_epsi_mask, theta_l(1), theta_l(2), theta_l(3), ...
            theta_l(4), theta_l(5), theta_l(6));

        F_linear_i = buildIntegrator(f_linear_i, [Nx, Nu], timeStep_in_sec, 'cvodes');
        F_accum_linear_i = F_linear_i.mapaccum('F_accum', N_Time);

        X_linear_i = F_accum_linear_i(x0, U_base);
        Y_linear_MC(i) = full(X_linear_i(Nx, end));

        if isnan(Y_linear_MC(i)) || Y_linear_MC(i) < 0 || Y_linear_MC(i) > 10
            valid_linear(i) = false;
        end
    catch
        valid_linear(i) = false;
        Y_linear_MC(i) = NaN;
    end
end
mc_time = toc;
fprintf(' Done!\n');

% Filter valid samples
valid_both = valid_power & valid_linear;
Y_power_MC_valid = Y_power_MC(valid_both);
Y_linear_MC_valid = Y_linear_MC(valid_both);
n_valid = sum(valid_both);

fprintf('\nMC completed in %.1f seconds\n', mc_time);
fprintf('Valid samples: %d/%d (%.1f%%)\n\n', n_valid, N_MC, 100*n_valid/N_MC);

% Statistics
if n_valid > 1
    mean_power_MC = mean(Y_power_MC_valid);
    std_power_MC = std(Y_power_MC_valid);
    mean_linear_MC = mean(Y_linear_MC_valid);
    std_linear_MC = std(Y_linear_MC_valid);

    fprintf('Power model (MC with full model):\n');
    fprintf('  Mean = %.6f g, Std = %.6f g, CV = %.2f%%\n', mean_power_MC, std_power_MC, 100*std_power_MC/mean_power_MC);
    fprintf('  95%% CI: [%.6f, %.6f]\n', prctile(Y_power_MC_valid, 2.5), prctile(Y_power_MC_valid, 97.5));

    fprintf('Linear model (MC with full model):\n');
    fprintf('  Mean = %.6f g, Std = %.6f g, CV = %.2f%%\n', mean_linear_MC, std_linear_MC, 100*std_linear_MC/mean_linear_MC);
    fprintf('  95%% CI: [%.6f, %.6f]\n', prctile(Y_linear_MC_valid, 2.5), prctile(Y_linear_MC_valid, 97.5));

    % Difference statistics
    diff_MC = Y_power_MC_valid - Y_linear_MC_valid;
    mean_diff_MC = mean(diff_MC);
    std_diff_MC = std(diff_MC);
    CI_diff_MC = [prctile(diff_MC, 2.5), prctile(diff_MC, 97.5)];
    prob_power_greater = mean(diff_MC > 0);

    fprintf('\nDiscrimination (MC):\n');
    fprintf('  Mean diff = %.6f g, Std = %.6f g\n', mean_diff_MC, std_diff_MC);
    fprintf('  95%% CI: [%.6f, %.6f]\n', CI_diff_MC);
    fprintf('  P(Power > Linear) = %.1f%%\n', 100*prob_power_greater);
else
    warning('Insufficient valid MC samples');
    mean_power_MC = Y_power_final;
    std_power_MC = NaN;
    mean_linear_MC = Y_linear_final;
    std_linear_MC = NaN;
    mean_diff_MC = Y_power_final - Y_linear_final;
    std_diff_MC = NaN;
    CI_diff_MC = [NaN, NaN];
    prob_power_greater = NaN;
    diff_MC = [];
end
fprintf('\n');

%% ========================================================================
%  METHOD 3: SIGMA-POINT TRANSFORM WITH FULL MODELS
%  ========================================================================
fprintf('=== METHOD 3: SIGMA-POINT TRANSFORM (Full Models) ===\n\n');

% --- Power Model (4 parameters -> 9 sigma points) ---
n_p = 4;
lambda_p = alpha_UT^2 * (n_p + kappa_UT) - n_p;
gamma_p = sqrt(n_p + lambda_p);

S_power = chol(Cov_power + 1e-10*eye(4), 'lower');

sigma_power = zeros(4, 2*n_p + 1);
sigma_power(:, 1) = theta_power;
for i = 1:n_p
    sigma_power(:, i+1) = theta_power + gamma_p * S_power(:, i);
    sigma_power(:, n_p+i+1) = theta_power - gamma_p * S_power(:, i);
end

% Weights
W_m_p = zeros(2*n_p + 1, 1);
W_c_p = zeros(2*n_p + 1, 1);
W_m_p(1) = lambda_p / (n_p + lambda_p);
W_c_p(1) = lambda_p / (n_p + lambda_p) + (1 - alpha_UT^2 + beta_UT);
W_m_p(2:end) = 1 / (2*(n_p + lambda_p));
W_c_p(2:end) = 1 / (2*(n_p + lambda_p));

fprintf('Power UT: lambda=%.4f, gamma=%.4f, W_m(1)=%.4f, W_m(other)=%.4f, sum(W_m)=%.4f\n', ...
    lambda_p, gamma_p, W_m_p(1), W_m_p(2), sum(W_m_p));
fprintf('Evaluating Power model at %d sigma points...\n', 2*n_p+1);
Y_sigma_power = zeros(1, 2*n_p + 1);

tic;
for i = 1:(2*n_p + 1)
    theta_p = sigma_power(:, i);
    try
        f_power_i = @(x, u) modelSFE_power_with_params(x, u, bed_mask, timeStep_in_sec, ...
            epsi_mask, one_minus_epsi_mask, theta_p(1), theta_p(2), theta_p(3), theta_p(4));

        F_power_i = buildIntegrator(f_power_i, [Nx, Nu], timeStep_in_sec, 'cvodes');
        F_accum_i = F_power_i.mapaccum('F_accum', N_Time);

        X_i = F_accum_i(x0, U_base);
        Y_sigma_power(i) = full(X_i(Nx, end));
    catch ME
        Y_sigma_power(i) = Y_power_final;  % Fallback to nominal
        warning('Sigma point %d failed for Power model: %s', i, ME.message);
    end
    fprintf('  Point %d: theta = [%.4f, %.4f, %.4f, %.4f] -> Y = %.6f\n', i, theta_p, Y_sigma_power(i));
end

mean_power_sigma = sum(W_m_p .* Y_sigma_power');
var_power_sigma = sum(W_c_p .* (Y_sigma_power' - mean_power_sigma).^2);
std_power_sigma = sqrt(max(var_power_sigma, 0));

fprintf('Power model (Sigma-point): Mean = %.6f, Std = %.6f\n\n', mean_power_sigma, std_power_sigma);

% --- Linear Model (6 parameters -> 13 sigma points) ---
n_l = 6;
lambda_l = alpha_UT^2 * (n_l + kappa_UT) - n_l;
gamma_l = sqrt(n_l + lambda_l);

S_linear = chol(Cov_linear_full + 1e-10*eye(6), 'lower');

sigma_linear = zeros(6, 2*n_l + 1);
sigma_linear(:, 1) = theta_linear;
for i = 1:n_l
    sigma_linear(:, i+1) = theta_linear + gamma_l * S_linear(:, i);
    sigma_linear(:, n_l+i+1) = theta_linear - gamma_l * S_linear(:, i);
end

W_m_l = zeros(2*n_l + 1, 1);
W_c_l = zeros(2*n_l + 1, 1);
W_m_l(1) = lambda_l / (n_l + lambda_l);
W_c_l(1) = lambda_l / (n_l + lambda_l) + (1 - alpha_UT^2 + beta_UT);
W_m_l(2:end) = 1 / (2*(n_l + lambda_l));
W_c_l(2:end) = 1 / (2*(n_l + lambda_l));

fprintf('Linear UT: lambda=%.4f, gamma=%.4f, W_m(1)=%.4f, W_m(other)=%.4f, sum(W_m)=%.4f\n', ...
    lambda_l, gamma_l, W_m_l(1), W_m_l(2), sum(W_m_l));
fprintf('Evaluating Linear model at %d sigma points...\n', 2*n_l+1);
Y_sigma_linear = zeros(1, 2*n_l + 1);

for i = 1:(2*n_l + 1)
    theta_l = sigma_linear(:, i);
    try
        f_linear_i = @(x, u) modelSFE_linear_with_params(x, u, bed_mask, timeStep_in_sec, ...
            epsi_mask, one_minus_epsi_mask, theta_l(1), theta_l(2), theta_l(3), ...
            theta_l(4), theta_l(5), theta_l(6));

        F_linear_i = buildIntegrator(f_linear_i, [Nx, Nu], timeStep_in_sec, 'cvodes');
        F_accum_i = F_linear_i.mapaccum('F_accum', N_Time);

        X_i = F_accum_i(x0, U_base);
        Y_sigma_linear(i) = full(X_i(Nx, end));
    catch ME
        Y_sigma_linear(i) = Y_linear_final;
        warning('Sigma point %d failed for Linear model: %s', i, ME.message);
    end
    fprintf('  Point %d: Y = %.6f\n', i, Y_sigma_linear(i));
end
sigma_time = toc;

mean_linear_sigma = sum(W_m_l .* Y_sigma_linear');
var_linear_sigma = sum(W_c_l .* (Y_sigma_linear' - mean_linear_sigma).^2);
std_linear_sigma = sqrt(max(var_linear_sigma, 0));

fprintf('Linear model (Sigma-point): Mean = %.6f, Std = %.6f\n', mean_linear_sigma, std_linear_sigma);
fprintf('Sigma-point evaluation completed in %.1f seconds\n\n', sigma_time);

% Discrimination
std_pooled_sigma = sqrt(var_power_sigma + var_linear_sigma);
diff_sigma = mean_power_sigma - mean_linear_sigma;
t_stat_sigma = abs(diff_sigma) / max(std_pooled_sigma, 1e-10);
CI_diff_sigma = [diff_sigma - 1.96*std_pooled_sigma, diff_sigma + 1.96*std_pooled_sigma];

fprintf('Discrimination (Sigma-point):\n');
fprintf('  Mean diff = %.6f g, Pooled Std = %.6f g\n', diff_sigma, std_pooled_sigma);
fprintf('  t-statistic = %.4f\n', t_stat_sigma);
fprintf('  95%% CI: [%.6f, %.6f]\n', CI_diff_sigma);
fprintf('\n');

%% ========================================================================
%  COMPARISON OF ALL METHODS
%  ========================================================================
fprintf('=============================================================================\n');
fprintf('                         COMPARISON OF METHODS                               \n');
fprintf('=============================================================================\n\n');

fprintf('%-22s %-14s %-14s %-14s\n', '', 'Delta', 'MC (Full)', 'Sigma (Full)');
fprintf('-----------------------------------------------------------------------------\n');
fprintf('%-22s %-14.6f %-14.6f %-14.6f\n', 'Power Mean [g]', Y_power_final, mean_power_MC, mean_power_sigma);
fprintf('%-22s %-14.6f %-14.6f %-14.6f\n', 'Power Std [g]', Std_Y_power_delta, std_power_MC, std_power_sigma);
fprintf('%-22s %-14.6f %-14.6f %-14.6f\n', 'Linear Mean [g]', Y_linear_final, mean_linear_MC, mean_linear_sigma);
fprintf('%-22s %-14.6f %-14.6f %-14.6f\n', 'Linear Std [g]', Std_Y_linear_delta, std_linear_MC, std_linear_sigma);
fprintf('-----------------------------------------------------------------------------\n');
fprintf('%-22s %-14.6f %-14.6f %-14.6f\n', 'Difference [g]', Y_power_final-Y_linear_final, mean_diff_MC, diff_sigma);
fprintf('%-22s %-14.6f %-14.6f %-14.6f\n', 'Pooled Std [g]', Std_pooled_delta, std_diff_MC, std_pooled_sigma);
fprintf('%-22s %-14.4f %-14s %-14.4f\n', 't-statistic', t_stat_delta, 'N/A', t_stat_sigma);
if ~isnan(prob_power_greater)
    fprintf('%-22s %-14s %-14.1f%% %-14s\n', 'P(Power>Linear)', 'N/A', 100*prob_power_greater, 'N/A');
end
fprintf('=============================================================================\n\n');

%% ========================================================================
%  VISUALIZATION
%  ========================================================================
fprintf('Generating visualizations...\n');

figure('Name', 'Full Model Uncertainty Analysis', 'Position', [100 100 1400 900]);

% Panel 1: MC distributions (full model)
subplot(2, 3, 1);
if n_valid > 1
    histogram(Y_power_MC_valid, 25, 'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.6);
    hold on;
    histogram(Y_linear_MC_valid, 25, 'Normalization', 'pdf', 'FaceColor', 'r', 'FaceAlpha', 0.6);
end
xline(Y_power_final, 'b--', 'LineWidth', 2);
xline(Y_linear_final, 'r--', 'LineWidth', 2);
xlabel('Final Yield [g]');
ylabel('Probability Density');
title('MC Output Distributions (Full Models)');
legend('Power', 'Linear', 'Location', 'best');
grid on;

% Panel 2: Difference distribution
subplot(2, 3, 2);
if n_valid > 1 && ~isempty(diff_MC)
    histogram(diff_MC, 25, 'Normalization', 'pdf', 'FaceColor', [0.5 0 0.5], 'FaceAlpha', 0.7);
    hold on;
    xline(0, 'k--', 'LineWidth', 2);
    xline(mean_diff_MC, 'g-', 'LineWidth', 2);
    if ~isnan(CI_diff_MC(1))
        xline(CI_diff_MC(1), 'm:', 'LineWidth', 1.5);
        xline(CI_diff_MC(2), 'm:', 'LineWidth', 1.5);
    end
end
xlabel('Difference (Power - Linear) [g]');
ylabel('Probability Density');
if ~isnan(prob_power_greater)
    title(sprintf('Difference Distribution, P(D>0) = %.1f%%', 100*prob_power_greater), 'Interpreter', 'none');
else
    title('Difference Distribution');
end
grid on;

% Panel 3: Sigma points
subplot(2, 3, 3);
scatter(1:(2*n_p+1), Y_sigma_power, 100, 'b', 'filled');
hold on;
scatter((2*n_p+2):(2*n_p+1+2*n_l+1), Y_sigma_linear, 100, 'r', 'filled');
yline(Y_power_final, 'b--', 'LineWidth', 1.5);
yline(Y_linear_final, 'r--', 'LineWidth', 1.5);
xlabel('Sigma Point Index');
ylabel('Yield [g]');
title('Sigma Point Evaluations (Full Models)');
legend('Power', 'Linear', 'Nom Power', 'Nom Linear', 'Location', 'best');
grid on;

% Panel 4: Method comparison
subplot(2, 3, 4);
methods = categorical({'Delta', 'MC (Full)', 'Sigma (Full)'});
methods = reordercats(methods, {'Delta', 'MC (Full)', 'Sigma (Full)'});
std_power_all = [Std_Y_power_delta, std_power_MC, std_power_sigma];
std_linear_all = [Std_Y_linear_delta, std_linear_MC, std_linear_sigma];
bar(methods, [std_power_all; std_linear_all]');
ylabel('Standard Deviation [g]');
title('Uncertainty Comparison by Method');
legend('Power', 'Linear', 'Location', 'best');
grid on;

% Panel 5: Confidence intervals comparison
subplot(2, 3, 5);
y_pos = [1, 2, 3];  % Must be increasing for yticks
hold on;
errorbar(Y_power_final - Y_linear_final, y_pos(3), ...
    CI_diff_delta(1)-(Y_power_final-Y_linear_final), ...
    CI_diff_delta(2)-(Y_power_final-Y_linear_final), ...
    'horizontal', 'bo', 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'Delta');
if ~isnan(CI_diff_MC(1))
    errorbar(mean_diff_MC, y_pos(2), ...
        CI_diff_MC(1)-mean_diff_MC, CI_diff_MC(2)-mean_diff_MC, ...
        'horizontal', 'rs', 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'MC (Full)');
end
errorbar(diff_sigma, y_pos(1), ...
    CI_diff_sigma(1)-diff_sigma, CI_diff_sigma(2)-diff_sigma, ...
    'horizontal', 'g^', 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'Sigma (Full)');
xline(0, 'k--', 'LineWidth', 1.5);
yticks(y_pos);
yticklabels({'Sigma (Full)', 'MC (Full)', 'Delta'});
xlabel('Difference (Power - Linear) [g]');
title('95% Confidence Intervals');
legend('Location', 'best');
grid on;

% Panel 6: Extraction trajectories
subplot(2, 3, 6);
plot(Time, Y_power_trajectory, 'b-', 'LineWidth', 2);
hold on;
plot(Time, Y_linear_trajectory, 'r-', 'LineWidth', 2);
if ~isnan(std_power_MC)
    fill([Time, fliplr(Time)], ...
         [Y_power_trajectory + 1.96*std_power_MC, fliplr(Y_power_trajectory - 1.96*std_power_MC)], ...
         'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    fill([Time, fliplr(Time)], ...
         [Y_linear_trajectory + 1.96*std_linear_MC, fliplr(Y_linear_trajectory - 1.96*std_linear_MC)], ...
         'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end
xlabel('Time [min]');
ylabel('Cumulative Yield [g]');
title('Extraction Curves with MC Uncertainty');
legend('Power', 'Linear', 'Location', 'southeast');
grid on;

sgtitle(sprintf('Full Model Uncertainty Analysis: T=%.0fK, P=%.0fbar, F=%.2em³/s', T0, P0, F0), 'FontSize', 14);

%% ========================================================================
%  SAVE RESULTS
%  ========================================================================
results = struct();
results.T = T0; results.P = P0; results.F = F0;
results.ExtractionTime = ExtractionTime;
results.rho = rho; results.Re = Re;
results.Y_power_final = Y_power_final;
results.Y_linear_final = Y_linear_final;
results.Time = Time;
results.Y_power_trajectory = Y_power_trajectory;
results.Y_linear_trajectory = Y_linear_trajectory;

% Delta results
results.delta.Std_power = Std_Y_power_delta;
results.delta.Std_linear = Std_Y_linear_delta;
results.delta.t_stat = t_stat_delta;
results.delta.CI_diff = CI_diff_delta;

% MC results (full model)
results.MC.mean_power = mean_power_MC;
results.MC.std_power = std_power_MC;
results.MC.mean_linear = mean_linear_MC;
results.MC.std_linear = std_linear_MC;
results.MC.mean_diff = mean_diff_MC;
results.MC.std_diff = std_diff_MC;
results.MC.CI_diff = CI_diff_MC;
results.MC.prob_power_greater = prob_power_greater;
results.MC.Y_power_samples = Y_power_MC_valid;
results.MC.Y_linear_samples = Y_linear_MC_valid;
results.MC.n_valid = n_valid;
results.MC.computation_time = mc_time;

% Sigma results (full model)
results.sigma.mean_power = mean_power_sigma;
results.sigma.std_power = std_power_sigma;
results.sigma.mean_linear = mean_linear_sigma;
results.sigma.std_linear = std_linear_sigma;
results.sigma.t_stat = t_stat_sigma;
results.sigma.CI_diff = CI_diff_sigma;
results.sigma.Y_sigma_power = Y_sigma_power;
results.sigma.Y_sigma_linear = Y_sigma_linear;
results.sigma.computation_time = sigma_time;

% Parameters
results.Cov_power = Cov_power;
results.Cov_linear_full = Cov_linear_full;
results.theta_power = theta_power;
results.theta_linear = theta_linear;

save('single_point_discrimination_full_results.mat', 'results');
fprintf('\nResults saved to single_point_discrimination_full_results.mat\n');

%% ========================================================================
%  FINAL SUMMARY
%  ========================================================================
fprintf('\n=============================================================================\n');
fprintf('                              FINAL SUMMARY                                  \n');
fprintf('=============================================================================\n\n');

fprintf('Operating point: T=%.0fK, P=%.0fbar, F=%.2em³/s\n\n', T0, P0, F0);
fprintf('Nominal difference: %.6f g\n\n', Y_power_final - Y_linear_final);

fprintf('Statistical significance:\n');
fprintf('  Delta:        t=%.3f -> %s\n', t_stat_delta, get_sig_label(t_stat_delta));
if ~isnan(CI_diff_MC(1))
    fprintf('  MC (Full):    95%%CI=[%.4f,%.4f] -> %s\n', CI_diff_MC, get_ci_label(CI_diff_MC));
end
fprintf('  Sigma (Full): t=%.3f -> %s\n', t_stat_sigma, get_sig_label(t_stat_sigma));
fprintf('\n');

fprintf('Method agreement: ');
mc_sig = ~isnan(CI_diff_MC(1)) && (CI_diff_MC(1) > 0 || CI_diff_MC(2) < 0);
mc_not = ~isnan(CI_diff_MC(1)) && (CI_diff_MC(1) < 0 && CI_diff_MC(2) > 0);
all_sig = (t_stat_delta > 1.96) && mc_sig && (t_stat_sigma > 1.96);
all_not = (t_stat_delta < 1.96) && mc_not && (t_stat_sigma < 1.96);
if all_sig
    fprintf('ALL methods agree - SIGNIFICANT\n');
elseif all_not
    fprintf('ALL methods agree - NOT significant\n');
else
    fprintf('Methods DISAGREE or inconclusive - marginal discrimination\n');
end

fprintf('\n=== Analysis Complete ===\n');

%% ========================================================================
%  LOCAL HELPER FUNCTIONS (must be at end of script)
%  ========================================================================
function label = get_sig_label(t)
    if t > 2.58
        label = 'HIGHLY SIGNIFICANT';
    elseif t > 1.96
        label = 'SIGNIFICANT';
    elseif t > 1.64
        label = 'MARGINAL';
    else
        label = 'NOT SIGNIFICANT';
    end
end

function label = get_ci_label(CI)
    if CI(1) > 0 || CI(2) < 0
        label = 'SIGNIFICANT';
    else
        label = 'NOT SIGNIFICANT';
    end
end
