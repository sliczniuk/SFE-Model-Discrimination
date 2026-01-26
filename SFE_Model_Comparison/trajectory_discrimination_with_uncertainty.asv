%% Trajectory-Based Model Discrimination with Parameter Uncertainty
% This script evaluates model discrimination at a FIXED operating point (T, P, F)
% while propagating kinetic parameter uncertainty through Monte Carlo sampling.
%
% For each MC sample:
%   - Sample Power model parameters from N(theta_power, Cov_power)
%   - Sample Linear model parameters from N(theta_linear, Cov_linear)
%   - Run full extraction trajectory
%   - Store yield trajectory Y(t)
%
% Then compute:
%   - Time-pointwise KL and JS divergence between output distributions
%   - Trajectory-integrated divergence metrics
%   - Confidence bands on trajectories
%   - Probability of model discrimination

%% Initialization
startup;

addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
import casadi.*

rng(42);  % Set seed for reproducibility

%% Start parallel pool if not already running
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local');
end
n_workers = pool.NumWorkers;
fprintf('Parallel pool started with %d workers\n', n_workers);

fprintf('=============================================================================\n');
fprintf('   TRAJECTORY DISCRIMINATION WITH PARAMETER UNCERTAINTY (FIXED T, P, F)     \n');
fprintf('=============================================================================\n\n');

%% ========================================================================
%  USER-CONFIGURABLE SETTINGS
%  ========================================================================
% Fixed operating conditions
T0 = 30+273;      % Temperature [K] (35 C)
P0 = 150;      % Pressure [bar]
F0 = 5e-5;     % Flow rate [m3/s]

% Time configuration
ExtractionTime = 600;  % Extraction time [minutes]
timeStep = 5;          % Time step [minutes]

% Monte Carlo configuration
N_MC = 500;            % Number of MC samples for parameter uncertainty

fprintf('Operating conditions (FIXED):\n');
fprintf('  Temperature: %.1f K (%.1f C)\n', T0, T0-273.15);
fprintf('  Pressure: %.1f bar\n', P0);
fprintf('  Flow rate: %.2e m3/s\n', F0);
fprintf('  Extraction time: %.0f min\n', ExtractionTime);
fprintf('  MC samples: %d\n', N_MC);
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
fprintf('  Power: k_w0=%.4f, a_w=%.4f, b_w=%.4f, n_k=%.4f\n', theta_power);
fprintf('  Linear: D_i=[%.4f, %.4f, %.4f], Ups=[%.4f, %.4f, %.4f]\n', theta_linear);
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

fprintf('  Time points: %d (every %d min up to %d min)\n', N_Time, timeStep, ExtractionTime);

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

% Compute fluid properties at operating conditions
Z = Compressibility(T0, P0, Parameters);
rho = rhoPB_Comp(T0, P0, Z, Parameters);
enthalpy_rho = rho * SpecificEnthalpy(T0, P0, Z, rho, Parameters);
MU = Viscosity(T0, rho);

% Reynolds number
%V_superficial = F0 / (pi * r^2);
%Re = dp * rho * V_superficial / MU * 1.3;

% Initial state
x0 = [C0fluid'; C0solid * bed_mask; enthalpy_rho * ones(nstages, 1); P0; 0];

% Input vectors (constant over time)
feedTemp = T0 * ones(1, N_Time);
feedPress = P0 * ones(1, N_Time);
feedFlow = F0 * ones(1, N_Time);
uu = [feedTemp', feedPress', feedFlow'];
U_base = [uu'; repmat(cell2mat(Parameters), 1, N_Time)];

%fprintf('  Fluid density: %.2f kg/m3\n', rho);
%fprintf('  Reynolds number: %.4f\n', Re);
%fprintf('\n');

%% ========================================================================
%  NOMINAL SIMULATION
%  ========================================================================
fprintf('Running nominal simulations...\n');

% Build nominal integrators
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

fprintf('  Nominal Power final yield:  %.6f g\n', Y_power_nom(end));
fprintf('  Nominal Linear final yield: %.6f g\n', Y_linear_nom(end));
fprintf('  Nominal difference: %.6f g\n', Y_power_nom(end) - Y_linear_nom(end));
fprintf('\n');

%% ========================================================================
%  MONTE CARLO SAMPLING WITH PARAMETER UNCERTAINTY (PARALLELIZED)
%  ========================================================================
fprintf('Running Monte Carlo simulations with parameter uncertainty...\n');
fprintf('  Sampling %d parameter realizations using %d workers...\n', N_MC, n_workers);

% Cholesky decomposition for sampling
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

% Storage for trajectories (use cell arrays for parfor compatibility)
Y_power_MC_cell = cell(N_MC, 1);
Y_linear_MC_cell = cell(N_MC, 1);
valid_power = true(N_MC, 1);
valid_linear = true(N_MC, 1);

% Extract variables needed inside parfor (avoid broadcast issues)
bed_mask_local = bed_mask;
epsi_mask_local = epsi_mask;
one_minus_epsi_mask_local = one_minus_epsi_mask;
timeStep_in_sec_local = timeStep_in_sec;
Nx_local = Nx;
Nu_local = Nu;
N_Time_local = N_Time;
x0_local = x0;
U_base_local = U_base;

fprintf('  Running parallel simulations...\n');
tic;

parfor i = 1:N_MC
    % --- Power model with sampled parameters ---
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

    % --- Linear model with sampled parameters ---
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
fprintf('  Parallel MC completed in %.1f seconds (%.2f samples/sec)\n', mc_time, N_MC/mc_time);

% Convert cell arrays back to matrices
Y_power_MC = cell2mat(Y_power_MC_cell);
Y_linear_MC = cell2mat(Y_linear_MC_cell);

% Filter valid samples
valid_both = valid_power & valid_linear;
Y_power_valid = Y_power_MC(valid_both, :);
Y_linear_valid = Y_linear_MC(valid_both, :);
n_valid = sum(valid_both);

fprintf('Valid samples: %d/%d (%.1f%%)\n', n_valid, N_MC, 100*n_valid/N_MC);
fprintf('Speedup estimate: ~%.1fx with %d workers\n', min(n_workers, N_MC/10), n_workers);

%% ========================================================================
%  COMPUTE TIME-POINTWISE DIVERGENCE METRICS
%  ========================================================================
fprintf('\nComputing divergence metrics over trajectory...\n');

Time_full = Time;  % Includes t=0
n_time_full = length(Time_full);

% Preallocate metrics
metrics = struct();
metrics.Time = Time_full;
metrics.T0 = T0;
metrics.P0 = P0;
metrics.F0 = F0;

% Time-pointwise metrics
metrics.kl_power_linear = zeros(1, n_time_full);  % KL(Power || Linear)
metrics.kl_linear_power = zeros(1, n_time_full);  % KL(Linear || Power)
metrics.js_divergence = zeros(1, n_time_full);    % JS divergence (symmetric)
metrics.ks_stat = zeros(1, n_time_full);
metrics.ks_pval = zeros(1, n_time_full);
metrics.mean_diff = zeros(1, n_time_full);
metrics.overlap_coef = zeros(1, n_time_full);     % Overlap coefficient

% Statistics
metrics.mean_power = zeros(1, n_time_full);
metrics.mean_linear = zeros(1, n_time_full);
metrics.std_power = zeros(1, n_time_full);
metrics.std_linear = zeros(1, n_time_full);
metrics.ci95_power = zeros(2, n_time_full);
metrics.ci95_linear = zeros(2, n_time_full);

for i_t = 1:n_time_full
    y_p = Y_power_valid(:, i_t);
    y_l = Y_linear_valid(:, i_t);

    % Basic statistics
    metrics.mean_power(i_t) = mean(y_p);
    metrics.mean_linear(i_t) = mean(y_l);
    metrics.std_power(i_t) = std(y_p);
    metrics.std_linear(i_t) = std(y_l);
    metrics.ci95_power(:, i_t) = [prctile(y_p, 2.5); prctile(y_p, 97.5)];
    metrics.ci95_linear(:, i_t) = [prctile(y_l, 2.5); prctile(y_l, 97.5)];

    % Mean difference
    metrics.mean_diff(i_t) = mean(y_p) - mean(y_l);

    % KS test
    [~, p_ks, ks_stat] = kstest2(y_p, y_l);
    metrics.ks_stat(i_t) = ks_stat;
    metrics.ks_pval(i_t) = p_ks;

    % Skip divergence computation at t=0 (all zeros)
    if i_t == 1 || (std(y_p) < 1e-10 && std(y_l) < 1e-10)
        continue;
    end

    % KDE-based divergence metrics
    [kl_pl, kl_lp, js, overlap] = compute_divergences(y_p, y_l);
    metrics.kl_power_linear(i_t) = kl_pl;
    metrics.kl_linear_power(i_t) = kl_lp;
    metrics.js_divergence(i_t) = js;
    metrics.overlap_coef(i_t) = overlap;
end

%% ========================================================================
%  COMPUTE INTEGRATED METRICS
%  ========================================================================
fprintf('Computing integrated metrics...\n');

% Exclude t=0 for integration (yields are all zero)
Time_int = Time_full(2:end);

metrics.kl_integrated = trapz(Time_int, metrics.kl_power_linear(2:end));
metrics.js_integrated = trapz(Time_int, metrics.js_divergence(2:end));
metrics.ks_integrated = trapz(Time_int, metrics.ks_stat(2:end));

% Maximum values and their times
[metrics.js_max, idx_js_max] = max(metrics.js_divergence);
metrics.js_max_time = Time_full(idx_js_max);

[metrics.kl_max, idx_kl_max] = max(metrics.kl_power_linear);
metrics.kl_max_time = Time_full(idx_kl_max);

[metrics.ks_max, idx_ks_max] = max(metrics.ks_stat);
metrics.ks_max_time = Time_full(idx_ks_max);

% Probability that Power > Linear at each time point
metrics.prob_power_greater = mean(Y_power_valid > Y_linear_valid, 1);

% Final yield statistics
metrics.final_diff_mean = mean(Y_power_valid(:, end) - Y_linear_valid(:, end));
metrics.final_diff_std = std(Y_power_valid(:, end) - Y_linear_valid(:, end));
metrics.final_diff_ci95 = [prctile(Y_power_valid(:, end) - Y_linear_valid(:, end), 2.5), ...
                           prctile(Y_power_valid(:, end) - Y_linear_valid(:, end), 97.5)];

fprintf('\n');
fprintf('Integrated metrics:\n');
fprintf('  KL(P||L) integrated: %.4f nats*min\n', metrics.kl_integrated);
fprintf('  JS integrated: %.4f nats*min\n', metrics.js_integrated);
fprintf('  KS integrated: %.4f min\n', metrics.ks_integrated);
fprintf('\n');
fprintf('Maximum divergence:\n');
fprintf('  Max JS = %.4f at t = %.0f min\n', metrics.js_max, metrics.js_max_time);
fprintf('  Max KL = %.4f at t = %.0f min\n', metrics.kl_max, metrics.kl_max_time);
fprintf('  Max KS = %.4f at t = %.0f min\n', metrics.ks_max, metrics.ks_max_time);
fprintf('\n');
fprintf('Final yield difference:\n');
fprintf('  Mean: %.6f g\n', metrics.final_diff_mean);
fprintf('  Std:  %.6f g\n', metrics.final_diff_std);
fprintf('  95%% CI: [%.6f, %.6f] g\n', metrics.final_diff_ci95);

%% ========================================================================
%  VISUALIZATION
%  ========================================================================
fprintf('\nGenerating visualizations...\n');

%% Figure 1: Trajectory ensemble with confidence bands
%{
figure('Name', 'Trajectory Ensemble with Parameter Uncertainty', 'Position', [100 100 1200 800]);

subplot(2, 2, 1);
% Plot individual trajectories (subset for clarity)
n_plot = min(100, n_valid);
idx_plot = randperm(n_valid, n_plot);
for i = 1:n_plot
    plot(Time_full, Y_power_valid(idx_plot(i), :), 'b-', 'LineWidth', 0.3, 'Color', [0 0 1 0.1]);
    hold on;
end
plot(Time_full, metrics.mean_power, 'b-', 'LineWidth', 2, 'DisplayName', 'Mean Power');
plot(Time_full, Y_power_nom, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Nominal Power');
fill([Time_full, fliplr(Time_full)], [metrics.ci95_power(1,:), fliplr(metrics.ci95_power(2,:))], ...
    'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
xlabel('Time [min]');
ylabel('Yield [g]');
title('Power Model Trajectories');
legend('Location', 'southeast');
grid on;

subplot(2, 2, 2);
for i = 1:n_plot
    plot(Time_full, Y_linear_valid(idx_plot(i), :), 'r-', 'LineWidth', 0.3, 'Color', [1 0 0 0.1]);
    hold on;
end
plot(Time_full, metrics.mean_linear, 'r-', 'LineWidth', 2, 'DisplayName', 'Mean Linear');
plot(Time_full, Y_linear_nom, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Nominal Linear');
fill([Time_full, fliplr(Time_full)], [metrics.ci95_linear(1,:), fliplr(metrics.ci95_linear(2,:))], ...
    'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
xlabel('Time [min]');
ylabel('Yield [g]');
title('Linear Model Trajectories');
legend('Location', 'southeast');
grid on;

subplot(2, 2, 3);
fill([Time_full, fliplr(Time_full)], [metrics.ci95_power(1,:), fliplr(metrics.ci95_power(2,:))], ...
    'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Power 95% CI');
hold on;
fill([Time_full, fliplr(Time_full)], [metrics.ci95_linear(1,:), fliplr(metrics.ci95_linear(2,:))], ...
    'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Linear 95% CI');
plot(Time_full, metrics.mean_power, 'b-', 'LineWidth', 2, 'DisplayName', 'Mean Power');
plot(Time_full, metrics.mean_linear, 'r-', 'LineWidth', 2, 'DisplayName', 'Mean Linear');
xlabel('Time [min]');
ylabel('Yield [g]');
title('Mean Trajectories with 95% CI');
legend('Location', 'southeast');
grid on;

subplot(2, 2, 4);
diff_trajectories = Y_power_valid - Y_linear_valid;
mean_diff_traj = mean(diff_trajectories, 1);
ci_diff = [prctile(diff_trajectories, 2.5, 1); prctile(diff_trajectories, 97.5, 1)];

fill([Time_full, fliplr(Time_full)], [ci_diff(1,:), fliplr(ci_diff(2,:))], ...
    [0.5 0 0.5], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
hold on;
plot(Time_full, mean_diff_traj, 'm-', 'LineWidth', 2);
yline(0, 'k--', 'LineWidth', 1);
xlabel('Time [min]');
ylabel('Difference (Power - Linear) [g]');
title('Yield Difference with 95% CI');
grid on;

sgtitle(sprintf('Trajectory Ensemble: T=%.0fK, P=%.0fbar, F=%.1e m3/s (N=%d)', T0, P0, F0, n_valid), 'FontSize', 12);
%}
%% Figure 2: Time-pointwise divergence metrics
figure('Name', 'Divergence Metrics Over Time', 'Position', [150 150 1200 400]);

subplot(1, 3, 1);
plot(Time_full, metrics.js_divergence, 'k-', 'LineWidth', 2);
hold on;
xline(metrics.js_max_time, 'r--', sprintf('Max=%.3f', metrics.js_max));
xlabel('Time [min]');
ylabel('JS Divergence [nats]');
title('Jensen-Shannon Divergence');
grid on;

subplot(1, 3, 2);
plot(Time_full, metrics.kl_power_linear, 'b-', 'LineWidth', 2, 'DisplayName', 'KL(P||L)');
hold on;
plot(Time_full, metrics.kl_linear_power, 'r-', 'LineWidth', 2, 'DisplayName', 'KL(L||P)');
xlabel('Time [min]');
ylabel('KL Divergence [nats]');
title('KL Divergence');
legend('Location', 'best');
grid on;

subplot(1, 3, 3);
plot(Time_full, metrics.ks_stat, 'g-', 'LineWidth', 2);
hold on;
xline(metrics.ks_max_time, 'r--', sprintf('Max=%.3f', metrics.ks_max));
xlabel('Time [min]');
ylabel('KS Statistic');
title('Kolmogorov-Smirnov Statistic');
grid on;

sgtitle('Divergence Metrics Over Extraction Time', 'FontSize', 14);

%% Figure 2a: Divergence metrics integrated over time
figure('Name', 'Divergence Metrics Intgerated Over Time', 'Position', [150 150 1200 400]);

subplot(1, 3, 1);
plot(Time_full, cumsum(metrics.js_divergence), 'k-', 'LineWidth', 2);
hold on;
xline(metrics.js_max_time, 'r--', sprintf('Max=%.3f', metrics.js_max));
xlabel('Time [min]');
ylabel('JS Divergence [nats]');
title('Jensen-Shannon Divergence');
grid on;

subplot(1, 3, 2);
plot(Time_full, cumsum(metrics.kl_power_linear), 'b-', 'LineWidth', 2, 'DisplayName', 'KL(P||L)');
hold on;
plot(Time_full, cumsum(metrics.kl_linear_power), 'r-', 'LineWidth', 2, 'DisplayName', 'KL(L||P)');
plot(Time_full, cumsum(metrics.kl_linear_power + metrics.kl_power_linear), 'k-', 'LineWidth', 2, 'DisplayName', 'KL(L||P) + KL(P||L)');
xlabel('Time [min]');
ylabel('KL Divergence [nats]');
title('KL Divergence');
legend('Location', 'best');
grid on;

subplot(1, 3, 3);
plot(Time_full, cumsum(metrics.ks_stat), 'g-', 'LineWidth', 2);
hold on;
xline(metrics.ks_max_time, 'r--', sprintf('Max=%.3f', metrics.ks_max));
xlabel('Time [min]');
ylabel('KS Statistic');
title('Kolmogorov-Smirnov Statistic');
grid on;

sgtitle('Divergence metrics integrated over time', 'FontSize', 14);

%% Figure 3: Distribution evolution at selected times
figure('Name', 'Distribution Evolution', 'Position', [200 200 1400 600]);

t_select = [2, round(n_time_full/5), round(2*n_time_full/5), round(3*n_time_full/5), ...
            round(4*n_time_full/5), n_time_full];
t_select = unique(max(t_select, 2));  % Ensure t > 0

for j = 1:length(t_select)
    subplot(2, length(t_select), j);

    i_t = t_select(j);
    y_p = Y_power_valid(:, i_t);
    y_l = Y_linear_valid(:, i_t);

    edges = linspace(min([y_p; y_l])*0.95, max([y_p; y_l])*1.05, 30);

    histogram(y_p, edges, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'Normalization', 'pdf', 'DisplayName', 'Power');
    hold on;
    histogram(y_l, edges, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'Normalization', 'pdf', 'DisplayName', 'Linear');

    title(sprintf('t = %.0f min', Time_full(i_t)));
    xlabel('Yield [g]');
    if j == 1
        ylabel('PDF');
    end
    legend('Location', 'best');
    grid on;

    % Add JS annotation
    text(0.95, 0.95, sprintf('JS=%.3f', metrics.js_divergence(i_t)), ...
        'Units', 'normalized', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

    % CDF subplot
    subplot(2, length(t_select), length(t_select) + j);
    [f_p, x_p] = ecdf(y_p);
    [f_l, x_l] = ecdf(y_l);
    plot(x_p, f_p, 'b-', 'LineWidth', 2);
    hold on;
    plot(x_l, f_l, 'r-', 'LineWidth', 2);
    xlabel('Yield [g]');
    if j == 1
        ylabel('CDF');
    end
    title(sprintf('KS = %.3f', metrics.ks_stat(i_t)));
    grid on;
end

sgtitle('Distribution Evolution Over Time (PDF top, CDF bottom)', 'FontSize', 14);

%% Figure 4: Probability of Power > Linear
figure('Name', 'Probability Power Greater', 'Position', [250 250 800 500]);

plot(Time_full, metrics.prob_power_greater * 100, 'k-', 'LineWidth', 2);
hold on;
yline(50, 'r--', 'No Difference', 'LineWidth', 1.5);
yline(95, 'g--', '95%');
yline(5, 'g--', '5%');
fill([Time_full(1), Time_full(end), Time_full(end), Time_full(1)], [45 45 55 55], ...
    'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

xlabel('Time [min]');
ylabel('P(Power > Linear) [%%]', 'Interpreter', 'none');
title('Probability that Power Model Predicts Higher Yield');
ylim([0 100]);
grid on;

%% Figure 5: Final yield comparison
figure('Name', 'Final Yield Comparison', 'Position', [300 300 1000 400]);

subplot(1, 2, 1);
Y_final_diff = Y_power_valid(:, end) - Y_linear_valid(:, end);
histogram(Y_final_diff, 30, 'FaceColor', [0.5 0 0.5], 'FaceAlpha', 0.7, 'Normalization', 'pdf');
hold on;
xline(0, 'k--', 'LineWidth', 2);
xline(metrics.final_diff_mean, 'g-', 'LineWidth', 2);
xline(metrics.final_diff_ci95(1), 'm:', 'LineWidth', 1.5);
xline(metrics.final_diff_ci95(2), 'm:', 'LineWidth', 1.5);
xlabel('Final Yield Difference (Power - Linear) [g]');
ylabel('PDF');
title(sprintf('Final Diff: %.4f +/- %.4f g', metrics.final_diff_mean, metrics.final_diff_std));
grid on;

subplot(1, 2, 2);
scatter(Y_linear_valid(:, end), Y_power_valid(:, end), 20, 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
lims = [min([Y_power_valid(:, end); Y_linear_valid(:, end)]), ...
        max([Y_power_valid(:, end); Y_linear_valid(:, end)])];
plot(lims, lims, 'k--', 'LineWidth', 1.5);
xlabel('Linear Final Yield [g]');
ylabel('Power Final Yield [g]');
title('Power vs Linear Final Yield');
axis equal;
xlim(lims);
ylim(lims);
grid on;

sgtitle(sprintf('Final Yield Comparison at t = %.0f min', ExtractionTime), 'FontSize', 14);

%% Figure 6: Integrated metrics summary bar chart
%{
figure('Name', 'Integrated Metrics Summary', 'Position', [350 350 600 400]);

metric_names = {'JS', 'KL(P||L)', 'KS'};
metric_vals = [metrics.js_integrated, metrics.kl_integrated, metrics.ks_integrated];

bar(categorical(metric_names), metric_vals);
ylabel('Integrated Value');
title('Integrated Divergence Metrics Over Trajectory');
grid on;

% Add value labels
for i = 1:length(metric_vals)
    text(i, metric_vals(i) + 0.02*max(metric_vals), sprintf('%.2f', metric_vals(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end
%}
%% ========================================================================
%  SUMMARY TABLE
%  ========================================================================
fprintf('\n');
fprintf('=============================================================================\n');
fprintf('                         SUMMARY RESULTS                                     \n');
fprintf('=============================================================================\n');
fprintf('Operating point: T=%.0fK (%.0fC), P=%.0f bar, F=%.2e m3/s\n', T0, T0-273, P0, F0);
fprintf('MC samples: %d valid out of %d\n', n_valid, N_MC);
fprintf('-----------------------------------------------------------------------------\n');
fprintf('%-25s %-15s %-15s\n', 'Metric', 'Value', 'Time of Max');
fprintf('-----------------------------------------------------------------------------\n');
fprintf('%-25s %-15.4f %-15.0f min\n', 'JS Divergence (int)', metrics.js_integrated, metrics.js_max_time);
fprintf('%-25s %-15.4f %-15.0f min\n', 'KL(P||L) (int)', metrics.kl_integrated, metrics.kl_max_time);
fprintf('%-25s %-15.4f %-15.0f min\n', 'KS Statistic (int)', metrics.ks_integrated, metrics.ks_max_time);
fprintf('-----------------------------------------------------------------------------\n');
fprintf('%-25s %-15.4f %-15s\n', 'Max JS', metrics.js_max, sprintf('%.0f min', metrics.js_max_time));
fprintf('%-25s %-15.4f %-15s\n', 'Max KS', metrics.ks_max, sprintf('%.0f min', metrics.ks_max_time));
fprintf('-----------------------------------------------------------------------------\n');
fprintf('%-25s %-15.6f\n', 'Final Diff Mean [g]', metrics.final_diff_mean);
fprintf('%-25s %-15.6f\n', 'Final Diff Std [g]', metrics.final_diff_std);
fprintf('%-25s [%.4f, %.4f]\n', 'Final Diff 95%% CI [g]', metrics.final_diff_ci95);
fprintf('=============================================================================\n');

%% Interpretation
fprintf('\n=== INTERPRETATION ===\n\n');

% Check if CI excludes zero
if metrics.final_diff_ci95(1) > 0
    fprintf('Final yield: Power SIGNIFICANTLY GREATER than Linear (95%% CI excludes 0)\n');
elseif metrics.final_diff_ci95(2) < 0
    fprintf('Final yield: Linear SIGNIFICANTLY GREATER than Power (95%% CI excludes 0)\n');
else
    fprintf('Final yield: NO SIGNIFICANT DIFFERENCE (95%% CI includes 0)\n');
end

% Timing of maximum separation
fprintf('Maximum separation occurs at t = %.0f min (JS), %.0f min (KS)\n', ...
    metrics.js_max_time, metrics.ks_max_time);

% Discrimination strength
if metrics.js_max > 0.5
    disc_str = 'STRONG';
elseif metrics.js_max > 0.2
    disc_str = 'MODERATE';
elseif metrics.js_max > 0.1
    disc_str = 'WEAK';
else
    disc_str = 'NEGLIGIBLE';
end
fprintf('Discrimination strength (by max JS): %s (JS_max = %.3f)\n', disc_str, metrics.js_max);

%% Save results
results = struct();
results.T0 = T0;
results.P0 = P0;
results.F0 = F0;
results.ExtractionTime = ExtractionTime;
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

save('trajectory_discrimination_with_uncertainty_results.mat', 'results');
fprintf('\nResults saved to trajectory_discrimination_with_uncertainty_results.mat\n');

fprintf('\n=== Analysis Complete ===\n');

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================
function [kl_pl, kl_lp, js, overlap] = compute_divergences(y_p, y_l)
    % Compute divergence metrics between two samples using KDE
    %
    % Outputs:
    %   kl_pl   - KL divergence KL(Power || Linear)
    %   kl_lp   - KL divergence KL(Linear || Power)
    %   js      - Jensen-Shannon divergence (symmetric)
    %   overlap - Overlap coefficient

    y_all = [y_p; y_l];
    y_min = min(y_all);
    y_max = max(y_all);

    if y_max <= y_min || numel(y_all) < 5
        kl_pl = 0; kl_lp = 0; js = 0; overlap = 1;
        return;
    end

    % KDE on common grid
    n_grid = 200;
    y_grid = linspace(y_min - 0.1*(y_max-y_min), y_max + 0.1*(y_max-y_min), n_grid);

    f_p = ksdensity(y_p, y_grid, 'Function', 'pdf');
    f_l = ksdensity(y_l, y_grid, 'Function', 'pdf');

    % Normalize to ensure proper PDFs
    dx = y_grid(2) - y_grid(1);
    f_p = f_p / (sum(f_p) * dx);
    f_l = f_l / (sum(f_l) * dx);

    % Add small epsilon to avoid log(0)
    eps_val = 1e-12;
    f_p = max(f_p, eps_val);
    f_l = max(f_l, eps_val);

    % KL divergence: KL(P || L) = integral(p * log(p/l))
    kl_pl = trapz(y_grid, f_p .* log(f_p ./ f_l));
    kl_lp = trapz(y_grid, f_l .* log(f_l ./ f_p));

    % Jensen-Shannon divergence: JS = 0.5*KL(P||M) + 0.5*KL(L||M) where M = 0.5*(P+L)
    f_m = 0.5 * (f_p + f_l);
    js = 0.5 * trapz(y_grid, f_p .* log(f_p ./ f_m)) + ...
         0.5 * trapz(y_grid, f_l .* log(f_l ./ f_m));

    % Overlap coefficient: OVL = integral(min(p, l))
    overlap = trapz(y_grid, min(f_p, f_l));
end
