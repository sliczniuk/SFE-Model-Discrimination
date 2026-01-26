%% Trajectory-Based Distribution Comparison for SFE Models
% This script compares Power and Linear model OUTPUT TRAJECTORIES
% by sampling T and F from uniform distributions at fixed pressure levels.
%
% Key difference from distribution_comparison.m:
% - Stores and analyzes FULL yield trajectories Y(t), not just final yield
% - Computes time-pointwise KS statistics and other metrics
% - Computes functional/integrated metrics over entire trajectory
%
% Method: Latin Hypercube Sampling (LHS) for efficient coverage of (T, F) space
% Output: Trajectory-based comparison metrics at each pressure level

%% Initialization
startup;

addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
import casadi.*

rng(42);  % Set seed for reproducibility

%% Parallel Pool Setup
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local');
end
n_workers = pool.NumWorkers;
fprintf('Parallel pool started with %d workers\n', n_workers);

%% Configuration
N_samples = 2000;  % Number of LHS samples
P_levels = [100, 125, 150, 175, 200];  % Fixed pressure levels [bar]

% Input ranges for uniform distributions
T_min = 303;  T_max = 313;  % Temperature [K] (30-40 C)
F_min = 3.3e-5; F_max = 6.7e-5; % Flow rate [m3/s]

fprintf('=== Trajectory-Based Distribution Comparison ===\n\n');
fprintf('Configuration:\n');
fprintf('  N_samples: %d (LHS)\n', N_samples);
fprintf('  Pressure levels: %s bar\n', mat2str(P_levels));
fprintf('  T range: [%.0f, %.0f] K (%.0f-%.0f C)\n', T_min, T_max, T_min-273, T_max-273);
fprintf('  F range: [%.2e, %.2e] m3/s\n', F_min, F_max);
fprintf('\n');

%% Load Data and Parameters
Parameters_table = readtable('Parameters.csv');
Parameters       = num2cell(Parameters_table{:,3});

%% Physical Parameters
m_total = 3.0;  % Total mass [g]

% Bed geometry
before = 0.04;
bed    = 0.92;

%% Time Configuration
PreparationTime = 0;
ExtractionTime  = 900;
timeStep        = 5;  % [minutes]

simulationTime   = PreparationTime + ExtractionTime;
timeStep_in_sec  = timeStep * 60;
Time_in_sec      = (timeStep:timeStep:simulationTime) * 60;
Time             = [0 Time_in_sec/60];
N_Time           = length(Time_in_sec);

fprintf('Time configuration:\n');
fprintf('  Extraction time: %d min\n', ExtractionTime);
fprintf('  Time step: %d min\n', timeStep);
fprintf('  Number of time points: %d\n', N_Time);
fprintf('\n');

%% Extractor Geometry
nstages = Parameters{1};
r       = Parameters{3};
epsi    = Parameters{4};
L       = Parameters{6};

nstagesbefore = 1:floor(before * nstages);
nstagesbed    = nstagesbefore(end)+1 : nstagesbefore(end) + floor(bed * nstages);
nstagesafter  = nstagesbed(end)+1 : nstages;

bed_mask                = nan(nstages, 1);
bed_mask(nstagesbefore) = 0;
bed_mask(nstagesbed)    = 1;
bed_mask(nstagesafter)  = 0;

%% Volume Calculations
V_slice   = (L/nstages) * pi * r^2;
V_before = V_slice * numel(nstagesbefore);
V_after  = V_slice * numel(nstagesafter);
V_bed    = V_slice * numel(nstagesbed);

V_before_fluid = repmat(V_before * 1          / numel(nstagesbefore), numel(nstagesbefore), 1);
V_bed_fluid    = repmat(V_bed    * (1 - epsi) / numel(nstagesbed),    numel(nstagesbed),    1);
V_after_fluid  = repmat(V_after  * 1          / numel(nstagesafter),  numel(nstagesafter),  1);
V_fluid        = [V_before_fluid; V_bed_fluid; V_after_fluid];

L_bed_after_nstages = linspace(0, L, nstages);
L_bed_after_nstages = L_bed_after_nstages(nstagesbed(1):end);
L_bed_after_nstages = L_bed_after_nstages - L_bed_after_nstages(1);
L_end               = L_bed_after_nstages(end);

%% State and Input Dimensions
Nx = 3 * nstages + 2;
Nu = 3 + numel(Parameters);

%% Initial Conditions Setup
msol_max   = m_total;
mSol_ratio = 1;
mSOL_s = msol_max * mSol_ratio;
mSOL_f = msol_max * (1 - mSol_ratio);

C0solid       = mSOL_s * 1e-3 / (V_bed * epsi);
Parameters{2} = C0solid;

G       = @(x) -(2*mSOL_f / L_end^2) * (x - L_end);
m_fluid = G(L_bed_after_nstages) * L_bed_after_nstages(2);
m_fluid = [zeros(1, numel(nstagesbefore)) m_fluid];
C0fluid = m_fluid * 1e-3 ./ V_fluid';

%% Precompute bed mask constants
epsi_mask           = epsi .* bed_mask;
one_minus_epsi_mask = 1 - epsi_mask;

%% Generate LHS samples for (T, F)
fprintf('Generating %d LHS samples...\n', N_samples);
lhs_samples = lhsdesign(N_samples, 2);

T_samples = T_min + (T_max - T_min) * lhs_samples(:, 1);
F_samples = F_min + (F_max - F_min) * lhs_samples(:, 2);

%% Preallocate results storage
n_P = length(P_levels);
Parameters_mat = cell2mat(Parameters);

% Store FULL TRAJECTORIES for each sample and pressure level
% Y_traj_power{i_P}(i_sample, i_time) = yield at time i_time for sample i_sample
Y_traj_power  = cell(n_P, 1);
Y_traj_linear = cell(n_P, 1);

for i_P = 1:n_P
    Y_traj_power{i_P}  = zeros(N_samples, N_Time);
    Y_traj_linear{i_P} = zeros(N_samples, N_Time);
end

% Store sampled inputs
results.T_samples = T_samples;
results.F_samples = F_samples;
results.P_levels  = P_levels;
results.Time      = Time(2:end);  % Exclude t=0
results.N_Time    = N_Time;

%% ========================================================================
%  MAIN SIMULATION LOOP (PARALLELIZED WITH BATCH PROCESSING)
%  ========================================================================
fprintf('\nRunning simulations in parallel...\n');
fprintf('Storing FULL yield trajectories for each sample...\n');
tic;

% Local copies for parfor
bed_mask_local = bed_mask;
timeStep_in_sec_local = timeStep_in_sec;
epsi_mask_local = epsi_mask;
one_minus_epsi_mask_local = one_minus_epsi_mask;
Nx_local = Nx;
Nu_local = Nu;
N_Time_local = N_Time;
nstages_local = nstages;
C0fluid_local = C0fluid;
C0solid_local = C0solid;
Parameters_local = Parameters;

% Batch processing
batch_size = ceil(N_samples / n_workers);

for i_P = 1:n_P
    P0 = P_levels(i_P);
    fprintf('\n--- Pressure: %.0f bar ---\n', P0);

    % Create batch indices
    batch_starts = 1:batch_size:N_samples;
    batch_ends = min(batch_starts + batch_size - 1, N_samples);
    n_batches_actual = length(batch_starts);

    % Cell arrays to collect batch results (full trajectories)
    Y_linear_batches = cell(n_batches_actual, 1);
    Y_power_batches  = cell(n_batches_actual, 1);

    % Parallel loop over batches
    parfor i_batch = 1:n_batches_actual
        import casadi.*

        % Build integrators ONCE per batch/worker
        f_linear = @(x, u) modelSFE(x, u, bed_mask_local, timeStep_in_sec_local, ...
            'Linear_model', epsi_mask_local, one_minus_epsi_mask_local);
        F_linear = buildIntegrator(f_linear, [Nx_local, Nu_local], timeStep_in_sec_local, 'cvodes');

        f_power = @(x, u) modelSFE(x, u, bed_mask_local, timeStep_in_sec_local, ...
            'Power_model', epsi_mask_local, one_minus_epsi_mask_local);
        F_power = buildIntegrator(f_power, [Nx_local, Nu_local], timeStep_in_sec_local, 'cvodes');

        F_accum_linear = F_linear.mapaccum('F_accum_linear', N_Time_local);
        F_accum_power  = F_power.mapaccum('F_accum_power', N_Time_local);

        % Get sample indices for this batch
        idx_start = batch_starts(i_batch);
        idx_end   = batch_ends(i_batch);
        batch_indices = idx_start:idx_end;
        n_batch = length(batch_indices);

        % Preallocate batch results - FULL TRAJECTORIES
        Y_linear_batch = zeros(n_batch, N_Time_local);
        Y_power_batch  = zeros(n_batch, N_Time_local);

        for j = 1:n_batch
            i_s = batch_indices(j);
            T0 = T_samples(i_s);
            F0 = F_samples(i_s);

            % Compute fluid properties
            Z            = Compressibility(T0, P0, Parameters_local);
            rho          = rhoPB_Comp(T0, P0, Z, Parameters_local);
            enthalpy_rho = rho .* SpecificEnthalpy(T0, P0, Z, rho, Parameters_local);

            % Build input vectors
            feedTemp  = T0 * ones(1, N_Time_local);
            feedPress = P0 * ones(1, N_Time_local);
            feedFlow  = F0 * ones(1, N_Time_local);

            uu = [feedTemp', feedPress', feedFlow'];

            % Initial state
            x0 = [C0fluid_local';
                  C0solid_local * bed_mask_local;
                  enthalpy_rho * ones(nstages_local, 1);
                  P0;
                  0];

            U_all = [uu'; repmat(Parameters_mat, 1, N_Time_local)];

            try
                X_all_linear = F_accum_linear(x0, U_all);
                X_all_power  = F_accum_power(x0, U_all);

                % Extract FULL yield trajectory (last state = cumulative yield)
                Y_linear_batch(j, :) = full(X_all_linear(Nx_local, :));
                Y_power_batch(j, :)  = full(X_all_power(Nx_local, :));

            catch ME
                Y_linear_batch(j, :) = NaN;
                Y_power_batch(j, :)  = NaN;
            end
        end

        Y_linear_batches{i_batch} = Y_linear_batch;
        Y_power_batches{i_batch}  = Y_power_batch;
    end

    % Reassemble results from batches
    Y_linear_temp = zeros(N_samples, N_Time);
    Y_power_temp  = zeros(N_samples, N_Time);

    for i_batch = 1:n_batches_actual
        idx_start = batch_starts(i_batch);
        idx_end   = batch_ends(i_batch);
        Y_linear_temp(idx_start:idx_end, :) = Y_linear_batches{i_batch};
        Y_power_temp(idx_start:idx_end, :)  = Y_power_batches{i_batch};
    end

    % Store full trajectories for this pressure level
    Y_traj_linear{i_P} = Y_linear_temp;
    Y_traj_power{i_P}  = Y_power_temp;

    % Progress update
    n_valid = sum(~any(isnan(Y_linear_temp), 2));
    fprintf('  Completed: %d/%d valid trajectories\n', n_valid, N_samples);
end

elapsed_time = toc;
fprintf('\nSimulations completed in %.1f seconds (%.1f samples/sec).\n', ...
    elapsed_time, N_samples * n_P / elapsed_time);

%% ========================================================================
%  COMPUTE TRAJECTORY-BASED COMPARISON METRICS
%  ========================================================================
fprintf('\n=== Computing Trajectory-Based Comparison Metrics ===\n\n');

% Time vector for analysis
Time_vec = Time(2:end);  % Exclude t=0

% Preallocate metrics structure
metrics = struct();
metrics.P_levels = P_levels;
metrics.Time = Time_vec;

% Time-pointwise metrics (for each pressure and time point)
metrics.ks_stat_t      = zeros(n_P, N_Time);  % KS statistic at each time
metrics.ks_p_t         = zeros(n_P, N_Time);  % KS p-value at each time
metrics.wasserstein_t  = zeros(n_P, N_Time);  % Wasserstein at each time
metrics.mean_diff_t    = zeros(n_P, N_Time);  % Mean difference at each time
metrics.cohens_d_t     = zeros(n_P, N_Time);  % Cohen's d at each time
metrics.kl_t           = zeros(n_P, N_Time);  % KL divergence at each time (Power || Linear)

% Integrated/functional metrics (single value per pressure)
metrics.ks_integrated     = zeros(n_P, 1);  % Integral of KS over time
metrics.ks_max            = zeros(n_P, 1);  % Maximum KS statistic
metrics.ks_max_time       = zeros(n_P, 1);  % Time of maximum KS
metrics.wasserstein_integrated = zeros(n_P, 1);  % Integral of Wasserstein
metrics.kl_integrated     = zeros(n_P, 1);  % Integral of KL over time
metrics.trajectory_rmse   = zeros(n_P, 1);  % RMSE between mean trajectories
metrics.trajectory_mae    = zeros(n_P, 1);  % MAE between mean trajectories
metrics.area_between_curves = zeros(n_P, 1);  % Area between mean curves

% Statistical bands
metrics.mean_power_t   = zeros(n_P, N_Time);
metrics.mean_linear_t  = zeros(n_P, N_Time);
metrics.std_power_t    = zeros(n_P, N_Time);
metrics.std_linear_t   = zeros(n_P, N_Time);
metrics.ci95_power_t   = zeros(n_P, N_Time, 2);  % 95% CI [lower, upper]
metrics.ci95_linear_t  = zeros(n_P, N_Time, 2);

for i_P = 1:n_P
    P0 = P_levels(i_P);
    fprintf('P = %.0f bar:\n', P0);

    Y_power_all  = Y_traj_power{i_P};
    Y_linear_all = Y_traj_linear{i_P};

    % Remove samples with any NaN values
    valid_idx = ~any(isnan(Y_power_all), 2) & ~any(isnan(Y_linear_all), 2);
    Y_power  = Y_power_all(valid_idx, :);
    Y_linear = Y_linear_all(valid_idx, :);
    n_valid  = sum(valid_idx);

    fprintf('  Valid trajectories: %d/%d\n', n_valid, N_samples);

    %% Compute time-pointwise statistics
    for i_t = 1:N_Time
        y_p = Y_power(:, i_t);
        y_l = Y_linear(:, i_t);

        % Basic statistics
        metrics.mean_power_t(i_P, i_t)  = mean(y_p);
        metrics.mean_linear_t(i_P, i_t) = mean(y_l);
        metrics.std_power_t(i_P, i_t)   = std(y_p);
        metrics.std_linear_t(i_P, i_t)  = std(y_l);

        % 95% CI
        sem_p = std(y_p) / sqrt(n_valid);
        sem_l = std(y_l) / sqrt(n_valid);
        metrics.ci95_power_t(i_P, i_t, :)  = [mean(y_p) - 1.96*sem_p, mean(y_p) + 1.96*sem_p];
        metrics.ci95_linear_t(i_P, i_t, :) = [mean(y_l) - 1.96*sem_l, mean(y_l) + 1.96*sem_l];

        % Mean difference
        metrics.mean_diff_t(i_P, i_t) = mean(y_p) - mean(y_l);

        % Cohen's d
        pooled_std = sqrt(((n_valid-1)*std(y_p)^2 + (n_valid-1)*std(y_l)^2) / (2*n_valid - 2));
        if pooled_std > 0
            metrics.cohens_d_t(i_P, i_t) = (mean(y_p) - mean(y_l)) / pooled_std;
        else
            metrics.cohens_d_t(i_P, i_t) = 0;
        end

        % Kolmogorov-Smirnov test
        [~, p_ks, ks_stat] = kstest2(y_p, y_l);
        metrics.ks_stat_t(i_P, i_t) = ks_stat;
        metrics.ks_p_t(i_P, i_t)    = p_ks;

        % Wasserstein distance
        y_p_sorted = sort(y_p);
        y_l_sorted = sort(y_l);
        metrics.wasserstein_t(i_P, i_t) = mean(abs(y_p_sorted - y_l_sorted));

        % KL divergence (Power || Linear) using KDE
        metrics.kl_t(i_P, i_t) = compute_kl_divergence(y_p, y_l);
    end

    %% Compute integrated/functional metrics
    dt = Time_vec(2) - Time_vec(1);  % Time step in minutes

    % Integrated KS statistic (trapezoidal rule)
    metrics.ks_integrated(i_P) = trapz(Time_vec, metrics.ks_stat_t(i_P, :));

    % Maximum KS and time of maximum
    [metrics.ks_max(i_P), idx_max] = max(metrics.ks_stat_t(i_P, :));
    metrics.ks_max_time(i_P) = Time_vec(idx_max);

    % Integrated Wasserstein distance
    metrics.wasserstein_integrated(i_P) = trapz(Time_vec, metrics.wasserstein_t(i_P, :));

    % Integrated KL divergence
    metrics.kl_integrated(i_P) = trapz(Time_vec, metrics.kl_t(i_P, :));

    % RMSE and MAE between mean trajectories
    mean_diff = metrics.mean_power_t(i_P, :) - metrics.mean_linear_t(i_P, :);
    metrics.trajectory_rmse(i_P) = sqrt(mean(mean_diff.^2));
    metrics.trajectory_mae(i_P)  = mean(abs(mean_diff));

    % Area between mean curves (absolute)
    metrics.area_between_curves(i_P) = trapz(Time_vec, abs(mean_diff));

    % Print summary
    fprintf('  Integrated KS: %.4f\n', metrics.ks_integrated(i_P));
    fprintf('  Max KS: %.4f at t = %.0f min\n', metrics.ks_max(i_P), metrics.ks_max_time(i_P));
    fprintf('  Trajectory RMSE: %.4f g\n', metrics.trajectory_rmse(i_P));
    fprintf('  Area between curves: %.4f g*min\n', metrics.area_between_curves(i_P));
    fprintf('\n');
end

%% Store results
results.Y_traj_power  = Y_traj_power;
results.Y_traj_linear = Y_traj_linear;
results.metrics = metrics;
results.N_samples = N_samples;

%% ========================================================================
%  VISUALIZATION
%  ========================================================================
fprintf('Generating visualizations...\n');

colors_P = lines(n_P);

%% Figure 1: Mean trajectories with confidence bands
figure('Name', 'Mean Yield Trajectories with 95pct CI', 'Position', [100 100 1400 800]);

for i_P = 1:n_P
    subplot(2, 3, i_P);

    % Power model - mean and CI
    mean_p = metrics.mean_power_t(i_P, :);
    ci_p_low = squeeze(metrics.ci95_power_t(i_P, :, 1));
    ci_p_high = squeeze(metrics.ci95_power_t(i_P, :, 2));

    % Linear model - mean and CI
    mean_l = metrics.mean_linear_t(i_P, :);
    ci_l_low = squeeze(metrics.ci95_linear_t(i_P, :, 1));
    ci_l_high = squeeze(metrics.ci95_linear_t(i_P, :, 2));

    % Plot confidence bands
    fill([Time_vec, fliplr(Time_vec)], [ci_p_low, fliplr(ci_p_high)], ...
        'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    hold on;
    fill([Time_vec, fliplr(Time_vec)], [ci_l_low, fliplr(ci_l_high)], ...
        'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % Plot mean trajectories
    plot(Time_vec, mean_p, 'b-', 'LineWidth', 2, 'DisplayName', 'Power');
    plot(Time_vec, mean_l, 'r--', 'LineWidth', 2, 'DisplayName', 'Linear');

    xlabel('Time [min]');
    ylabel('Yield [g]');
    title(sprintf('P = %.0f bar', P_levels(i_P)));
    legend('Location', 'southeast');
    grid on;
end

subplot(2, 3, 6);
% Plot all mean trajectories together
for i_P = 1:n_P
    plot(Time_vec, metrics.mean_power_t(i_P, :), '-', 'LineWidth', 2, 'Color', colors_P(i_P,:), ...
        'DisplayName', sprintf('Power P=%.0f', P_levels(i_P)));
    hold on;
    plot(Time_vec, metrics.mean_linear_t(i_P, :), '--', 'LineWidth', 1.5, 'Color', colors_P(i_P,:), ...
        'HandleVisibility', 'off');
end
xlabel('Time [min]');
ylabel('Yield [g]');
title('All Pressures (solid=Power, dashed=Linear)');
legend('Location', 'southeast');
grid on;

sgtitle('Mean Yield Trajectories with 95%% Confidence Intervals', 'FontSize', 14, 'Interpreter', 'none');

%% Figure 2: Time-pointwise KS statistics
figure('Name', 'Time-Pointwise KS Statistics', 'Position', [150 150 1200 800]);

subplot(2, 2, 1);
for i_P = 1:n_P
    plot(Time_vec, metrics.ks_stat_t(i_P, :), 'o-', 'LineWidth', 2, 'Color', colors_P(i_P,:), ...
        'MarkerFaceColor', colors_P(i_P,:), 'DisplayName', sprintf('P = %.0f bar', P_levels(i_P)));
    hold on;
end
xlabel('Time [min]');
ylabel('KS Statistic');
title('Kolmogorov-Smirnov Statistic vs Time');
legend('Location', 'best');
grid on;

subplot(2, 2, 2);
for i_P = 1:n_P
    semilogy(Time_vec, metrics.ks_p_t(i_P, :), 'o-', 'LineWidth', 2, 'Color', colors_P(i_P,:), ...
        'MarkerFaceColor', colors_P(i_P,:), 'DisplayName', sprintf('P = %.0f bar', P_levels(i_P)));
    hold on;
end
yline(0.05, 'k--', '\alpha = 0.05', 'LineWidth', 1.5);
yline(0.01, 'k:', '\alpha = 0.01', 'LineWidth', 1.5);
xlabel('Time [min]');
ylabel('p-value');
title('KS Test p-value vs Time');
legend('Location', 'best');
grid on;

subplot(2, 2, 3);
for i_P = 1:n_P
    plot(Time_vec, metrics.wasserstein_t(i_P, :), 's-', 'LineWidth', 2, 'Color', colors_P(i_P,:), ...
        'MarkerFaceColor', colors_P(i_P,:), 'DisplayName', sprintf('P = %.0f bar', P_levels(i_P)));
    hold on;
end
xlabel('Time [min]');
ylabel('Wasserstein Distance [g]');
title('Wasserstein Distance vs Time');
legend('Location', 'best');
grid on;

subplot(2, 2, 4);
for i_P = 1:n_P
    plot(Time_vec, abs(metrics.cohens_d_t(i_P, :)), '^-', 'LineWidth', 2, 'Color', colors_P(i_P,:), ...
        'MarkerFaceColor', colors_P(i_P,:), 'DisplayName', sprintf('P = %.0f bar', P_levels(i_P)));
    hold on;
end
yline(0.2, 'k--', 'Small');
yline(0.5, 'k--', 'Medium');
yline(0.8, 'k--', 'Large');
xlabel('Time [min]');
ylabel('|Cohen''s d|');
title('Effect Size vs Time');
legend('Location', 'best');
grid on;

sgtitle('Time-Pointwise Distribution Comparison Metrics', 'FontSize', 14);

%% Figure 3: Integrated metrics summary
figure('Name', 'Integrated Trajectory Metrics', 'Position', [200 200 1400 500]);

subplot(1, 5, 1);
bar(P_levels, metrics.ks_integrated);
xlabel('Pressure [bar]');
ylabel('Integrated KS');
title('Integrated KS Statistic');
grid on;

subplot(1, 5, 2);
bar(P_levels, metrics.ks_max);
hold on;
% Add time annotations
for i_P = 1:n_P
    text(P_levels(i_P), metrics.ks_max(i_P) + 0.01, sprintf('t=%.0f', metrics.ks_max_time(i_P)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8);
end
xlabel('Pressure [bar]');
ylabel('Max KS Statistic');
title('Maximum KS (with time)');
grid on;

subplot(1, 5, 3);
bar(P_levels, metrics.wasserstein_integrated);
xlabel('Pressure [bar]');
ylabel('Integrated Wasserstein [g*min]');
title('Integrated Wasserstein Distance');
grid on;

subplot(1, 5, 4);
bar(P_levels, metrics.kl_integrated);
xlabel('Pressure [bar]');
ylabel('Integrated KL [nats*min]');
title('Integrated KL Divergence');
grid on;

subplot(1, 5, 5);
bar(P_levels, metrics.area_between_curves);
xlabel('Pressure [bar]');
ylabel('Area [g*min]');
title('Area Between Mean Curves');
grid on;

sgtitle('Integrated Trajectory Comparison Metrics', 'FontSize', 14);

%% Figure 3b: Time-pointwise KL divergence
figure('Name', 'Time-Pointwise KL Divergence', 'Position', [220 220 1200 500]);

for i_P = 1:n_P
    plot(Time_vec, metrics.kl_t(i_P, :), 'd-', 'LineWidth', 2, 'Color', colors_P(i_P,:), ...
        'MarkerFaceColor', colors_P(i_P,:), 'DisplayName', sprintf('P = %.0f bar', P_levels(i_P)));
    hold on;
end
xlabel('Time [min]');
ylabel('KL Divergence (Power || Linear) [nats]');
title('KL Divergence vs Time');
legend('Location', 'best');
grid on;

%% Figure 4: Heatmap of KS statistic over time and pressure
figure('Name', 'KS Statistic Heatmap', 'Position', [250 250 1000 400]);

subplot(1, 2, 1);
imagesc(Time_vec, P_levels, metrics.ks_stat_t);
colorbar;
xlabel('Time [min]');
ylabel('Pressure [bar]');
title('KS Statistic');
set(gca, 'YDir', 'normal');
colormap(gca, 'hot');

subplot(1, 2, 2);
imagesc(Time_vec, P_levels, abs(metrics.cohens_d_t));
colorbar;
xlabel('Time [min]');
ylabel('Pressure [bar]');
title('|Cohen''s d|');
set(gca, 'YDir', 'normal');
colormap(gca, 'hot');

sgtitle('Discrimination Metrics: Pressure vs Time', 'FontSize', 14);

%% Figure 5: Sample trajectories at best pressure
[~, idx_best_P] = max(metrics.ks_integrated);
P_best = P_levels(idx_best_P);

figure('Name', sprintf('Sample Trajectories at P = %.0f bar', P_best), 'Position', [300 300 1000 600]);

Y_power_best  = Y_traj_power{idx_best_P};
Y_linear_best = Y_traj_linear{idx_best_P};

% Remove NaN rows
valid_idx = ~any(isnan(Y_power_best), 2) & ~any(isnan(Y_linear_best), 2);
Y_power_best  = Y_power_best(valid_idx, :);
Y_linear_best = Y_linear_best(valid_idx, :);

% Plot a subset of trajectories
n_plot = min(50, size(Y_power_best, 1));
idx_plot = randperm(size(Y_power_best, 1), n_plot);

subplot(1, 2, 1);
for i = 1:n_plot
    plot(Time_vec, Y_power_best(idx_plot(i), :), 'b-', 'LineWidth', 0.5, 'Color', [0 0 1 0.2]);
    hold on;
end
plot(Time_vec, mean(Y_power_best, 1), 'b-', 'LineWidth', 3, 'DisplayName', 'Mean');
xlabel('Time [min]');
ylabel('Yield [g]');
title(sprintf('Power Model (n=%d samples shown)', n_plot));
grid on;

subplot(1, 2, 2);
for i = 1:n_plot
    plot(Time_vec, Y_linear_best(idx_plot(i), :), 'r-', 'LineWidth', 0.5, 'Color', [1 0 0 0.2]);
    hold on;
end
plot(Time_vec, mean(Y_linear_best, 1), 'r-', 'LineWidth', 3, 'DisplayName', 'Mean');
xlabel('Time [min]');
ylabel('Yield [g]');
title(sprintf('Linear Model (n=%d samples shown)', n_plot));
grid on;

sgtitle(sprintf('Sample Yield Trajectories at P = %.0f bar', P_best), 'FontSize', 14);

%% Figure 6: Distribution evolution at selected time points
figure('Name', 'Distribution Evolution at Selected Times', 'Position', [350 350 1400 800]);

% Select time points to visualize
t_select = [1, round(N_Time/4), round(N_Time/2), round(3*N_Time/4), N_Time];
t_select = unique(t_select);

for i_P = 1:min(n_P, 5)  % Limit to 4 pressures for clarity
    for j = 1:length(t_select)
        subplot(min(n_P, 5), length(t_select), (i_P-1)*length(t_select) + j);

        Y_power_all  = Y_traj_power{i_P};
        Y_linear_all = Y_traj_linear{i_P};

        valid_idx = ~any(isnan(Y_power_all), 2) & ~any(isnan(Y_linear_all), 2);
        y_p = Y_power_all(valid_idx, t_select(j));
        y_l = Y_linear_all(valid_idx, t_select(j));

        edges = linspace(min([y_p; y_l])*0.95, max([y_p; y_l])*1.05, 20);

        histogram(y_p, edges, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'EdgeColor', 'b');
        hold on;
        histogram(y_l, edges, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'r');

        if i_P == 1
            title(sprintf('t = %.0f min', Time_vec(t_select(j))));
        end
        if j == 1
            ylabel(sprintf('P=%.0f bar', P_levels(i_P)));
        end
        if i_P == 5
            xlabel('Yield [g]');
        end

        % Add KS annotation
        ks_val = metrics.ks_stat_t(i_P, t_select(j));
        text(0.95, 0.95, sprintf('KS=%.2f', ks_val), 'Units', 'normalized', ...
            'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 8);

        grid on;
    end
end

sgtitle('Yield Distribution Evolution Over Time', 'FontSize', 14);

%% Figure 7: Trajectory difference analysis
figure('Name', 'Trajectory Difference Analysis', 'Position', [400 400 1200 600]);

subplot(2, 2, 1);
for i_P = 1:n_P
    plot(Time_vec, metrics.mean_diff_t(i_P, :), 'o-', 'LineWidth', 2, 'Color', colors_P(i_P,:), ...
        'MarkerFaceColor', colors_P(i_P,:), 'DisplayName', sprintf('P = %.0f bar', P_levels(i_P)));
    hold on;
end
yline(0, 'k--');
xlabel('Time [min]');
ylabel('Mean Difference [g]');
title('Mean Trajectory Difference (Power - Linear)');
legend('Location', 'best');
grid on;

subplot(2, 2, 2);
% Cumulative difference
for i_P = 1:n_P
    cumulative_diff = cumtrapz(Time_vec, abs(metrics.mean_diff_t(i_P, :)));
    plot(Time_vec, cumulative_diff, 'o-', 'LineWidth', 2, 'Color', colors_P(i_P,:), ...
        'MarkerFaceColor', colors_P(i_P,:), 'DisplayName', sprintf('P = %.0f bar', P_levels(i_P)));
    hold on;
end
xlabel('Time [min]');
ylabel('Cumulative |Difference| [g*min]');
title('Cumulative Absolute Difference');
legend('Location', 'best');
grid on;

subplot(2, 2, 3);
% Relative difference
for i_P = 1:n_P
    mean_avg = 0.5 * (metrics.mean_power_t(i_P, :) + metrics.mean_linear_t(i_P, :));
    rel_diff = 100 * metrics.mean_diff_t(i_P, :) ./ mean_avg;
    plot(Time_vec, rel_diff, 'o-', 'LineWidth', 2, 'Color', colors_P(i_P,:), ...
        'MarkerFaceColor', colors_P(i_P,:), 'DisplayName', sprintf('P = %.0f bar', P_levels(i_P)));
    hold on;
end
yline(0, 'k--');
xlabel('Time [min]');
ylabel('Relative Difference [%%]', 'Interpreter', 'none');
title('Relative Mean Difference');
legend('Location', 'best');
grid on;

subplot(2, 2, 4);
% Coefficient of variation comparison
for i_P = 1:n_P
    cv_power  = 100 * metrics.std_power_t(i_P, :) ./ metrics.mean_power_t(i_P, :);
    cv_linear = 100 * metrics.std_linear_t(i_P, :) ./ metrics.mean_linear_t(i_P, :);

    plot(Time_vec, cv_power, '-', 'LineWidth', 2, 'Color', colors_P(i_P,:), ...
        'DisplayName', sprintf('Power P=%.0f', P_levels(i_P)));
    hold on;
    plot(Time_vec, cv_linear, '--', 'LineWidth', 1.5, 'Color', colors_P(i_P,:), ...
        'HandleVisibility', 'off');
end
xlabel('Time [min]');
ylabel('CV [%%]', 'Interpreter', 'none');
title('Coefficient of Variation (solid=Power, dashed=Linear)');
legend('Location', 'best');
grid on;

sgtitle('Trajectory Difference Analysis', 'FontSize', 14);

%% Figure 8: Local discrimination maps (T, F) for different times and pressures
% Compute point-wise absolute difference |Y_power - Y_linear| for each sample
% This shows where in (T, F) space the models differ most

fprintf('Computing local discrimination maps...\n');

% Select time points for visualization
t_map_select = [round(N_Time/4), round(N_Time/2), round(3*N_Time/4), N_Time];
t_map_select = unique(t_map_select);
n_t_select = length(t_map_select);

% Create grid for interpolation
n_grid = 100;
T_grid_vals = linspace(T_min, T_max, n_grid);
F_grid_vals = linspace(F_min, F_max, n_grid);
[T_grid, F_grid] = meshgrid(T_grid_vals, F_grid_vals);

% Store local difference maps
local_diff_maps = cell(n_P, n_t_select);

for i_P = 1:n_P
    Y_power_all  = Y_traj_power{i_P};
    Y_linear_all = Y_traj_linear{i_P};

    % Get valid samples
    valid_idx = ~any(isnan(Y_power_all), 2) & ~any(isnan(Y_linear_all), 2);
    T_valid = T_samples(valid_idx);
    F_valid = F_samples(valid_idx);
    Y_power_valid = Y_power_all(valid_idx, :);
    Y_linear_valid = Y_linear_all(valid_idx, :);

    for j = 1:n_t_select
        i_t = t_map_select(j);

        % Compute absolute difference at this time point for each sample
        diff_abs = abs(Y_power_valid(:, i_t) - Y_linear_valid(:, i_t));

        % Interpolate to grid using scattered interpolation
        F_interp = scatteredInterpolant(T_valid, F_valid, diff_abs, 'natural', 'nearest');
        local_diff_maps{i_P, j} = F_interp(T_grid, F_grid);
    end
end

% Figure 8a: Heatmaps for each pressure at selected times
figure('Name', 'Local Discrimination Maps by Pressure', 'Position', [100 100 1600 900]);

for i_P = 1:min(n_P, 5)
    for j = 1:n_t_select
        subplot(min(n_P, 5), n_t_select, (i_P-1)*n_t_select + j);

        imagesc(T_grid_vals - 273, F_grid_vals * 1e5 * 60, local_diff_maps{i_P, j});
        set(gca, 'YDir', 'normal');
        colormap(gca, 'hot');

        if i_P == 1
            title(sprintf('t = %.0f min', Time_vec(t_map_select(j))));
        end
        if j == 1
            ylabel(sprintf('P=%.0f bar\nF [mL/min]', P_levels(i_P)));
        else
            ylabel('F [mL/min]');
        end
        if i_P == min(n_P, 5)
            xlabel('T [C]');
        end

        if j == n_t_select
            cb = colorbar;
            cb.Label.String = '|Y_p - Y_l| [g]';
        end
    end
end

sgtitle('Local Model Difference |Y_{Power} - Y_{Linear}| in (T, F) Space', 'FontSize', 14);

% Figure 8b: Heatmaps for selected pressure showing time evolution
[~, idx_best_P_map] = max(metrics.ks_integrated);
P_best_map = P_levels(idx_best_P_map);

% More time points for detailed evolution
t_evolution = round(linspace(1, N_Time, 9));
t_evolution = unique(t_evolution);

figure('Name', sprintf('Discrimination Evolution at P = %.0f bar', P_best_map), 'Position', [150 150 1400 900]);

Y_power_best = Y_traj_power{idx_best_P_map};
Y_linear_best = Y_traj_linear{idx_best_P_map};
valid_idx_best = ~any(isnan(Y_power_best), 2) & ~any(isnan(Y_linear_best), 2);
T_valid_best = T_samples(valid_idx_best);
F_valid_best = F_samples(valid_idx_best);
Y_power_valid_best = Y_power_best(valid_idx_best, :);
Y_linear_valid_best = Y_linear_best(valid_idx_best, :);

% Find global color scale
all_diffs = [];
for k = 1:length(t_evolution)
    diff_k = abs(Y_power_valid_best(:, t_evolution(k)) - Y_linear_valid_best(:, t_evolution(k)));
    all_diffs = [all_diffs; diff_k];
end
clim_max = prctile(all_diffs, 98);

n_rows = ceil(sqrt(length(t_evolution)));
n_cols = ceil(length(t_evolution) / n_rows);

for k = 1:length(t_evolution)
    subplot(n_rows, n_cols, k);

    i_t = t_evolution(k);
    diff_abs = abs(Y_power_valid_best(:, i_t) - Y_linear_valid_best(:, i_t));

    % Scatter plot with color coding
    scatter(T_valid_best - 273, F_valid_best * 1e5 * 60, 30, diff_abs, 'filled');

    colormap(gca, 'hot');
    caxis([0, clim_max]);

    xlabel('T [C]');
    ylabel('F [mL/min]');
    title(sprintf('t = %.0f min', Time_vec(i_t)));
    grid on;

    if k == length(t_evolution)
        cb = colorbar;
        cb.Label.String = '|Y_p - Y_l| [g]';
    end
end

sgtitle(sprintf('Model Difference Evolution at P = %.0f bar', P_best_map), 'FontSize', 14);

% Figure 8c: Relative difference maps (percentage)
figure('Name', 'Relative Local Discrimination Maps', 'Position', [200 200 1600 700]);

% Select fewer pressures and times for clarity
P_select_idx = [1, round(n_P/2), n_P];  % Low, mid, high pressure
t_rel_select = [round(N_Time/3), round(2*N_Time/3), N_Time];
t_rel_select = unique(t_rel_select);

for ip = 1:length(P_select_idx)
    i_P = P_select_idx(ip);

    Y_power_all = Y_traj_power{i_P};
    Y_linear_all = Y_traj_linear{i_P};
    valid_idx = ~any(isnan(Y_power_all), 2) & ~any(isnan(Y_linear_all), 2);
    T_valid = T_samples(valid_idx);
    F_valid = F_samples(valid_idx);
    Y_power_valid = Y_power_all(valid_idx, :);
    Y_linear_valid = Y_linear_all(valid_idx, :);

    for jt = 1:length(t_rel_select)
        subplot(length(P_select_idx), length(t_rel_select), (ip-1)*length(t_rel_select) + jt);

        i_t = t_rel_select(jt);

        % Compute relative difference
        y_avg = 0.5 * (Y_power_valid(:, i_t) + Y_linear_valid(:, i_t));
        rel_diff = 100 * abs(Y_power_valid(:, i_t) - Y_linear_valid(:, i_t)) ./ max(y_avg, 1e-6);

        % Scatter plot
        scatter(T_valid - 273, F_valid * 1e5 * 60, 30, rel_diff, 'filled');

        colormap(gca, 'parula');
        caxis([0, min(max(rel_diff), 50)]);  % Cap at 50% for visualization

        if ip == 1
            title(sprintf('t = %.0f min', Time_vec(i_t)));
        end
        if jt == 1
            ylabel(sprintf('P=%.0f bar\nF [mL/min]', P_levels(i_P)));
        else
            ylabel('F [mL/min]');
        end
        if ip == length(P_select_idx)
            xlabel('T [C]');
        end

        if jt == length(t_rel_select)
            cb = colorbar;
            cb.Label.String = 'Rel. Diff. [%%]';
            cb.Label.Interpreter = 'none';
        end

        grid on;
    end
end

sgtitle('Relative Model Difference in (T, F) Space', 'FontSize', 14, 'Interpreter', 'none');

%% Figure 9: Local KS statistic map in (T, F) space using binning
% Compute KS statistic within local bins of (T, F) space
% This requires sufficient samples in each bin

fprintf('Computing local KS statistic maps...\n');

% Define bin edges for T and F
n_bins_T = 8;
n_bins_F = 8;
T_edges = linspace(T_min, T_max, n_bins_T + 1);
F_edges = linspace(F_min, F_max, n_bins_F + 1);

% Bin centers for plotting
T_centers = 0.5 * (T_edges(1:end-1) + T_edges(2:end));
F_centers = 0.5 * (F_edges(1:end-1) + F_edges(2:end));

% Select time points for KS maps
t_ks_select = [round(N_Time/4), round(N_Time/2), round(3*N_Time/4), N_Time];
t_ks_select = unique(t_ks_select);
n_t_ks = length(t_ks_select);

% Store local KS maps: KS_local(i_P, i_t, i_T_bin, i_F_bin)
KS_local_maps = nan(n_P, n_t_ks, n_bins_T, n_bins_F);
min_samples_per_bin = 5;  % Minimum samples needed for KS test

for i_P = 1:n_P
    Y_power_all  = Y_traj_power{i_P};
    Y_linear_all = Y_traj_linear{i_P};

    % Get valid samples
    valid_idx = ~any(isnan(Y_power_all), 2) & ~any(isnan(Y_linear_all), 2);
    T_valid = T_samples(valid_idx);
    F_valid = F_samples(valid_idx);
    Y_power_valid = Y_power_all(valid_idx, :);
    Y_linear_valid = Y_linear_all(valid_idx, :);

    for jt = 1:n_t_ks
        i_t = t_ks_select(jt);

        for iT = 1:n_bins_T
            for iF = 1:n_bins_F
                % Find samples in this bin
                in_bin = (T_valid >= T_edges(iT)) & (T_valid < T_edges(iT+1)) & ...
                         (F_valid >= F_edges(iF)) & (F_valid < F_edges(iF+1));

                if sum(in_bin) >= min_samples_per_bin
                    y_p = Y_power_valid(in_bin, i_t);
                    y_l = Y_linear_valid(in_bin, i_t);

                    % Compute two-sample KS statistic
                    [~, ~, ks_stat] = kstest2(y_p, y_l);
                    KS_local_maps(i_P, jt, iT, iF) = ks_stat;
                end
            end
        end
    end
end

% Figure 9a: Pcolor plots of local KS for each pressure and time
figure('Name', 'Local KS Statistic Maps (T, F)', 'Position', [100 100 1600 900]);

[T_plot, F_plot] = meshgrid(T_centers - 273, F_centers * 1e5 * 60);

for i_P = 1:min(n_P, 5)
    for jt = 1:n_t_ks
        subplot(min(n_P, 5), n_t_ks, (i_P-1)*n_t_ks + jt);

        KS_data = squeeze(KS_local_maps(i_P, jt, :, :))';

        pcolor(T_centers - 273, F_centers * 1e5 * 60, KS_data);
        shading interp;
        colormap(gca, 'jet');
        caxis([0, 1]);

        if i_P == 1
            title(sprintf('t = %.0f min', Time_vec(t_ks_select(jt))));
        end
        if jt == 1
            ylabel(sprintf('P=%.0f bar\nF [mL/min]', P_levels(i_P)));
        else
            ylabel('F [mL/min]');
        end
        if i_P == min(n_P, 5)
            xlabel('T [C]');
        end

        if jt == n_t_ks
            cb = colorbar;
            cb.Label.String = 'KS statistic';
        end
    end
end

sgtitle('Local Kolmogorov-Smirnov Statistic in (T, F) Space', 'FontSize', 14);

% Figure 9b: Detailed pcolor for best pressure with more time resolution
[~, idx_best_P_ks] = max(metrics.ks_integrated);

% Recompute with finer time resolution for best pressure
t_fine = round(linspace(1, N_Time, 12));
t_fine = unique(t_fine);
n_t_fine = length(t_fine);

KS_fine = nan(n_t_fine, n_bins_T, n_bins_F);

Y_power_best = Y_traj_power{idx_best_P_ks};
Y_linear_best = Y_traj_linear{idx_best_P_ks};
valid_idx_best = ~any(isnan(Y_power_best), 2) & ~any(isnan(Y_linear_best), 2);
T_valid_best = T_samples(valid_idx_best);
F_valid_best = F_samples(valid_idx_best);
Y_power_valid_best = Y_power_best(valid_idx_best, :);
Y_linear_valid_best = Y_linear_best(valid_idx_best, :);

for jt = 1:n_t_fine
    i_t = t_fine(jt);

    for iT = 1:n_bins_T
        for iF = 1:n_bins_F
            in_bin = (T_valid_best >= T_edges(iT)) & (T_valid_best < T_edges(iT+1)) & ...
                     (F_valid_best >= F_edges(iF)) & (F_valid_best < F_edges(iF+1));

            if sum(in_bin) >= min_samples_per_bin
                y_p = Y_power_valid_best(in_bin, i_t);
                y_l = Y_linear_valid_best(in_bin, i_t);
                [~, ~, ks_stat] = kstest2(y_p, y_l);
                KS_fine(jt, iT, iF) = ks_stat;
            end
        end
    end
end

figure('Name', sprintf('Local KS Evolution at P = %.0f bar', P_levels(idx_best_P_ks)), 'Position', [150 150 1500 900]);

n_rows_fine = 3;
n_cols_fine = 4;

for jt = 1:n_t_fine
    subplot(n_rows_fine, n_cols_fine, jt);

    KS_data = squeeze(KS_fine(jt, :, :))';

    pcolor(T_centers - 273, F_centers * 1e5 * 60, KS_data);
    shading interp;
    colormap(gca, 'jet');
    caxis([0, 1]);

    xlabel('T [C]');
    ylabel('F [mL/min]');
    title(sprintf('t = %.0f min', Time_vec(t_fine(jt))));

    if jt == n_t_fine
        cb = colorbar;
        cb.Label.String = 'KS statistic';
    end
end

sgtitle(sprintf('Local KS Statistic Evolution at P = %.0f bar', P_levels(idx_best_P_ks)), 'FontSize', 14);

% Figure 9c: Time-averaged KS map for each pressure
figure('Name', 'Time-Averaged Local KS Maps', 'Position', [200 200 1400 400]);

for i_P = 1:n_P
    subplot(1, n_P, i_P);

    % Average KS over all time points
    KS_avg = squeeze(nanmean(KS_local_maps(i_P, :, :, :), 2))';

    pcolor(T_centers - 273, F_centers * 1e5 * 60, KS_avg);
    shading interp;
    colormap(gca, 'jet');
    caxis([0, 0.8]);

    xlabel('T [C]');
    ylabel('F [mL/min]');
    title(sprintf('P = %.0f bar', P_levels(i_P)));

    cb = colorbar;
    cb.Label.String = 'Mean KS';
end

sgtitle('Time-Averaged Local KS Statistic in (T, F) Space', 'FontSize', 14);

% Figure 9d: Surface plot of KS(T, F) at final time for best pressure
figure('Name', 'KS Surface Plot', 'Position', [250 250 800 600]);

KS_final = squeeze(KS_local_maps(idx_best_P_ks, end, :, :))';

surf(T_centers - 273, F_centers * 1e5 * 60, KS_final);
shading interp;
colormap('jet');
colorbar;

xlabel('T [C]');
ylabel('F [mL/min]');
zlabel('KS statistic');
title(sprintf('Local KS at P = %.0f bar, t = %.0f min', P_levels(idx_best_P_ks), Time_vec(t_ks_select(end))));
view(45, 30);
grid on;

%% ========================================================================
%  PRINT SUMMARY TABLE
%  ========================================================================
fprintf('\n');
fprintf('=============================================================================\n');
fprintf('                    TRAJECTORY COMPARISON SUMMARY                            \n');
fprintf('=============================================================================\n');
fprintf('%-8s %-12s %-10s %-12s %-12s %-12s %-12s\n', ...
    'P [bar]', 'Int. KS', 'Max KS', 't_max [min]', 'Int. W1', 'Int. KL', 'RMSE [g]');
fprintf('-----------------------------------------------------------------------------\n');

for i_P = 1:n_P
    fprintf('%-8.0f %-12.4f %-10.4f %-12.0f %-12.4f %-12.4f %-12.4f\n', ...
        P_levels(i_P), ...
        metrics.ks_integrated(i_P), ...
        metrics.ks_max(i_P), ...
        metrics.ks_max_time(i_P), ...
        metrics.wasserstein_integrated(i_P), ...
        metrics.kl_integrated(i_P), ...
        metrics.trajectory_rmse(i_P));
end

fprintf('=============================================================================\n');

%% Interpretation
fprintf('\n=== INTERPRETATION ===\n\n');

% Find optimal conditions
[~, idx_best_int_ks] = max(metrics.ks_integrated);
[~, idx_best_int_kl] = max(metrics.kl_integrated);
[~, idx_best_int_w1] = max(metrics.wasserstein_integrated);
[~, idx_best_max_ks] = max(metrics.ks_max);
[~, idx_best_area]   = max(metrics.area_between_curves);
[~, idx_best_rmse]   = max(metrics.trajectory_rmse);

fprintf('Best pressure for trajectory-based discrimination:\n');
fprintf('  Integrated KS:      P = %.0f bar (Int. KS = %.4f)\n', ...
    P_levels(idx_best_int_ks), metrics.ks_integrated(idx_best_int_ks));
fprintf('  Integrated KL:      P = %.0f bar (Int. KL = %.4f)\n', ...
    P_levels(idx_best_int_kl), metrics.kl_integrated(idx_best_int_kl));
fprintf('  Integrated W1:      P = %.0f bar (Int. W1 = %.4f)\n', ...
    P_levels(idx_best_int_w1), metrics.wasserstein_integrated(idx_best_int_w1));
fprintf('  Maximum KS:         P = %.0f bar (Max KS = %.4f at t = %.0f min)\n', ...
    P_levels(idx_best_max_ks), metrics.ks_max(idx_best_max_ks), metrics.ks_max_time(idx_best_max_ks));
fprintf('  Area between curves: P = %.0f bar (Area = %.4f g*min)\n', ...
    P_levels(idx_best_area), metrics.area_between_curves(idx_best_area));
fprintf('  Trajectory RMSE:    P = %.0f bar (RMSE = %.4f g)\n', ...
    P_levels(idx_best_rmse), metrics.trajectory_rmse(idx_best_rmse));

% Comment on consistency of metrics
if idx_best_int_ks == idx_best_int_kl && idx_best_int_ks == idx_best_int_w1
    fprintf('\nConsensus: KS, KL, and W1 agree on P = %.0f bar as the most discriminative.\n', ...
        P_levels(idx_best_int_ks));
else
    fprintf('\nConsensus check: KS->P=%.0f, KL->P=%.0f, W1->P=%.0f bar.\n', ...
        P_levels(idx_best_int_ks), P_levels(idx_best_int_kl), P_levels(idx_best_int_w1));
end

% Strength of separation (best vs second-best)
ks_sorted = sort(metrics.ks_integrated, 'descend');
kl_sorted = sort(metrics.kl_integrated, 'descend');
w1_sorted = sort(metrics.wasserstein_integrated, 'descend');

if numel(ks_sorted) > 1 && ks_sorted(1) >= 1.2 * ks_sorted(2)
    fprintf('Integrated KS shows a clear winner (top >= 1.2x second).\n');
else
    fprintf('Integrated KS shows comparable pressures (top < 1.2x second).\n');
end

if numel(kl_sorted) > 1 && kl_sorted(1) >= 1.2 * kl_sorted(2)
    fprintf('Integrated KL shows a clear winner (top >= 1.2x second).\n');
else
    fprintf('Integrated KL shows comparable pressures (top < 1.2x second).\n');
end

if numel(w1_sorted) > 1 && w1_sorted(1) >= 1.2 * w1_sorted(2)
    fprintf('Integrated W1 shows a clear winner (top >= 1.2x second).\n');
else
    fprintf('Integrated W1 shows comparable pressures (top < 1.2x second).\n');
end

fprintf('\nTime of maximum discrimination (by pressure):\n');
for i_P = 1:n_P
    fprintf('  P = %.0f bar: max KS at t = %.0f min (KS = %.4f)\n', ...
        P_levels(i_P), metrics.ks_max_time(i_P), metrics.ks_max(i_P));
end

% Early vs late separation
half_time = Time_vec(end) / 2;
early_max = sum(metrics.ks_max_time <= half_time);
late_max = sum(metrics.ks_max_time > half_time);
fprintf('\nTiming of peak separation: %d pressures peak early (t <= %.0f min), %d peak late.\n', ...
    early_max, half_time, late_max);

% Significance analysis
fprintf('\nFraction of time points with significant difference (p < 0.05):\n');
for i_P = 1:n_P
    frac_sig = sum(metrics.ks_p_t(i_P, :) < 0.05) / N_Time * 100;
    fprintf('  P = %.0f bar: %.1f%% of time points\n', P_levels(i_P), frac_sig);
end

% Comment on strongest significance
frac_sig_all = sum(metrics.ks_p_t < 0.05, 2) / N_Time * 100;
[max_sig, idx_max_sig] = max(frac_sig_all);
fprintf('\nStrongest overall significance: P = %.0f bar (%.1f%% of time points).\n', ...
    P_levels(idx_max_sig), max_sig);

%% Identify regions in (T, F) with strongest discrimination
fprintf('\n=== Discrimination Hot-Spots in (T, F) ===\n');

top_pct = 90;  % top percentile for strongest discrimination
min_top_n = 8; % minimum samples to summarize region

for i_P = 1:n_P
    Y_power_all  = Y_traj_power{i_P};
    Y_linear_all = Y_traj_linear{i_P};

    valid_idx = ~any(isnan(Y_power_all), 2) & ~any(isnan(Y_linear_all), 2);
    if sum(valid_idx) < min_top_n
        fprintf('P = %.0f bar: insufficient valid samples for region summary.\n', P_levels(i_P));
        continue;
    end

    T_valid = T_samples(valid_idx);
    F_valid = F_samples(valid_idx);

    % Integrated absolute discrimination over time per sample
    diff_abs = abs(Y_power_all(valid_idx, :) - Y_linear_all(valid_idx, :));
    disc_int = trapz(Time_vec, diff_abs, 2);

    thr_int = prctile(disc_int, top_pct);
    top_idx = disc_int >= thr_int;
    if sum(top_idx) < min_top_n
        thr_int = prctile(disc_int, 80);
        top_idx = disc_int >= thr_int;
    end

    T_top = T_valid(top_idx) - 273;
    F_top = F_valid(top_idx) * 1e5 * 60;

    if ~isempty(T_top)
        fprintf('P = %.0f bar (integrated |ΔY|): T = %.1f–%.1f C (p10–p90), ', P_levels(i_P), ...
            prctile(T_top, 10), prctile(T_top, 90));
        fprintf('F = %.2f–%.2f mL/min (p10–p90), n = %d\n', ...
            prctile(F_top, 10), prctile(F_top, 90), numel(T_top));
    end

    % Region at time of maximum KS
    [~, t_idx] = min(abs(Time_vec - metrics.ks_max_time(i_P)));
    diff_t = abs(Y_power_all(valid_idx, t_idx) - Y_linear_all(valid_idx, t_idx));

    thr_t = prctile(diff_t, top_pct);
    top_idx_t = diff_t >= thr_t;
    if sum(top_idx_t) < min_top_n
        thr_t = prctile(diff_t, 80);
        top_idx_t = diff_t >= thr_t;
    end

    T_top_t = T_valid(top_idx_t) - 273;
    F_top_t = F_valid(top_idx_t) * 1e5 * 60;

    if ~isempty(T_top_t)
        fprintf('  t = %.0f min (max KS): T = %.1f–%.1f C (p10–p90), F = %.2f–%.2f mL/min (p10–p90), n = %d\n', ...
            Time_vec(t_idx), prctile(T_top_t, 10), prctile(T_top_t, 90), ...
            prctile(F_top_t, 10), prctile(F_top_t, 90), numel(T_top_t));
    end
end

% Highlight the most discriminative pressure by integrated KS
best_P = P_levels(idx_best_int_ks);
fprintf('\nOverall best discrimination by integrated KS: P = %.0f bar.\n', best_P);

%% Save results
save('distribution_comparison_trajectory_results.mat', 'results', 'metrics');
fprintf('\nResults saved to distribution_comparison_trajectory_results.mat\n');

fprintf('\n=== Trajectory-Based Analysis Complete ===\n');

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================
function kl_val = compute_kl_divergence(y_p, y_l)
    % Compute KL divergence between two 1D samples using KDE on a common grid.
    y_all = [y_p; y_l];
    if numel(y_all) < 3 || all(y_all == y_all(1))
        kl_val = 0;
        return;
    end

    y_min = min(y_all);
    y_max = max(y_all);
    if y_max <= y_min
        kl_val = 0;
        return;
    end

    y_grid = linspace(y_min, y_max, 200);
    f_p = ksdensity(y_p, y_grid, 'Function', 'pdf');
    f_l = ksdensity(y_l, y_grid, 'Function', 'pdf');

    eps_val = 1e-12;
    f_p = max(f_p, eps_val);
    f_l = max(f_l, eps_val);

    kl_val = trapz(y_grid, f_p .* log(f_p ./ f_l));  % KL(Power || Linear)
end
