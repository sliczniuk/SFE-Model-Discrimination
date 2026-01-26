%% Temporal Distribution-Based Model Comparison for SFE Models
% This script compares Power and Linear model output distributions
% at multiple extraction times to analyze how discrimination evolves.
%
% Extraction times: 60, 150, 300, 600 minutes
% Method: Latin Hypercube Sampling (LHS) for efficient coverage of (T, F) space
% Output: Distribution comparison metrics at each pressure level and time

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
N_samples = 1000;  % Reduced for temporal analysis (5 pressures x 4 times)
P_levels = [100, 125, 150, 175, 200];  % Fixed pressure levels [bar]
ExtractionTimes = [60, 150, 300, 600];  % Extraction times to evaluate [min]

% Input ranges for uniform distributions
T_min = 303;  T_max = 313;  % Temperature [K] (30-40 C)
F_min = 3.3e-5; F_max = 6.7e-5; % Flow rate [m3/s]

fprintf('=== Temporal Distribution-Based Model Comparison ===\n\n');
fprintf('Configuration:\n');
fprintf('  N_samples: %d (LHS)\n', N_samples);
fprintf('  Pressure levels: %s bar\n', mat2str(P_levels));
fprintf('  Extraction times: %s min\n', mat2str(ExtractionTimes));
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
n_T_extract = length(ExtractionTimes);

% Store constants for parallel workers
Parameters_mat = cell2mat(Parameters);

% Results structures for all times
all_results = struct();
all_metrics = struct();

%% ========================================================================
%  MAIN SIMULATION LOOP - ITERATE OVER EXTRACTION TIMES
%  ========================================================================
fprintf('\n=== Starting temporal analysis ===\n');
total_tic = tic;

for i_time = 1:n_T_extract
    ExtractionTime = ExtractionTimes(i_time);
    fprintf('\n########## EXTRACTION TIME: %d minutes ##########\n', ExtractionTime);

    %% Time Configuration for this extraction time
    PreparationTime = 0;
    timeStep        = 5;  % [minutes]

    simulationTime   = PreparationTime + ExtractionTime;
    timeStep_in_sec  = timeStep * 60;
    Time_in_sec      = (timeStep:timeStep:simulationTime) * 60;
    Time             = [0 Time_in_sec/60];
    N_Time           = length(Time_in_sec);

    fprintf('  Time steps: %d, Final time: %.0f min\n', N_Time, Time(end));

    % Preallocate for this extraction time
    Y_final_power  = zeros(N_samples, n_P);
    Y_final_linear = zeros(N_samples, n_P);

    tic;

    for i_P = 1:n_P
        P0 = P_levels(i_P);
        fprintf('  --- Pressure: %.0f bar ---\n', P0);

        % Preallocate temporary arrays
        Y_linear_temp = zeros(N_samples, 1);
        Y_power_temp  = zeros(N_samples, 1);

        % Create batch indices
        batch_size = ceil(N_samples / n_workers);
        batch_starts = 1:batch_size:N_samples;
        batch_ends = min(batch_starts + batch_size - 1, N_samples);
        n_batches_actual = length(batch_starts);

        % Cell arrays to collect batch results
        Y_linear_batches = cell(n_batches_actual, 1);
        Y_power_batches  = cell(n_batches_actual, 1);

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

            % Preallocate batch results
            Y_linear_batch = zeros(n_batch, 1);
            Y_power_batch  = zeros(n_batch, 1);

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

                    Y_linear_batch(j) = full(X_all_linear(end, end));
                    Y_power_batch(j)  = full(X_all_power(end, end));
                catch ME
                    Y_linear_batch(j) = NaN;
                    Y_power_batch(j)  = NaN;
                end
            end

            Y_linear_batches{i_batch} = Y_linear_batch;
            Y_power_batches{i_batch}  = Y_power_batch;
        end

        % Reassemble results from batches
        for i_batch = 1:n_batches_actual
            idx_start = batch_starts(i_batch);
            idx_end   = batch_ends(i_batch);
            Y_linear_temp(idx_start:idx_end) = Y_linear_batches{i_batch};
            Y_power_temp(idx_start:idx_end)  = Y_power_batches{i_batch};
        end

        Y_final_linear(:, i_P) = Y_linear_temp;
        Y_final_power(:, i_P)  = Y_power_temp;

        n_valid = sum(~isnan(Y_linear_temp));
        fprintf('    Completed: %d/%d valid samples\n', n_valid, N_samples);
    end

    elapsed_time = toc;
    fprintf('  Time = %d min completed in %.1f seconds\n', ExtractionTime, elapsed_time);

    %% Compute metrics for this extraction time
    metrics = struct();
    metrics.P_levels = P_levels;
    metrics.ExtractionTime = ExtractionTime;

    for i_P = 1:n_P
        Y_power  = Y_final_power(:, i_P);
        Y_linear = Y_final_linear(:, i_P);

        valid_idx = ~isnan(Y_power) & ~isnan(Y_linear);
        Y_power  = Y_power(valid_idx);
        Y_linear = Y_linear(valid_idx);
        n_valid  = sum(valid_idx);

        % Basic Statistics
        metrics.mean_power(i_P)   = mean(Y_power);
        metrics.mean_linear(i_P)  = mean(Y_linear);
        metrics.std_power(i_P)    = std(Y_power);
        metrics.std_linear(i_P)   = std(Y_linear);

        % Two-sample t-test
        [~, p_ttest, ~, stats_ttest] = ttest2(Y_power, Y_linear);
        metrics.ttest_t(i_P)  = stats_ttest.tstat;
        metrics.ttest_p(i_P)  = p_ttest;

        % Kolmogorov-Smirnov Test
        [~, p_ks, ks_stat] = kstest2(Y_power, Y_linear);
        metrics.ks_stat(i_P) = ks_stat;
        metrics.ks_p(i_P)    = p_ks;

        % Wasserstein Distance
        Y_power_sorted  = sort(Y_power);
        Y_linear_sorted = sort(Y_linear);
        if length(Y_power_sorted) == length(Y_linear_sorted)
            metrics.wasserstein(i_P) = mean(abs(Y_power_sorted - Y_linear_sorted));
        else
            n_interp = max(length(Y_power), length(Y_linear));
            cdf_power  = interp1(linspace(0,1,length(Y_power_sorted)), Y_power_sorted, linspace(0,1,n_interp));
            cdf_linear = interp1(linspace(0,1,length(Y_linear_sorted)), Y_linear_sorted, linspace(0,1,n_interp));
            metrics.wasserstein(i_P) = mean(abs(cdf_power - cdf_linear));
        end

        % Jensen-Shannon Divergence
        n_bins = 50;
        edges = linspace(min([Y_power; Y_linear]), max([Y_power; Y_linear]), n_bins + 1);
        hist_power  = histcounts(Y_power, edges, 'Normalization', 'probability');
        hist_linear = histcounts(Y_linear, edges, 'Normalization', 'probability');

        eps_val = 1e-10;
        hist_power  = hist_power + eps_val;
        hist_linear = hist_linear + eps_val;
        hist_power  = hist_power / sum(hist_power);
        hist_linear = hist_linear / sum(hist_linear);

        M = 0.5 * (hist_power + hist_linear);
        KL_PM = sum(hist_power .* log(hist_power ./ M));
        KL_QM = sum(hist_linear .* log(hist_linear ./ M));
        metrics.js_divergence(i_P) = 0.5 * KL_PM + 0.5 * KL_QM;

        % Cohen's d
        pooled_std = sqrt(((n_valid-1)*std(Y_power)^2 + (n_valid-1)*std(Y_linear)^2) / (2*n_valid - 2));
        metrics.cohens_d(i_P) = (mean(Y_power) - mean(Y_linear)) / pooled_std;

        % Bhattacharyya Coefficient
        BC = sum(sqrt(hist_power .* hist_linear));
        metrics.bhattacharyya_coef(i_P) = BC;
        if BC > 0
            metrics.bhattacharyya_dist(i_P) = -log(BC);
        else
            metrics.bhattacharyya_dist(i_P) = Inf;
        end
    end

    % Store results for this extraction time
    all_results(i_time).ExtractionTime = ExtractionTime;
    all_results(i_time).Y_final_power  = Y_final_power;
    all_results(i_time).Y_final_linear = Y_final_linear;
    all_results(i_time).metrics = metrics;
    all_results(i_time).Time = Time;
end

total_elapsed = toc(total_tic);
fprintf('\n=== All simulations completed in %.1f minutes ===\n', total_elapsed/60);

%% ========================================================================
%  TEMPORAL ANALYSIS - COMPARE METRICS ACROSS TIME
%  ========================================================================
fprintf('\n=== TEMPORAL COMPARISON OF METRICS ===\n\n');

% Extract metrics across times for each pressure
ks_stat_matrix     = zeros(n_T_extract, n_P);
js_div_matrix      = zeros(n_T_extract, n_P);
wasserstein_matrix = zeros(n_T_extract, n_P);
cohens_d_matrix    = zeros(n_T_extract, n_P);
ttest_t_matrix     = zeros(n_T_extract, n_P);
mean_diff_matrix   = zeros(n_T_extract, n_P);

for i_time = 1:n_T_extract
    ks_stat_matrix(i_time, :)     = all_results(i_time).metrics.ks_stat;
    js_div_matrix(i_time, :)      = all_results(i_time).metrics.js_divergence;
    wasserstein_matrix(i_time, :) = all_results(i_time).metrics.wasserstein;
    cohens_d_matrix(i_time, :)    = all_results(i_time).metrics.cohens_d;
    ttest_t_matrix(i_time, :)     = all_results(i_time).metrics.ttest_t;
    mean_diff_matrix(i_time, :)   = all_results(i_time).metrics.mean_power - all_results(i_time).metrics.mean_linear;
end

%% Print summary tables
fprintf('=============================================================================\n');
fprintf('                    KS STATISTIC vs TIME and PRESSURE                        \n');
fprintf('=============================================================================\n');
fprintf('%-10s', 'Time [min]');
for i_P = 1:n_P
    fprintf('P=%-7.0f', P_levels(i_P));
end
fprintf('\n');
fprintf('-----------------------------------------------------------------------------\n');
for i_time = 1:n_T_extract
    fprintf('%-10d', ExtractionTimes(i_time));
    for i_P = 1:n_P
        fprintf('%-9.4f', ks_stat_matrix(i_time, i_P));
    end
    fprintf('\n');
end
fprintf('=============================================================================\n\n');

fprintf('=============================================================================\n');
fprintf('                    WASSERSTEIN DISTANCE vs TIME and PRESSURE                \n');
fprintf('=============================================================================\n');
fprintf('%-10s', 'Time [min]');
for i_P = 1:n_P
    fprintf('P=%-7.0f', P_levels(i_P));
end
fprintf('\n');
fprintf('-----------------------------------------------------------------------------\n');
for i_time = 1:n_T_extract
    fprintf('%-10d', ExtractionTimes(i_time));
    for i_P = 1:n_P
        fprintf('%-9.4f', wasserstein_matrix(i_time, i_P));
    end
    fprintf('\n');
end
fprintf('=============================================================================\n\n');

fprintf('=============================================================================\n');
fprintf('                    COHEN''S d vs TIME and PRESSURE                          \n');
fprintf('=============================================================================\n');
fprintf('%-10s', 'Time [min]');
for i_P = 1:n_P
    fprintf('P=%-7.0f', P_levels(i_P));
end
fprintf('\n');
fprintf('-----------------------------------------------------------------------------\n');
for i_time = 1:n_T_extract
    fprintf('%-10d', ExtractionTimes(i_time));
    for i_P = 1:n_P
        fprintf('%-9.4f', cohens_d_matrix(i_time, i_P));
    end
    fprintf('\n');
end
fprintf('=============================================================================\n\n');

fprintf('=============================================================================\n');
fprintf('                    MEAN DIFFERENCE (Power - Linear) vs TIME                 \n');
fprintf('=============================================================================\n');
fprintf('%-10s', 'Time [min]');
for i_P = 1:n_P
    fprintf('P=%-7.0f', P_levels(i_P));
end
fprintf('\n');
fprintf('-----------------------------------------------------------------------------\n');
for i_time = 1:n_T_extract
    fprintf('%-10d', ExtractionTimes(i_time));
    for i_P = 1:n_P
        fprintf('%-9.4f', mean_diff_matrix(i_time, i_P));
    end
    fprintf('\n');
end
fprintf('=============================================================================\n\n');

%% ========================================================================
%  VISUALIZATION - TEMPORAL EVOLUTION
%  ========================================================================
fprintf('Generating temporal visualizations...\n');

colors_P = lines(n_P);

%% Figure 1: KS Statistic vs Time for each Pressure
figure('Name', 'Temporal Evolution of KS Statistic', 'Position', [100 100 1000 600]);

subplot(2, 2, 1);
for i_P = 1:n_P
    plot(ExtractionTimes, ks_stat_matrix(:, i_P), 'o-', 'LineWidth', 2, ...
        'Color', colors_P(i_P,:), 'MarkerFaceColor', colors_P(i_P,:), ...
        'DisplayName', sprintf('P = %.0f bar', P_levels(i_P)));
    hold on;
end
xlabel('Extraction Time [min]');
ylabel('KS Statistic');
title('Kolmogorov-Smirnov Statistic');
legend('Location', 'best');
grid on;

subplot(2, 2, 2);
for i_P = 1:n_P
    plot(ExtractionTimes, wasserstein_matrix(:, i_P), 's-', 'LineWidth', 2, ...
        'Color', colors_P(i_P,:), 'MarkerFaceColor', colors_P(i_P,:), ...
        'DisplayName', sprintf('P = %.0f bar', P_levels(i_P)));
    hold on;
end
xlabel('Extraction Time [min]');
ylabel('Wasserstein Distance');
title('Wasserstein (Earth Mover) Distance');
legend('Location', 'best');
grid on;

subplot(2, 2, 3);
for i_P = 1:n_P
    plot(ExtractionTimes, abs(cohens_d_matrix(:, i_P)), '^-', 'LineWidth', 2, ...
        'Color', colors_P(i_P,:), 'MarkerFaceColor', colors_P(i_P,:), ...
        'DisplayName', sprintf('P = %.0f bar', P_levels(i_P)));
    hold on;
end
yline(0.2, 'k--', 'Small', 'LabelHorizontalAlignment', 'left');
yline(0.5, 'k--', 'Medium', 'LabelHorizontalAlignment', 'left');
yline(0.8, 'k--', 'Large', 'LabelHorizontalAlignment', 'left');
xlabel('Extraction Time [min]');
ylabel('|Cohen''s d|');
title('Effect Size (Cohen''s d)');
legend('Location', 'best');
grid on;

subplot(2, 2, 4);
for i_P = 1:n_P
    plot(ExtractionTimes, js_div_matrix(:, i_P), 'd-', 'LineWidth', 2, ...
        'Color', colors_P(i_P,:), 'MarkerFaceColor', colors_P(i_P,:), ...
        'DisplayName', sprintf('P = %.0f bar', P_levels(i_P)));
    hold on;
end
xlabel('Extraction Time [min]');
ylabel('JS Divergence');
title('Jensen-Shannon Divergence');
legend('Location', 'best');
grid on;

sgtitle('Temporal Evolution of Model Discrimination Metrics', 'FontSize', 14);

%% Figure 2: Heatmaps of metrics
figure('Name', 'Metric Heatmaps', 'Position', [150 150 1200 500]);

subplot(1, 3, 1);
imagesc(P_levels, ExtractionTimes, ks_stat_matrix);
colorbar;
xlabel('Pressure [bar]');
ylabel('Extraction Time [min]');
title('KS Statistic');
set(gca, 'YDir', 'normal');
colormap(gca, 'hot');

subplot(1, 3, 2);
imagesc(P_levels, ExtractionTimes, wasserstein_matrix);
colorbar;
xlabel('Pressure [bar]');
ylabel('Extraction Time [min]');
title('Wasserstein Distance');
set(gca, 'YDir', 'normal');
colormap(gca, 'hot');

subplot(1, 3, 3);
imagesc(P_levels, ExtractionTimes, abs(cohens_d_matrix));
colorbar;
xlabel('Pressure [bar]');
ylabel('Extraction Time [min]');
title('|Cohen''s d|');
set(gca, 'YDir', 'normal');
colormap(gca, 'hot');

sgtitle('Model Discrimination Metrics: Time vs Pressure', 'FontSize', 14);

%% Figure 3: Distribution evolution at best pressure
[~, idx_best_P] = max(mean(ks_stat_matrix, 1));
P_best = P_levels(idx_best_P);

figure('Name', sprintf('Distribution Evolution at P = %.0f bar', P_best), 'Position', [200 200 1200 800]);

for i_time = 1:n_T_extract
    subplot(2, 2, i_time);

    Y_power  = all_results(i_time).Y_final_power(:, idx_best_P);
    Y_linear = all_results(i_time).Y_final_linear(:, idx_best_P);

    Y_power  = Y_power(~isnan(Y_power));
    Y_linear = Y_linear(~isnan(Y_linear));

    edges = linspace(min([Y_power; Y_linear])*0.95, max([Y_power; Y_linear])*1.05, 30);

    histogram(Y_power, edges, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'DisplayName', 'Power');
    hold on;
    histogram(Y_linear, edges, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'DisplayName', 'Linear');

    xlabel('Final Yield [g]');
    ylabel('Frequency');
    title(sprintf('t = %d min (KS=%.3f, d=%.2f)', ExtractionTimes(i_time), ...
        ks_stat_matrix(i_time, idx_best_P), cohens_d_matrix(i_time, idx_best_P)));
    legend('Location', 'best');
    grid on;
end

sgtitle(sprintf('Distribution Evolution at P = %.0f bar', P_best), 'FontSize', 14);

%% Figure 4: Mean yields evolution
figure('Name', 'Mean Yield Evolution', 'Position', [250 250 1000 600]);

subplot(1, 2, 1);
for i_P = 1:n_P
    mean_power = zeros(n_T_extract, 1);
    mean_linear = zeros(n_T_extract, 1);
    for i_time = 1:n_T_extract
        mean_power(i_time)  = all_results(i_time).metrics.mean_power(i_P);
        mean_linear(i_time) = all_results(i_time).metrics.mean_linear(i_P);
    end
    plot(ExtractionTimes, mean_power, 'o-', 'LineWidth', 2, 'Color', colors_P(i_P,:), ...
        'DisplayName', sprintf('Power P=%.0f', P_levels(i_P)));
    hold on;
    plot(ExtractionTimes, mean_linear, 's--', 'LineWidth', 2, 'Color', colors_P(i_P,:), ...
        'HandleVisibility', 'off');
end
xlabel('Extraction Time [min]');
ylabel('Mean Yield [g]');
title('Mean Yield vs Time (solid=Power, dashed=Linear)');
legend('Location', 'best');
grid on;

subplot(1, 2, 2);
for i_P = 1:n_P
    plot(ExtractionTimes, mean_diff_matrix(:, i_P), 'o-', 'LineWidth', 2, ...
        'Color', colors_P(i_P,:), 'MarkerFaceColor', colors_P(i_P,:), ...
        'DisplayName', sprintf('P = %.0f bar', P_levels(i_P)));
    hold on;
end
yline(0, 'k--');
xlabel('Extraction Time [min]');
ylabel('Mean Difference [g]');
title('Mean Yield Difference (Power - Linear)');
legend('Location', 'best');
grid on;

sgtitle('Mean Yield Evolution and Difference', 'FontSize', 14);

%% Figure 5: 3D surface plot
figure('Name', '3D Discrimination Surface', 'Position', [300 300 800 600]);

[P_grid, T_grid] = meshgrid(P_levels, ExtractionTimes);
surf(P_grid, T_grid, ks_stat_matrix, 'FaceAlpha', 0.8);
hold on;
scatter3(P_grid(:), T_grid(:), ks_stat_matrix(:), 50, 'k', 'filled');

xlabel('Pressure [bar]');
ylabel('Extraction Time [min]');
zlabel('KS Statistic');
title('Model Discrimination Surface');
colorbar;
colormap('jet');
view(45, 30);
grid on;

%% ========================================================================
%  FIND OPTIMAL CONDITIONS FOR DISCRIMINATION
%  ========================================================================
fprintf('\n=== OPTIMAL CONDITIONS FOR MODEL DISCRIMINATION ===\n\n');

% Find maximum KS statistic
[max_ks, idx_max_ks] = max(ks_stat_matrix(:));
[i_time_best_ks, i_P_best_ks] = ind2sub(size(ks_stat_matrix), idx_max_ks);
fprintf('Maximum KS statistic: %.4f\n', max_ks);
fprintf('  Occurs at: t = %d min, P = %.0f bar\n\n', ExtractionTimes(i_time_best_ks), P_levels(i_P_best_ks));

% Find maximum Wasserstein distance
[max_w1, idx_max_w1] = max(wasserstein_matrix(:));
[i_time_best_w1, i_P_best_w1] = ind2sub(size(wasserstein_matrix), idx_max_w1);
fprintf('Maximum Wasserstein distance: %.4f g\n', max_w1);
fprintf('  Occurs at: t = %d min, P = %.0f bar\n\n', ExtractionTimes(i_time_best_w1), P_levels(i_P_best_w1));

% Find maximum |Cohen's d|
[max_d, idx_max_d] = max(abs(cohens_d_matrix(:)));
[i_time_best_d, i_P_best_d] = ind2sub(size(cohens_d_matrix), idx_max_d);
fprintf('Maximum |Cohen''s d|: %.4f\n', max_d);
fprintf('  Occurs at: t = %d min, P = %.0f bar\n\n', ExtractionTimes(i_time_best_d), P_levels(i_P_best_d));

% Effect size interpretation at each time
fprintf('Effect size evolution (average across pressures):\n');
for i_time = 1:n_T_extract
    mean_d = mean(abs(cohens_d_matrix(i_time, :)));
    if mean_d < 0.2
        effect_str = 'negligible';
    elseif mean_d < 0.5
        effect_str = 'small';
    elseif mean_d < 0.8
        effect_str = 'medium';
    else
        effect_str = 'LARGE';
    end
    fprintf('  t = %3d min: mean |d| = %.3f (%s)\n', ExtractionTimes(i_time), mean_d, effect_str);
end

%% Save results
temporal_results = struct();
temporal_results.ExtractionTimes = ExtractionTimes;
temporal_results.P_levels = P_levels;
temporal_results.T_samples = T_samples;
temporal_results.F_samples = F_samples;
temporal_results.N_samples = N_samples;
temporal_results.all_results = all_results;
temporal_results.ks_stat_matrix = ks_stat_matrix;
temporal_results.js_div_matrix = js_div_matrix;
temporal_results.wasserstein_matrix = wasserstein_matrix;
temporal_results.cohens_d_matrix = cohens_d_matrix;
temporal_results.mean_diff_matrix = mean_diff_matrix;
temporal_results.ttest_t_matrix = ttest_t_matrix;

save('distribution_comparison_temporal_results.mat', 'temporal_results');
fprintf('\nResults saved to distribution_comparison_temporal_results.mat\n');

fprintf('\n=== Temporal Analysis Complete ===\n');
