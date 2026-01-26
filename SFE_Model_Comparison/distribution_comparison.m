%% Distribution-Based Model Comparison for SFE Models
% This script compares Power and Linear model output distributions
% by sampling T and F from uniform distributions at fixed pressure levels.
%
% Method: Latin Hypercube Sampling (LHS) for efficient coverage of (T, F) space
% Output: Distribution comparison metrics at each pressure level

%% Initialization
startup;

addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
import casadi.*

rng(42);  % Set seed for reproducibility

%% Parallel Pool Setup
% Start parallel pool if not already running
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local');  % Use all available cores
end
n_workers = pool.NumWorkers;
fprintf('Parallel pool started with %d workers\n', n_workers);

%% Configuration
N_samples = 1500;  % Number of LHS samples (increase for final analysis)
P_levels = [100, 125, 150, 175, 200];  % Fixed pressure levels [bar]

% Input ranges for uniform distributions
T_min = 303;  T_max = 313;  % Temperature [K] (30-40 C)
F_min = 3.3e-5; F_max = 6.7e-5; % Flow rate [m3/s]

fprintf('=== Distribution-Based Model Comparison ===\n\n');
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
ExtractionTime  = 60;
timeStep        = 5;  % [minutes]

simulationTime   = PreparationTime + ExtractionTime;
timeStep_in_sec  = timeStep * 60;
Time_in_sec      = (timeStep:timeStep:simulationTime) * 60;
Time             = [0 Time_in_sec/60];
N_Time           = length(Time_in_sec);

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

%% Store constants needed for parallel workers
% Pack all constants into a struct for easy transfer to workers
model_config = struct();
model_config.bed_mask = bed_mask;
model_config.timeStep_in_sec = timeStep_in_sec;
model_config.epsi_mask = epsi_mask;
model_config.one_minus_epsi_mask = one_minus_epsi_mask;
model_config.Nx = Nx;
model_config.Nu = Nu;
model_config.N_Time = N_Time;
model_config.nstages = nstages;
model_config.C0fluid = C0fluid;
model_config.C0solid = C0solid;
model_config.Parameters = Parameters;

%% Generate LHS samples for (T, F)
fprintf('Generating %d LHS samples...\n', N_samples);
lhs_samples = lhsdesign(N_samples, 2);

T_samples = T_min + (T_max - T_min) * lhs_samples(:, 1);
F_samples = F_min + (F_max - F_min) * lhs_samples(:, 2);

%% Preallocate results storage
n_P = length(P_levels);

% Final yield distributions for each pressure level
Y_final_power  = zeros(N_samples, n_P);
Y_final_linear = zeros(N_samples, n_P);

% Store sampled inputs
results.T_samples = T_samples;
results.F_samples = F_samples;
results.P_levels  = P_levels;

%% ========================================================================
%  MAIN SIMULATION LOOP (PARALLELIZED WITH BATCH PROCESSING)
%  ========================================================================
fprintf('\nRunning simulations in parallel...\n');
tic;

% Extract constants for parfor (avoid broadcasting entire struct)
bed_mask_local = model_config.bed_mask;
timeStep_in_sec_local = model_config.timeStep_in_sec;
epsi_mask_local = model_config.epsi_mask;
one_minus_epsi_mask_local = model_config.one_minus_epsi_mask;
Nx_local = model_config.Nx;
Nu_local = model_config.Nu;
N_Time_local = model_config.N_Time;
nstages_local = model_config.nstages;
C0fluid_local = model_config.C0fluid;
C0solid_local = model_config.C0solid;
Parameters_local = model_config.Parameters;
Parameters_mat = cell2mat(Parameters_local);

% Determine batch size per worker (build integrator once per batch)
batch_size = ceil(N_samples / n_workers);
n_batches = ceil(N_samples / batch_size);

fprintf('Processing %d samples in %d batches (batch size: %d)\n', N_samples, n_batches, batch_size);

for i_P = 1:n_P
    P0 = P_levels(i_P);
    fprintf('\n--- Pressure: %.0f bar ---\n', P0);

    % Preallocate temporary arrays for this pressure level
    Y_linear_temp = zeros(N_samples, 1);
    Y_power_temp  = zeros(N_samples, 1);

    % Create batch indices
    batch_starts = 1:batch_size:N_samples;
    batch_ends = min(batch_starts + batch_size - 1, N_samples);
    n_batches_actual = length(batch_starts);

    % Cell arrays to collect batch results
    Y_linear_batches = cell(n_batches_actual, 1);
    Y_power_batches  = cell(n_batches_actual, 1);

    % Parallel loop over batches (each worker builds integrator once)
    parfor i_batch = 1:n_batches_actual
        % Import CasADi inside parfor (each worker needs it)
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

        % Loop over samples in this batch (sequential within worker)
        for j = 1:n_batch
            i_s = batch_indices(j);
            T0 = T_samples(i_s);
            F0 = F_samples(i_s);

            % Compute fluid properties at operating conditions
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

            % Build input matrix
            U_all = [uu'; repmat(Parameters_mat, 1, N_Time_local)];

            % Run simulations
            try
                X_all_linear = F_accum_linear(x0, U_all);
                X_all_power  = F_accum_power(x0, U_all);

                % Extract final yield
                Y_linear_batch(j) = full(X_all_linear(end, end));
                Y_power_batch(j)  = full(X_all_power(end, end));

            catch ME
                Y_linear_batch(j) = NaN;
                Y_power_batch(j)  = NaN;
            end
        end

        % Store batch results
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

    % Store results for this pressure level
    Y_final_linear(:, i_P) = Y_linear_temp;
    Y_final_power(:, i_P)  = Y_power_temp;

    % Progress update
    n_valid = sum(~isnan(Y_linear_temp));
    fprintf('  Completed: %d/%d valid samples\n', n_valid, N_samples);
end

elapsed_time = toc;
fprintf('\nSimulations completed in %.1f seconds (%.1f samples/sec).\n', ...
    elapsed_time, N_samples * n_P / elapsed_time);

%% ========================================================================
%  COMPUTE DISTRIBUTION COMPARISON METRICS
%  ========================================================================
fprintf('\n=== Computing Distribution Comparison Metrics ===\n\n');

% Preallocate metrics structure
metrics = struct();
metrics.P_levels = P_levels;

for i_P = 1:n_P
    P0 = P_levels(i_P);

    Y_power  = Y_final_power(:, i_P);
    Y_linear = Y_final_linear(:, i_P);

    % Remove NaN values
    valid_idx = ~isnan(Y_power) & ~isnan(Y_linear);
    Y_power  = Y_power(valid_idx);
    Y_linear = Y_linear(valid_idx);
    n_valid  = sum(valid_idx);

    fprintf('P = %.0f bar (n = %d valid samples):\n', P0, n_valid);

    %% 1. Basic Statistics
    metrics.mean_power(i_P)   = mean(Y_power);
    metrics.mean_linear(i_P)  = mean(Y_linear);
    metrics.std_power(i_P)    = std(Y_power);
    metrics.std_linear(i_P)   = std(Y_linear);
    metrics.median_power(i_P) = median(Y_power);
    metrics.median_linear(i_P)= median(Y_linear);

    fprintf('  Mean:   Power = %.4f, Linear = %.4f\n', metrics.mean_power(i_P), metrics.mean_linear(i_P));
    fprintf('  Std:    Power = %.4f, Linear = %.4f\n', metrics.std_power(i_P), metrics.std_linear(i_P));

    %% 2. Two-sample t-test
    [~, p_ttest, ~, stats_ttest] = ttest2(Y_power, Y_linear);
    metrics.ttest_t(i_P)      = stats_ttest.tstat;
    metrics.ttest_p(i_P)      = p_ttest;
    metrics.ttest_df(i_P)     = stats_ttest.df;

    fprintf('  T-test: t = %.3f, p = %.4e\n', metrics.ttest_t(i_P), metrics.ttest_p(i_P));

    %% 3. Kolmogorov-Smirnov Test
    [~, p_ks, ks_stat] = kstest2(Y_power, Y_linear);
    metrics.ks_stat(i_P) = ks_stat;
    metrics.ks_p(i_P)    = p_ks;

    fprintf('  KS test: D = %.4f, p = %.4e\n', metrics.ks_stat(i_P), metrics.ks_p(i_P));

    %% 4. Wasserstein Distance (Earth Mover's Distance)
    % Sort both samples and compute L1 distance between CDFs
    Y_power_sorted  = sort(Y_power);
    Y_linear_sorted = sort(Y_linear);

    % Compute Wasserstein-1 distance using sorted samples
    % For equal-sized samples: W1 = mean(|sorted_X - sorted_Y|)
    if length(Y_power_sorted) == length(Y_linear_sorted)
        metrics.wasserstein(i_P) = mean(abs(Y_power_sorted - Y_linear_sorted));
    else
        % Use interpolation for unequal samples
        n_interp = max(length(Y_power), length(Y_linear));
        cdf_power  = interp1(linspace(0,1,length(Y_power_sorted)), Y_power_sorted, linspace(0,1,n_interp));
        cdf_linear = interp1(linspace(0,1,length(Y_linear_sorted)), Y_linear_sorted, linspace(0,1,n_interp));
        metrics.wasserstein(i_P) = mean(abs(cdf_power - cdf_linear));
    end

    fprintf('  Wasserstein: W1 = %.4f\n', metrics.wasserstein(i_P));

    %% 5. Bhattacharyya Coefficient and Distance
    % Use kernel density estimation for continuous distributions
    n_bins = 50;
    edges = linspace(min([Y_power; Y_linear]), max([Y_power; Y_linear]), n_bins + 1);

    hist_power  = histcounts(Y_power, edges, 'Normalization', 'probability');
    hist_linear = histcounts(Y_linear, edges, 'Normalization', 'probability');

    % Bhattacharyya coefficient: BC = sum(sqrt(p * q))
    BC = sum(sqrt(hist_power .* hist_linear));
    metrics.bhattacharyya_coef(i_P) = BC;

    % Bhattacharyya distance: DB = -ln(BC)
    if BC > 0
        metrics.bhattacharyya_dist(i_P) = -log(BC);
    else
        metrics.bhattacharyya_dist(i_P) = Inf;
    end

    fprintf('  Bhattacharyya: BC = %.4f, DB = %.4f\n', metrics.bhattacharyya_coef(i_P), metrics.bhattacharyya_dist(i_P));

    %% 6. Jensen-Shannon Divergence
    % JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)
    % Avoid log(0) by adding small epsilon
    eps_val = 1e-10;
    hist_power  = hist_power + eps_val;
    hist_linear = hist_linear + eps_val;

    % Normalize again after adding epsilon
    hist_power  = hist_power / sum(hist_power);
    hist_linear = hist_linear / sum(hist_linear);

    M = 0.5 * (hist_power + hist_linear);

    KL_PM = sum(hist_power .* log(hist_power ./ M));
    KL_QM = sum(hist_linear .* log(hist_linear ./ M));

    metrics.js_divergence(i_P) = 0.5 * KL_PM + 0.5 * KL_QM;

    fprintf('  Jensen-Shannon: JS = %.4f\n', metrics.js_divergence(i_P));

    %% 7. Cohen's d (Effect Size)
    pooled_std = sqrt(((n_valid-1)*std(Y_power)^2 + (n_valid-1)*std(Y_linear)^2) / (2*n_valid - 2));
    metrics.cohens_d(i_P) = (mean(Y_power) - mean(Y_linear)) / pooled_std;

    fprintf('  Cohen''s d: %.4f\n', metrics.cohens_d(i_P));

    fprintf('\n');
end

%% Store results
results.Y_final_power  = Y_final_power;
results.Y_final_linear = Y_final_linear;
results.metrics = metrics;
results.Time = Time;
results.N_samples = N_samples;

%% ========================================================================
%  VISUALIZATION
%  ========================================================================
fprintf('Generating visualizations...\n');

%% Figure 1: Distribution histograms at each pressure
figure('Name', 'Output Distributions by Pressure', 'Position', [100 100 1400 800]);

for i_P = 1:n_P
    subplot(2, 3, i_P);

    Y_power  = Y_final_power(:, i_P);
    Y_linear = Y_final_linear(:, i_P);

    % Remove NaN
    Y_power  = Y_power(~isnan(Y_power));
    Y_linear = Y_linear(~isnan(Y_linear));

    % Create overlapping histograms
    edges = linspace(min([Y_power; Y_linear])*0.95, max([Y_power; Y_linear])*1.05, 25);

    histogram(Y_power, edges, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'EdgeColor', 'b', 'DisplayName', 'Power');
    hold on;
    histogram(Y_linear, edges, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'r', 'DisplayName', 'Linear');

    xlabel('Final Yield [g]');
    ylabel('Frequency');
    title(sprintf('P = %.0f bar', P_levels(i_P)));
    legend('Location', 'best');
    grid on;

    % Add metrics annotation
    text(0.02, 0.98, sprintf('KS: %.3f\nJS: %.3f\nW1: %.3f', ...
        metrics.ks_stat(i_P), metrics.js_divergence(i_P), metrics.wasserstein(i_P)), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 8);
end

subplot(2, 3, 6);
% Summary bar chart
bar_data = [metrics.ks_stat; metrics.js_divergence; metrics.wasserstein]';
bar(P_levels, bar_data);
xlabel('Pressure [bar]');
ylabel('Metric Value');
title('Comparison Metrics vs Pressure');
legend('KS Statistic', 'JS Divergence', 'Wasserstein', 'Location', 'best');
grid on;

sgtitle('Model Output Distributions at Different Pressures', 'FontSize', 14);

%% Figure 2: Cumulative Distribution Functions
figure('Name', 'Empirical CDFs', 'Position', [150 150 1200 800]);

for i_P = 1:n_P
    subplot(2, 3, i_P);

    Y_power  = Y_final_power(:, i_P);
    Y_linear = Y_final_linear(:, i_P);

    % Remove NaN
    Y_power  = Y_power(~isnan(Y_power));
    Y_linear = Y_linear(~isnan(Y_linear));

    % Plot ECDFs
    [f_power, x_power]   = ecdf(Y_power);
    [f_linear, x_linear] = ecdf(Y_linear);

    plot(x_power, f_power, 'b-', 'LineWidth', 2, 'DisplayName', 'Power');
    hold on;
    plot(x_linear, f_linear, 'r--', 'LineWidth', 2, 'DisplayName', 'Linear');

    xlabel('Final Yield [g]');
    ylabel('Cumulative Probability');
    title(sprintf('P = %.0f bar (KS = %.3f)', P_levels(i_P), metrics.ks_stat(i_P)));
    legend('Location', 'southeast');
    grid on;
end

subplot(2, 3, 6);
% Plot KS statistic vs pressure
plot(P_levels, metrics.ks_stat, 'ko-', 'LineWidth', 2, 'MarkerFaceColor', 'k', 'MarkerSize', 8);
xlabel('Pressure [bar]');
ylabel('KS Statistic');
title('Model Separation vs Pressure');
grid on;

sgtitle('Empirical Cumulative Distribution Functions', 'FontSize', 14);

%% Figure 3: Scatter plot of Power vs Linear predictions
figure('Name', 'Power vs Linear Predictions', 'Position', [200 200 1000 800]);

colors = parula(n_P);
hold on;

for i_P = 1:n_P
    Y_power  = Y_final_power(:, i_P);
    Y_linear = Y_final_linear(:, i_P);

    % Remove NaN
    valid_idx = ~isnan(Y_power) & ~isnan(Y_linear);
    Y_power  = Y_power(valid_idx);
    Y_linear = Y_linear(valid_idx);

    scatter(Y_linear, Y_power, 30, colors(i_P,:), 'filled', 'DisplayName', sprintf('P = %.0f bar', P_levels(i_P)));
end

% Add diagonal line (perfect agreement)
all_Y = [Y_final_power(:); Y_final_linear(:)];
all_Y = all_Y(~isnan(all_Y));
lims = [min(all_Y)*0.95, max(all_Y)*1.05];
plot(lims, lims, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Perfect Agreement');

xlabel('Linear Model Yield [g]');
ylabel('Power Model Yield [g]');
title('Power vs Linear Model Predictions');
legend('Location', 'southeast');
grid on;
axis equal;
xlim(lims);
ylim(lims);

%% Figure 4: Metrics Summary
figure('Name', 'Distribution Comparison Metrics Summary', 'Position', [250 250 1200 600]);

subplot(2, 3, 1);
plot(P_levels, metrics.ttest_t, 'bo-', 'LineWidth', 2, 'MarkerFaceColor', 'b');
xlabel('Pressure [bar]');
ylabel('t-statistic');
title('T-test Statistic');
grid on;

subplot(2, 3, 2);
semilogy(P_levels, metrics.ttest_p, 'ro-', 'LineWidth', 2, 'MarkerFaceColor', 'r');
hold on;
yline(0.05, 'k--', '\alpha = 0.05');
xlabel('Pressure [bar]');
ylabel('p-value');
title('T-test p-value');
grid on;

subplot(2, 3, 3);
plot(P_levels, metrics.ks_stat, 'go-', 'LineWidth', 2, 'MarkerFaceColor', 'g');
xlabel('Pressure [bar]');
ylabel('KS Statistic');
title('Kolmogorov-Smirnov Statistic');
grid on;

subplot(2, 3, 4);
plot(P_levels, metrics.wasserstein, 'mo-', 'LineWidth', 2, 'MarkerFaceColor', 'm');
xlabel('Pressure [bar]');
ylabel('W1 Distance');
title('Wasserstein Distance');
grid on;

subplot(2, 3, 5);
plot(P_levels, metrics.bhattacharyya_coef, 'co-', 'LineWidth', 2, 'MarkerFaceColor', 'c');
xlabel('Pressure [bar]');
ylabel('BC');
title('Bhattacharyya Coefficient');
ylim([0 1]);
grid on;

subplot(2, 3, 6);
plot(P_levels, metrics.cohens_d, 'ko-', 'LineWidth', 2, 'MarkerFaceColor', 'k');
xlabel('Pressure [bar]');
ylabel('Cohen''s d');
title('Effect Size (Cohen''s d)');
grid on;

sgtitle('Distribution Comparison Metrics vs Pressure', 'FontSize', 14);

%% Figure 5: Input sampling visualization
figure('Name', 'LHS Sampling Pattern', 'Position', [300 300 600 500]);

scatter(T_samples - 273, F_samples * 1e5 * 60, 50, 'b', 'filled');
xlabel('Temperature [C]');
ylabel('Flow Rate [g/min]');
title(sprintf('Latin Hypercube Samples (N = %d)', N_samples));
grid on;
xlim([T_min-273-1, T_max-273+1]);
ylim([F_min*1e5*60*0.95, F_max*1e5*60*1.05]);

%% ========================================================================
%  PRINT SUMMARY TABLE
%  ========================================================================
fprintf('\n');
fprintf('=============================================================================\n');
fprintf('                    DISTRIBUTION COMPARISON SUMMARY                          \n');
fprintf('=============================================================================\n');
fprintf('%-8s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
    'P [bar]', 'KS stat', 'KS p-val', 'JS div', 'W1 dist', 'BC', 'Cohen d');
fprintf('-----------------------------------------------------------------------------\n');

for i_P = 1:n_P
    fprintf('%-8.0f %-10.4f %-10.4e %-10.4f %-10.4f %-10.4f %-10.4f\n', ...
        P_levels(i_P), ...
        metrics.ks_stat(i_P), ...
        metrics.ks_p(i_P), ...
        metrics.js_divergence(i_P), ...
        metrics.wasserstein(i_P), ...
        metrics.bhattacharyya_coef(i_P), ...
        metrics.cohens_d(i_P));
end

fprintf('=============================================================================\n');

%% Interpretation
fprintf('\n=== INTERPRETATION ===\n\n');

% Find pressure with maximum discrimination
[~, idx_best_ks] = max(metrics.ks_stat);
[~, idx_best_js] = max(metrics.js_divergence);
[~, idx_best_w1] = max(metrics.wasserstein);

fprintf('Best pressure for discrimination:\n');
fprintf('  KS statistic:    P = %.0f bar (KS = %.4f)\n', P_levels(idx_best_ks), metrics.ks_stat(idx_best_ks));
fprintf('  JS divergence:   P = %.0f bar (JS = %.4f)\n', P_levels(idx_best_js), metrics.js_divergence(idx_best_js));
fprintf('  Wasserstein:     P = %.0f bar (W1 = %.4f)\n', P_levels(idx_best_w1), metrics.wasserstein(idx_best_w1));

fprintf('\nStatistical significance (alpha = 0.05):\n');
for i_P = 1:n_P
    if metrics.ks_p(i_P) < 0.05
        sig_str = 'SIGNIFICANT';
    else
        sig_str = 'not significant';
    end
    fprintf('  P = %.0f bar: KS p = %.4e (%s)\n', P_levels(i_P), metrics.ks_p(i_P), sig_str);
end

fprintf('\nEffect size interpretation (Cohen''s d):\n');
fprintf('  |d| < 0.2: negligible, 0.2-0.5: small, 0.5-0.8: medium, > 0.8: large\n');
for i_P = 1:n_P
    d = abs(metrics.cohens_d(i_P));
    if d < 0.2
        effect_str = 'negligible';
    elseif d < 0.5
        effect_str = 'small';
    elseif d < 0.8
        effect_str = 'medium';
    else
        effect_str = 'LARGE';
    end
    fprintf('  P = %.0f bar: d = %.3f (%s)\n', P_levels(i_P), metrics.cohens_d(i_P), effect_str);
end

%% Save results
save('distribution_comparison_results.mat', 'results', 'metrics');
fprintf('\nResults saved to distribution_comparison_results.mat\n');

fprintf('\n=== Analysis Complete ===\n');
