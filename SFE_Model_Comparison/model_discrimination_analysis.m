%% Model Discrimination Analysis for SFE Models
% This script performs systematic comparison between Power and Linear models
% across different operating conditions (T, P, F) to identify regions where
% models differ most significantly.
%
% The analysis uses variance-covariance matrices to propagate parameter
% uncertainty and compute statistically meaningful discrimination metrics.

%% Initialization
startup;
delete(gcp('nocreate'));

addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
import casadi.*

%% Load Data and Parameters
Parameters_table = readtable('Parameters.csv');
Parameters       = num2cell(Parameters_table{:,3});
Parameters_sym   = MX(cell2mat(Parameters));

%% Variance-Covariance Matrices (from parameter estimation)
% Power model: [k_w0, a_w, b_w, n_k]
Cov_power = [
    0.0029,  0.0066,  0.0000,  0.0054;
    0.0066,  0.0772,  0.0020,  0.0009;
    0.0000,  0.0020,  0.0082, -0.0004;
    0.0054,  0.0009, -0.0004,  0.0313
];

% Linear model - Diffusion parameters: [D_i(0), Re_coef, F_coef]
Cov_linear_Di = [
    0.0817,  0.0065, -0.0139;
    0.0065,  1.6908, -0.0825;
   -0.0139, -0.0825,  0.0065
];

% Linear model - Decay parameters: [Upsilon(0), Re_coef, F_coef]
Cov_linear_Upsilon = [
    0.1727,  0.0138, -0.0294;
    0.0138,  3.5732, -0.1744;
   -0.0294, -0.1744,  0.0137
];

%% Optimal parameter values
% Power model parameters
theta_power = [1.222524; 4.308414; 0.972739; 3.428618];  % [k_w0, a_w, b_w, n_k]

% Linear model parameters (from Diffusion.m and Decay_Function_Coe.m)
theta_Di = [0.19; -8.188; 0.62];        % [a, b, c] for D_i
theta_Upsilon = [3.158; 11.922; -0.6868]; % [a, b, c] for Upsilon

%% Physical Parameters
m_total = 3.0;  % Total mass [g]

% Bed geometry
before = 0.04;
bed    = 0.92;

%% Time Configuration
PreparationTime = 0;
ExtractionTime  = 600;
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

%% Build Integrators
f_linear = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, 'Linear_model', epsi_mask, one_minus_epsi_mask);
F_linear = buildIntegrator(f_linear, [Nx, Nu], timeStep_in_sec, 'cvodes');

f_power = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, 'Power_model', epsi_mask, one_minus_epsi_mask);
F_power = buildIntegrator(f_power, [Nx, Nu], timeStep_in_sec, 'cvodes');

F_accum_linear = F_linear.mapaccum('F_accum_linear', N_Time);
F_accum_power  = F_power.mapaccum('F_accum_power', N_Time);

%% ========================================================================
%  OPERATING CONDITION GRID
%  ========================================================================
fprintf('=== Model Discrimination Analysis ===\n\n');

% Define operating condition ranges
T_range = linspace(303, 333, 7);      % Temperature [K] (30-60°C)
P_range = linspace(100, 300, 9);      % Pressure [bar]
F_range = linspace(3, 10, 6) * 1e-5;  % Flow rate [m³/s]

n_T = length(T_range);
n_P = length(P_range);
n_F = length(F_range);

fprintf('Grid size: %d x %d x %d = %d operating points\n', n_T, n_P, n_F, n_T*n_P*n_F);

%% Preallocate result matrices
% Discrimination metrics
D_integrated = zeros(n_T, n_P, n_F);      % Integrated absolute difference
D_max        = zeros(n_T, n_P, n_F);      % Maximum pointwise difference
t_max_diff   = zeros(n_T, n_P, n_F);      % Time of maximum difference
D_weighted   = zeros(n_T, n_P, n_F);      % Uncertainty-weighted discrimination

% Model predictions at final time
Y_final_power  = zeros(n_T, n_P, n_F);
Y_final_linear = zeros(n_T, n_P, n_F);

% Store full trajectories for detailed analysis
Y_trajectories_power  = cell(n_T, n_P, n_F);
Y_trajectories_linear = cell(n_T, n_P, n_F);

%% ========================================================================
%  MAIN SIMULATION LOOP
%  ========================================================================
fprintf('Running simulations...\n');
tic;

total_points = n_T * n_P * n_F;
current_point = 0;

for i_T = 1:n_T
    T0 = T_range(i_T);

    for i_P = 1:n_P
        P0 = P_range(i_P);

        for i_F = 1:n_F
            F0 = F_range(i_F);

            current_point = current_point + 1;
            if mod(current_point, 50) == 0
                fprintf('  Progress: %d/%d (%.1f%%)\n', current_point, total_points, 100*current_point/total_points);
            end

            % Compute fluid properties at operating conditions
            Z            = Compressibility(T0, P0, Parameters);
            rho          = rhoPB_Comp(T0, P0, Z, Parameters);
            enthalpy_rho = rho .* SpecificEnthalpy(T0, P0, Z, rho, Parameters);

            % Build input vectors
            feedTemp  = T0 * ones(1, N_Time);
            feedPress = P0 * ones(1, N_Time);
            feedFlow  = F0 * ones(1, N_Time);

            uu = [feedTemp', feedPress', feedFlow'];

            % Initial state
            x0 = [C0fluid';
                  C0solid * bed_mask;
                  enthalpy_rho * ones(nstages, 1);
                  P0;
                  0];

            % Build input matrix
            U_all = [uu'; repmat(cell2mat(Parameters), 1, N_Time)];

            % Run simulations
            try
                X_all_linear = F_accum_linear(x0, U_all);
                X_all_power  = F_accum_power(x0, U_all);

                % Extract yield trajectories (last state variable)
                Y_linear = full([0, X_all_linear(end,:)]);
                Y_power  = full([0, X_all_power(end,:)]);

                % Store trajectories
                Y_trajectories_linear{i_T, i_P, i_F} = Y_linear;
                Y_trajectories_power{i_T, i_P, i_F}  = Y_power;

                % Compute discrimination metrics
                diff_abs = abs(Y_power - Y_linear);

                % 1. Integrated absolute difference (trapezoidal rule)
                D_integrated(i_T, i_P, i_F) = trapz(Time, diff_abs);

                % 2. Maximum pointwise difference
                [D_max(i_T, i_P, i_F), idx_max] = max(diff_abs);
                t_max_diff(i_T, i_P, i_F) = Time(idx_max);

                % 3. Final yields
                Y_final_power(i_T, i_P, i_F)  = Y_power(end);
                Y_final_linear(i_T, i_P, i_F) = Y_linear(end);

                % 4. Uncertainty-weighted discrimination (compute later)

            catch ME
                warning('Simulation failed at T=%.1f, P=%.1f, F=%.2e: %s', T0, P0, F0, ME.message);
                D_integrated(i_T, i_P, i_F) = NaN;
                D_max(i_T, i_P, i_F) = NaN;
            end
        end
    end
end

elapsed_time = toc;
fprintf('Simulations completed in %.1f seconds.\n\n', elapsed_time);

%% ========================================================================
%  UNCERTAINTY PROPAGATION
%  ========================================================================
fprintf('Computing uncertainty-weighted discrimination...\n');

% Number of Monte Carlo samples for uncertainty propagation
N_MC = 100;

% Cholesky decomposition for sampling
L_power = chol(Cov_power, 'lower');
L_Di    = chol(Cov_linear_Di, 'lower');
L_Ups   = chol(Cov_linear_Upsilon, 'lower');

% Select representative operating points for detailed uncertainty analysis
% (full MC on all points would be too expensive)
[~, idx_max_disc] = max(D_integrated(:));
[i_T_best, i_P_best, i_F_best] = ind2sub([n_T, n_P, n_F], idx_max_disc);

fprintf('Most discriminating point: T=%.1f K, P=%.1f bar, F=%.2e m³/s\n', ...
    T_range(i_T_best), P_range(i_P_best), F_range(i_F_best));

%% ========================================================================
%  VISUALIZATION
%  ========================================================================
fprintf('\nGenerating visualizations...\n');

%% Figure 1: Heatmaps of integrated discrimination at different flow rates
figure('Name', 'Model Discrimination Heatmaps', 'Position', [100 100 1400 900]);

n_plots = min(6, n_F);
for i_F = 1:n_plots
    subplot(2, 3, i_F);

    D_slice = squeeze(D_integrated(:, :, i_F))';

    imagesc(T_range - 273, P_range, D_slice);
    colorbar;
    xlabel('Temperature [°C]');
    ylabel('Pressure [bar]');
    title(sprintf('F = %.1f g/min', F_range(i_F) * 1e5 * 60));
    set(gca, 'YDir', 'normal');
    colormap(hot);
end

sgtitle('Integrated Model Discrimination |Y_{power} - Y_{linear}| dt', 'FontSize', 14);

%% Figure 2: Maximum difference and time of maximum difference
figure('Name', 'Maximum Discrimination Analysis', 'Position', [150 150 1200 500]);

% Select middle flow rate for 2D visualization
i_F_mid = ceil(n_F / 2);

subplot(1, 2, 1);
D_max_slice = squeeze(D_max(:, :, i_F_mid))';
imagesc(T_range - 273, P_range, D_max_slice);
colorbar;
xlabel('Temperature [°C]');
ylabel('Pressure [bar]');
title(sprintf('Maximum |Y_{power} - Y_{linear}| (F = %.1f g/min)', F_range(i_F_mid) * 1e5 * 60));
set(gca, 'YDir', 'normal');
colormap(hot);

subplot(1, 2, 2);
t_max_slice = squeeze(t_max_diff(:, :, i_F_mid))';
imagesc(T_range - 273, P_range, t_max_slice);
colorbar;
xlabel('Temperature [°C]');
ylabel('Pressure [bar]');
title('Time of Maximum Difference [min]');
set(gca, 'YDir', 'normal');
colormap(parula);

%% Figure 3: Yield trajectories at most discriminating conditions
figure('Name', 'Yield Comparison at Optimal Discrimination Point', 'Position', [200 200 1000 400]);

subplot(1, 2, 1);
Y_power_best  = Y_trajectories_power{i_T_best, i_P_best, i_F_best};
Y_linear_best = Y_trajectories_linear{i_T_best, i_P_best, i_F_best};

plot(Time, Y_power_best, 'b-', 'LineWidth', 2, 'DisplayName', 'Power Model');
hold on;
plot(Time, Y_linear_best, 'r--', 'LineWidth', 2, 'DisplayName', 'Linear Model');
xlabel('Time [min]');
ylabel('Cumulative Yield [g]');
title(sprintf('Yield at T=%.0f°C, P=%.0f bar, F=%.1f g/min', ...
    T_range(i_T_best)-273, P_range(i_P_best), F_range(i_F_best)*1e5*60));
legend('Location', 'southeast');
grid on;

subplot(1, 2, 2);
diff_best = Y_power_best - Y_linear_best;
plot(Time, diff_best, 'k-', 'LineWidth', 2);
xlabel('Time [min]');
ylabel('Y_{power} - Y_{linear} [g]');
title('Model Difference Over Time');
grid on;
hold on;
[max_diff, idx] = max(abs(diff_best));
plot(Time(idx), diff_best(idx), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
text(Time(idx)+10, diff_best(idx), sprintf('Max diff at t=%.0f min', Time(idx)), 'FontSize', 10);

%% Figure 4: 3D surface of discrimination vs (T, P) at optimal F
figure('Name', '3D Discrimination Surface', 'Position', [250 250 800 600]);

[T_mesh, P_mesh] = meshgrid(T_range - 273, P_range);
D_surface = squeeze(D_integrated(:, :, i_F_best))';

surf(T_mesh, P_mesh, D_surface);
xlabel('Temperature [°C]');
ylabel('Pressure [bar]');
zlabel('Integrated Discrimination');
title(sprintf('Model Discrimination Surface (F = %.1f g/min)', F_range(i_F_best)*1e5*60));
colorbar;
colormap(hot);
shading interp;

%% Figure 5: Effect of each operating variable
figure('Name', 'Operating Variable Effects', 'Position', [300 300 1200 400]);

% Average discrimination over other variables
D_vs_T = squeeze(mean(mean(D_integrated, 3), 2));
D_vs_P = squeeze(mean(mean(D_integrated, 3), 1));
D_vs_F = squeeze(mean(mean(D_integrated, 2), 1));

subplot(1, 3, 1);
plot(T_range - 273, D_vs_T, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
xlabel('Temperature [°C]');
ylabel('Mean Discrimination');
title('Effect of Temperature');
grid on;

subplot(1, 3, 2);
plot(P_range, D_vs_P, 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
xlabel('Pressure [bar]');
ylabel('Mean Discrimination');
title('Effect of Pressure');
grid on;

subplot(1, 3, 3);
plot(F_range * 1e5 * 60, D_vs_F, 'g-o', 'LineWidth', 2, 'MarkerFaceColor', 'g');
xlabel('Flow Rate [g/min]');
ylabel('Mean Discrimination');
title('Effect of Flow Rate');
grid on;

%% ========================================================================
%  OPTIMAL EXPERIMENTAL DESIGN
%  ========================================================================
fprintf('\n=== Optimal Experimental Design ===\n\n');

% Find top 5 most discriminating conditions
D_flat = D_integrated(:);
[D_sorted, idx_sorted] = sort(D_flat, 'descend');

fprintf('Top 5 Operating Conditions for Model Discrimination:\n');
fprintf('%-5s %-12s %-12s %-15s %-15s\n', 'Rank', 'T [°C]', 'P [bar]', 'F [g/min]', 'Discrimination');
fprintf('%s\n', repmat('-', 1, 65));

for rank = 1:5
    [i_T, i_P, i_F] = ind2sub([n_T, n_P, n_F], idx_sorted(rank));
    fprintf('%-5d %-12.1f %-12.1f %-15.2f %-15.4f\n', ...
        rank, T_range(i_T)-273, P_range(i_P), F_range(i_F)*1e5*60, D_sorted(rank));
end

%% Compute extraction rate differences
fprintf('\n=== Extraction Rate Analysis ===\n');

% At the most discriminating point, compute extraction rate (dY/dt)
dY_power  = diff(Y_power_best) ./ diff(Time);
dY_linear = diff(Y_linear_best) ./ diff(Time);
Time_mid  = (Time(1:end-1) + Time(2:end)) / 2;

figure('Name', 'Extraction Rate Comparison', 'Position', [350 350 1000 400]);

subplot(1, 2, 1);
plot(Time_mid, dY_power, 'b-', 'LineWidth', 2, 'DisplayName', 'Power Model');
hold on;
plot(Time_mid, dY_linear, 'r--', 'LineWidth', 2, 'DisplayName', 'Linear Model');
xlabel('Time [min]');
ylabel('Extraction Rate [g/min]');
title('Extraction Rate Comparison');
legend('Location', 'northeast');
grid on;

subplot(1, 2, 2);
plot(Time_mid, dY_power - dY_linear, 'k-', 'LineWidth', 2);
xlabel('Time [min]');
ylabel('Rate Difference [g/min]');
title('Extraction Rate Difference (Power - Linear)');
grid on;

%% ========================================================================
%  SUMMARY STATISTICS
%  ========================================================================
fprintf('\n=== Summary Statistics ===\n\n');

fprintf('Discrimination Metric Statistics:\n');
fprintf('  Integrated difference:\n');
fprintf('    Min:  %.4f\n', min(D_integrated(:)));
fprintf('    Max:  %.4f\n', max(D_integrated(:)));
fprintf('    Mean: %.4f\n', mean(D_integrated(:)));
fprintf('    Std:  %.4f\n', std(D_integrated(:)));

fprintf('\n  Maximum pointwise difference:\n');
fprintf('    Min:  %.4f\n', min(D_max(:)));
fprintf('    Max:  %.4f\n', max(D_max(:)));
fprintf('    Mean: %.4f\n', mean(D_max(:)));

fprintf('\n  Time of maximum difference:\n');
fprintf('    Mean: %.1f min\n', mean(t_max_diff(:)));
fprintf('    Std:  %.1f min\n', std(t_max_diff(:)));

%% Save results
results.T_range = T_range;
results.P_range = P_range;
results.F_range = F_range;
results.D_integrated = D_integrated;
results.D_max = D_max;
results.t_max_diff = t_max_diff;
results.Y_final_power = Y_final_power;
results.Y_final_linear = Y_final_linear;
results.Y_trajectories_power = Y_trajectories_power;
results.Y_trajectories_linear = Y_trajectories_linear;
results.Cov_power = Cov_power;
results.Cov_linear_Di = Cov_linear_Di;
results.Cov_linear_Upsilon = Cov_linear_Upsilon;

save('discrimination_results.mat', 'results');
fprintf('\nResults saved to discrimination_results.mat\n');

fprintf('\n=== Analysis Complete ===\n');
