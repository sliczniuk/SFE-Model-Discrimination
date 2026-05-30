%% Local Sensitivity Analysis for SFE Process Models
%
% Uses CasADi automatic differentiation to derive the sensitivity equations:
%
%   dS/dt = (df/dx) * S + (df/dtheta)
%
% where S = dx/dtheta is the sensitivity matrix. These ODEs are augmented
% with the original process model dx/dt = f(x,u) and integrated together
% using CasADi's CVODES integrator.
%
% Configuration options:
%   model_type  - 'Power_model' or 'Linear_model'
%   sens_*      - true/false flags to select individual parameters/controls
%
% The sensitivity equations are derived symbolically via CasADi's jacobian()
% and assembled by the Sensitivity() helper function.

clear; clc; close all;

%% =====================================================================
%%  1. CONFIGURATION
%% =====================================================================

model_type = 'Linear_model';        % 'Power_model' | 'Linear_model'

% --- Select which sensitivities to compute ---
% Set each to true/false individually.
%
% Controls:
sens_T = false;          % Feed temperature
sens_P = true;          % Feed pressure
sens_F = false;          % Volumetric flow rate
%
% Power model parameters (used when model_type = 'Power_model'):
sens_kw0 = false;        % k_{w,0}  — base rate constant
sens_aw  = false;        % a_w      — density exponent
sens_bw  = false;        % b_w      — flow exponent
sens_n   = false;        % n        — depletion exponent
%
% Linear model parameters (used when model_type = 'Linear_model'):
sens_aD  = false;        % a_D      — diffusion intercept
sens_bD  = false;        % b_D      — diffusion Re coefficient
sens_cD  = false;        % c_D      — diffusion F coefficient
sens_aG  = false;        % a_Υ      — decay intercept
sens_bG  = false;        % b_Υ      — decay Re coefficient
sens_cG  = false;        % c_Υ      — decay F coefficient

% --- Operating conditions ---
T_feed   = 308;                             % Feed temperature [K]
P_values = [100, 125, 150, 175, 200];       % Feed pressures to sweep [bar]
F_feed   = 5e-5;                            % Volumetric flow rate [m3/s]

% --- Simulation settings ---
timeStep  = 1;     % Time step [min]
finalTime = 10^5;    % Total duration [min]

% --- CasADi path ---
casadi_path = 'C:\Dev\casadi-3.6.3-windows64-matlab2018b';

%% =====================================================================
%%  2. SETUP (pressure-independent)
%% =====================================================================

addpath(casadi_path);
import casadi.*

% --- Load parameters ---
Parameters_table = readtable('Parameters.csv');
Parameters       = num2cell(Parameters_table{:,3});

% --- Time grid ---
Time_in_sec     = (timeStep:timeStep:finalTime) * 60;      % [s]
Time            = [0, Time_in_sec / 60];                    % [min]
N_Time          = length(Time_in_sec);
timeStep_in_sec = timeStep * 60;

% --- Extractor geometry ---
m_total = 3.0;          % Total initial solid mass [g]
before  = 0.04;         % Fraction of extractor before bed
bed     = 0.92;         % Fraction of extractor occupied by bed

nstages = Parameters{1};
r_col   = Parameters{3};   % Extractor radius [m]
epsi    = Parameters{4};   % Void fraction [-]
L       = Parameters{6};   % Extractor length [m]

nstagesbefore = 1:floor(before * nstages);
nstagesbed    = nstagesbefore(end)+1 : nstagesbefore(end) + floor(bed * nstages);
nstagesafter  = nstagesbed(end)+1 : nstages;

bed_mask = zeros(nstages, 1);
bed_mask(nstagesbed) = 1;

V_slice = (L / nstages) * pi * r_col^2;
V_bed   = V_slice * numel(nstagesbed);

C0solid = m_total * 1e-3 / (V_bed * epsi);
Parameters{2} = C0solid;

C0fluid = zeros(nstages, 1);   % No initial fluid-phase extract

epsi_mask           = epsi .* bed_mask;
one_minus_epsi_mask = 1 - epsi_mask;

Nx = 3 * nstages + 2;          % States: [C_fluid; C_solid; enthalpy_rho; P; yield]
Nu = 3 + numel(Parameters);    % Inputs: [T_u, P_u, F_u, parameters{:}]

z_grid = linspace(0, L*100, nstages);   % spatial coordinate [cm]

%% =====================================================================
%%  3. SENSITIVITY TARGET SELECTION
%% =====================================================================

% Input vector layout: u = [T_u, P_u, F_u, params{1}, params{2}, ...]
%   Controls:  T -> index 1,  P -> index 2,  F -> index 3
%   Model parameters are at offset +3 from their Parameters{} index

% --- Build selection vectors from the true/false flags above ---
theta_names  = {};
which_theta  = [];

% Controls (always available regardless of model)
all_control_names = {'T',       'P',       'F'};
all_control_idx   = [1,         2,         3];
all_control_flags = [sens_T,    sens_P,    sens_F];

sel = logical(all_control_flags);
theta_names = [theta_names, all_control_names(sel)];
which_theta = [which_theta, all_control_idx(sel)];

% Model-specific parameters
if strcmp(model_type, 'Power_model')
    all_param_names = {'k_{w,0}', 'a_w',    'b_w',    'n'};
    all_param_idx   = (44:47) + 3;          % Parameters{44:47} -> u(47:50)
    all_param_flags = [sens_kw0,  sens_aw,  sens_bw,  sens_n];
elseif strcmp(model_type, 'Linear_model')
    all_param_names = {'D_i^R (0)',    'D_i^R (1)',    'D_i^R (2)',    '\Upsilon (0)', '\Upsilon (1)', '\Upsilon (2)'};
    all_param_idx   = (48:53) + 3;          % Parameters{48:53} -> u(51:56)
    all_param_flags = [sens_aD,  sens_bD,  sens_cD,  sens_aG,      sens_bG,      sens_cG];
else
    error('Unknown model_type: %s', model_type);
end

sel = logical(all_param_flags);
theta_names = [theta_names, all_param_names(sel)];
which_theta = [which_theta, all_param_idx(sel)];

if isempty(which_theta)
    error('No sensitivity targets selected. Enable at least one sens_* flag.');
end

n_theta = numel(which_theta);

fprintf('============================================================\n');
fprintf('  Local Sensitivity Analysis — SFE Process Model\n');
fprintf('============================================================\n\n');
fprintf('Model              : %s\n', model_type);
fprintf('Sensitivity w.r.t. : %s\n', strjoin(theta_names, ', '));
fprintf('Pressures          : %s bar\n', num2str(P_values));
fprintf('Num. parameters    : %d\n', n_theta);
fprintf('State dimension    : %d\n', Nx);
fprintf('Augmented dimension: %d (= %d + %d x %d)\n\n', ...
    Nx + Nx*n_theta, Nx, Nx, n_theta);

%% =====================================================================
%%  4. AUGMENTED SYSTEM INFO (build happens per-worker inside parfor)
%% =====================================================================

Nx_aug = Nx + Nx * n_theta;

fprintf('Augmented sensitivity ODE will be built per worker...\n');
fprintf('  Augmented state dimension: %d\n', Nx_aug);
fprintf('  Integrator: CVODES, %d steps of %.0f s\n\n', N_Time, timeStep_in_sec);

%% =====================================================================
%%  5. SIMULATE — LOOP OVER PRESSURES
%% =====================================================================

N_P = numel(P_values);
mid_bed_idx = nstagesbed(round(numel(nstagesbed)/2));

% --- Parallel pool (restart if stale) ---
pool = gcp('nocreate');
if ~isempty(pool)
    delete(pool);
end
parpool('local', min(N_P, feature('numcores')));
parfevalOnAll(@() addpath(casadi_path), 0);

% Pre-compute constant parts of the input vector
params_vec = cell2mat(Parameters);

% Collect results in a cell array (parfor-compatible)
results_cell = cell(N_P, 1);

t_par = tic;
fprintf('Solving %d pressures in parallel...\n', N_P);

parfor ip = 1:N_P
    P_feed = P_values(ip);

    % --- Build integrator on this worker (avoids CasADi serialization) ---
    import casadi.*
    x_sym_w = MX.sym('x', Nx);
    u_sym_w = MX.sym('u', Nu);
    xdot_sym_w = modelSFE_thermal_lag(x_sym_w, u_sym_w, bed_mask, ...
        timeStep_in_sec, model_type, epsi_mask, one_minus_epsi_mask);
    [S_aug_w, p_aug_w, Sdot_aug_w] = Sensitivity(x_sym_w, xdot_sym_w, ...
        u_sym_w, which_theta);
    f_aug_w = casadi.Function('f_aug', {S_aug_w, p_aug_w}, {Sdot_aug_w}, ...
        {'x_aug', 'u'}, {'xdot_aug'});
    Nx_aug_w = Nx + Nx * n_theta;
    F_aug_w  = buildIntegrator(f_aug_w, [Nx_aug_w, Nu], timeStep_in_sec, 'cvodes');

    % --- Polynomial-consistent initial conditions ---
    Z_init   = Compressibility(T_feed, P_feed, Parameters);
    rho_init = rhoPB_Comp(T_feed, P_feed, Z_init, Parameters);
    H_init   = SpecificEnthalpy(T_feed, P_feed, Z_init, rho_init, Parameters);
    enthalpy_rho_exact = rho_init * H_init;

    fun = @(h) reconstruct_T_polynomial_approximation(log(-h), P_feed) - T_feed;
    enthalpy_rho_init = fzero(fun, enthalpy_rho_exact);

    x0 = [C0fluid; ...
          C0solid * bed_mask; ...
          (enthalpy_rho_init / 1e4) * ones(nstages, 1); ...
          P_feed; ...
          0];

    % --- Augmented IC ---
    s0 = zeros(Nx * n_theta, 1);
    S0 = [x0; s0];

    % --- Input vector (constant across time) ---
    u_val = [T_feed; P_feed; F_feed; params_vec];

    % --- Simulate via explicit time-stepping (avoids mapaccum memory) ---
    t_sim = tic;
    X_aug_all = zeros(Nx_aug_w, N_Time + 1);
    X_aug_all(:, 1) = S0;
    for k = 1:N_Time
        X_aug_all(:, k+1) = full(F_aug_w(X_aug_all(:, k), u_val));
    end
    sim_time = toc(t_sim);
    fprintf('  P = %d bar: %.2f s\n', P_feed, sim_time);

    % --- Extract states ---
    X_states = X_aug_all(1:Nx, :);

    r = struct();
    r.Yield    = X_states(Nx, :);
    r.C_exit   = X_states(nstages, :);
    r.Pressure = X_states(3*nstages+1, :);

    ENTHALPY_RHO_mb = X_states(2*nstages + mid_bed_idx, :) * 1e4;
    T_mb = zeros(1, size(X_states, 2));
    for k = 1:numel(T_mb)
        T_mb(k) = reconstruct_T_polynomial_approximation( ...
            log(-ENTHALPY_RHO_mb(k)), r.Pressure(k));
    end
    r.T_mid_bed = T_mb;

    S_raw   = X_aug_all(Nx+1:end, :);
    S_3D_ip = reshape(S_raw, Nx, n_theta, []);
    r.S_3D  = S_3D_ip;

    r.S_yield = zeros(n_theta, size(X_states, 2));
    r.S_Cexit = zeros(n_theta, size(X_states, 2));
    for j = 1:n_theta
        r.S_yield(j, :) = S_3D_ip(Nx, j, :);
        r.S_Cexit(j, :) = S_3D_ip(nstages, j, :);
    end

    r.theta_nom = u_val(which_theta)';

    results_cell{ip} = r;
end

fprintf('All pressures done: %.1f s total\n\n', toc(t_par));

% --- Unpack into RES struct ---
RES = struct();
N_t = numel(Time);
RES.Yield     = nan(N_P, N_t);
RES.C_exit    = nan(N_P, N_t);
RES.Pressure  = nan(N_P, N_t);
RES.T_mid_bed = nan(N_P, N_t);
RES.S_yield   = nan(N_P, n_theta, N_t);
RES.S_Cexit   = nan(N_P, n_theta, N_t);
RES.S_3D      = cell(N_P, 1);
RES.theta_nom = nan(N_P, n_theta);

for ip = 1:N_P
    r = results_cell{ip};
    RES.Yield(ip, :)      = r.Yield;
    RES.C_exit(ip, :)     = r.C_exit;
    RES.Pressure(ip, :)   = r.Pressure;
    RES.T_mid_bed(ip, :)  = r.T_mid_bed;
    RES.S_yield(ip, :, :) = r.S_yield;
    RES.S_Cexit(ip, :, :) = r.S_Cexit;
    RES.S_3D{ip}          = r.S_3D;
    RES.theta_nom(ip, :)  = r.theta_nom;
end

%% =====================================================================
%%  6. VISUALIZATION
%% =====================================================================

close all

model_label = strrep(model_type, '_', ' ');
P_labels    = arrayfun(@(p) sprintf('%d bar', p), P_values, 'UniformOutput', false);
P_colors    = lines(N_P);

n_rows = ceil(n_theta / 3);
n_cols = min(n_theta, 3);
%{\

% ---- Figure 1: State trajectories (overlaid) ----
figure('Name', 'State Trajectories', 'Position', [100 100 1200 1200]);

subplot(2,2,1); hold on;
for ip = 1:N_P
    plot(Time, RES.Yield(ip,:), 'Color', P_colors(ip,:), 'LineWidth', 1.5, ...
        'DisplayName', P_labels{ip});
end
xlabel('Time [min]'); ylabel('Yield [g]');
title('Cumulative Yield'); legend('Location','best'); grid on;

subplot(2,2,2); hold on;
for ip = 1:N_P
    plot(Time, RES.T_mid_bed(ip,:), 'Color', P_colors(ip,:), 'LineWidth', 1.5, ...
        'DisplayName', P_labels{ip});
end
xlabel('Time [min]'); ylabel('Temperature [K]');
title('Mid-bed Temperature'); legend('Location','best'); grid on;

subplot(2,2,3); hold on;
for ip = 1:N_P
    plot(Time, RES.C_exit(ip,:)*1e3, 'Color', P_colors(ip,:), 'LineWidth', 1.5, ...
        'DisplayName', P_labels{ip});
end
xlabel('Time [min]'); ylabel('Concentration [g/m^3]');
title('Exit Fluid Concentration'); legend('Location','best'); grid on;

subplot(2,2,4); hold on;
for ip = 1:N_P
    plot(Time, RES.Pressure(ip,:), 'Color', P_colors(ip,:), 'LineWidth', 1.5, ...
        'DisplayName', P_labels{ip});
end
xlabel('Time [min]'); ylabel('Pressure [bar]');
title('System Pressure'); legend('Location','best'); grid on;

sgtitle(sprintf('%s — T=%.0fK, F=%.1e m^3/s', model_label, T_feed, F_feed));
%}
% ---- Figure 2: Sensitivity of yield (overlaid per parameter) ----
figure('Name', 'Yield Sensitivity', 'Position', [150 150 400*n_cols 350*n_rows]);

for j = 1:n_theta
    subplot(n_rows, n_cols, j); hold on;
    for ip = 1:N_P
        plot(Time, squeeze(RES.S_yield(ip,j,:)), 'Color', P_colors(ip,:), ...
            'LineWidth', 1.5, 'DisplayName', P_labels{ip});
    end
    xlabel('Time [min]');
    ylabel(sprintf('$\\frac{\\partial Y}{\\partial %s}$', theta_names{j}), 'Interpreter', 'latex');
    %title(sprintf('%s', theta_names{j}));
    legend('Location','best', 'NumColumns', 1, 'Box', 'off', 'FontSize',10); grid on;
    set(gca, 'FontSize', 16, 'XScale', 'log');
end
%sgtitle(sprintf('Yield Sensitivity — %s', model_label));
exportgraphics(figure(2), ['Yield_sens.png'], "Resolution",750);

% ---- Figure 3: Normalized yield sensitivity (subplots per parameter) ----
figure('Name', 'Normalized Sensitivity', 'Position', [200 200 400*n_cols 350*n_rows]);

for j = 1:n_theta
    subplot(n_rows, n_cols, j); hold on;
    for ip = 1:N_P
        Y_safe   = max(abs(RES.Yield(ip,2:end)), 1e-12);
        S_norm_j = (RES.theta_nom(ip,j) ./ Y_safe) .* squeeze(RES.S_yield(ip,j,2:end))';
        plot(Time(1:end-1), S_norm_j, 'Color', P_colors(ip,:), 'LineWidth', 1.5, ...
            'DisplayName', P_labels{ip});
    end
    xlabel('Time [min]');
    ylabel(sprintf('$\\frac{\\partial Y}{\\partial %s}\\frac{%s}{Y}$', theta_names{j}, theta_names{j}), 'Interpreter', 'latex');
    %title(sprintf('%s', theta_names{j}));
    set(gca, 'FontSize', 16, 'XScale', 'log');
    legend('Location','southwest', 'NumColumns', 1, 'Box', 'off', 'FontSize',8); grid on;
end
exportgraphics(figure(3), ['Yield_norm_sens.png'], "Resolution",750);
%sgtitle(sprintf('Normalized Yield Sensitivity — %s', model_label));
%}
%{
% ---- Figure 4: Final sensitivity bar chart (grouped) ----
figure('Name', 'Final Sensitivity Summary', 'Position', [250 250 800 400]);

S_norm_bar = zeros(N_P, n_theta);
for ip = 1:N_P
    for j = 1:n_theta
        S_yield_end = RES.S_yield(ip, j, end);
        S_norm_bar(ip, j) = abs(RES.theta_nom(ip,j) / max(abs(RES.Yield(ip,end)), 1e-12) * S_yield_end);
    end
end
bar(S_norm_bar);
set(gca, 'XTickLabel', P_labels, 'XTickLabelRotation', 0);
xlabel('Pressure'); ylabel('|Normalized Sensitivity|');
legend(theta_names, 'Location','best');
title(sprintf('Final Yield Sensitivity (t=%d min) — %s', finalTime, model_label));
grid on;

% ---- Figure 5: Exit concentration sensitivity (overlaid) ----
figure('Name', 'Exit Concentration Sensitivity', 'Position', [300 300 400*n_cols 350*n_rows]);

for j = 1:n_theta
    subplot(n_rows, n_cols, j); hold on;
    for ip = 1:N_P
        plot(Time, squeeze(RES.S_Cexit(ip,j,:)), 'Color', P_colors(ip,:), ...
            'LineWidth', 1.5, 'DisplayName', P_labels{ip});
    end
    xlabel('Time [min]');
    ylabel(sprintf('\\partial C_{exit} / \\partial %s', theta_names{j}));
    title(sprintf('%s', theta_names{j}));
    legend('Location','best'); grid on;
end
sgtitle(sprintf('Exit Concentration Sensitivity — %s', model_label));

% ---- Figure 6: Spatial sensitivity profiles (overlaid) ----
snap_times  = [60, 180, 360, 600];
snap_labels = arrayfun(@(t) sprintf('t=%d min', t), snap_times, 'UniformOutput', false);

figure('Name', 'Spatial Sensitivity Profiles', 'Position', [350 50 400*n_cols 700]);

for j = 1:n_theta
    subplot(2, n_cols, j); hold on;
    for ip = 1:N_P
        for si = 1:numel(snap_times)
            [~, snap_idx] = min(abs(Time - snap_times(si)));
            S_Cf_spatial = RES.S_3D{ip}(1:nstages, j, snap_idx);
            plot(z_grid, S_Cf_spatial(:), 'Color', P_colors(ip,:), ...
                'LineWidth', 1.2, 'LineStyle', style_list{mod(si-1,numel(style_list))+1}, ...
                'HandleVisibility', 'off');
        end
    end
    xlabel('z [cm]'); ylabel(sprintf('\\partialC_f/\\partial%s', theta_names{j}));
    title(sprintf('Fluid — %s', theta_names{j})); grid on;

    subplot(2, n_cols, n_cols + j); hold on;
    for ip = 1:N_P
        for si = 1:numel(snap_times)
            [~, snap_idx] = min(abs(Time - snap_times(si)));
            S_Cs_spatial = RES.S_3D{ip}(nstages+1:2*nstages, j, snap_idx);
            plot(z_grid, S_Cs_spatial(:), 'Color', P_colors(ip,:), ...
                'LineWidth', 1.2, 'LineStyle', style_list{mod(si-1,numel(style_list))+1}, ...
                'HandleVisibility', 'off');
        end
    end
    xlabel('z [cm]'); ylabel(sprintf('\\partialC_s/\\partial%s', theta_names{j}));
    title(sprintf('Solid — %s', theta_names{j})); grid on;
end
sgtitle(sprintf('Spatial Sensitivity Profiles — %s', model_label));

%

% ---- Pcolor — Fluid concentration sensitivity (separate figure per pressure) ----
[T_grid, Z_grid] = meshgrid(Time, z_grid);

for j = 1:n_theta
    for ip = 1:N_P
        tiledlayout(1,1, 'Padding', 'loose'); % or 'compact'
        nexttile;

        S_Cf_zt = squeeze(RES.S_3D{ip}(1:nstages, j, :));
        pcolor(T_grid, Z_grid, S_Cf_zt);
        shading interp; colormap turbo; cb = colorbar; cb.FontSize=20;
        %ylabel(cb, sprintf('\\partial C_f / \\partial %s \n', theta_names{j}));
        set(gca, 'YDir', 'reverse', 'FontSize', 20);
        xlabel('Time [min]'); ylabel('Bed length [cm]');
        %set(gcf, 'Position', [200, 200, 1200, 800]); % [left, bottom, width, height] 
        exportgraphics(gcf, ['cf',mat2str(j),mat2str(ip),'.png'], 'Resolution', 750); close all;
    end
end

% ---- Pcolor — Solid concentration sensitivity (separate figure per pressure) ----
for j = 1:n_theta
    for ip = 1:N_P
        tiledlayout(1,1, 'Padding', 'loose'); % or 'compact'
        nexttile;

        S_Cs_zt = squeeze(RES.S_3D{ip}(nstages+1:2*nstages, j, :));
        pcolor(T_grid, Z_grid, S_Cs_zt);
        shading interp; colormap turbo; cb = colorbar; cb.FontSize=20;
        %ylabel(cb, sprintf('\\partial C_s / \\partial %s \n', theta_names{j}));
        set(gca, 'YDir', 'reverse', 'FontSize', 20);
        xlabel('Time [min]'); ylabel('Bed length [cm]');
        %set(gcf, 'Position', [200, 200, 1200, 800]); % [left, bottom, width, height] 
        exportgraphics(gcf, ['cs',mat2str(j),mat2str(ip),'.png'], 'Resolution', 750); close all;
    end
end

%}

%% =====================================================================
%%  7. SUMMARY TABLE
%% =====================================================================

fprintf('\n============================================================\n');
fprintf('  Sensitivity Summary at t = %d min\n', finalTime);
fprintf('============================================================\n');
for ip = 1:N_P
    fprintf('\n  P = %d bar  |  Yield = %.4f g\n', P_values(ip), RES.Yield(ip, end));
    fprintf('  %-20s %15s %15s\n', 'Parameter', 'dY/dtheta', 'Normalized');
    fprintf('  %s\n', repmat('-', 1, 52));
    for j = 1:n_theta
        S_end = RES.S_yield(ip, j, end);
        S_norm = RES.theta_nom(ip,j) / max(abs(RES.Yield(ip,end)), 1e-12) * S_end;
        fprintf('  %-20s %15.6e %15.4f\n', theta_names{j}, S_end, S_norm);
    end
end
