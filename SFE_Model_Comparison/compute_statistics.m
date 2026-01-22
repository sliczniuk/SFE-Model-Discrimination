function [results_struct, T_stats, Corr_k, t_crit, std_errors, param_names, residuals_all, sigma2, cond_number] = compute_statistics(KOUT, which_k, Parameters, LabResults, bed_mask, timeStep_in_sec, epsi_mask, one_minus_epsi_mask, Nx, Nu, N_Time, N_Sample, C0fluid, C0solid, N_trial)
% COMPUTE_STATISTICS builds finite-difference sensitivities and parameter
% statistics for the current optimal parameters.

import casadi.*

fprintf('\n=== Computing Parameter Statistics ===\n');

% Build Jacobian matrix (sensitivity of residuals w.r.t. parameters)
N_observations = 0;
for jj = N_trial
    data_org = LabResults(6:19, jj+1)';
    N_observations = N_observations + (length(data_org) - 1);  % Differential data
end

Nk = numel(which_k);
fprintf('Number of observations: %d\n', N_observations);
fprintf('Number of parameters: %d\n', Nk);
fprintf('Degrees of freedom: %d\n', N_observations - Nk);

% Allocate Jacobian and residuals
J_matrix = zeros(N_observations, Nk);
residuals_all = zeros(N_observations, 1);

% Build integrator for sensitivity analysis (per-step map)
f_sens = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, [], epsi_mask, one_minus_epsi_mask);
F_sens = buildIntegrator(f_sens, [Nx, Nu], timeStep_in_sec);
F_sens_accum = F_sens.mapaccum('F_sens_accum', N_Time);

% Finite difference settings for Jacobian
rel_delta = 1e-4;   % Relative perturbation (0.01% of parameter value)
min_delta = 1e-8;   % Absolute floor to avoid zero step

fprintf('Computing Jacobian matrix (this may take a while)...\n');
tic;

obs_idx = 0;
for jj = N_trial
    fprintf('  Processing trial %d/%d...\n', find(N_trial==jj), numel(N_trial));

    % Get experimental data
    data_org = LabResults(6:19, jj+1)';
    data_diff = diff(data_org);

    % Operating conditions
    T0homog   = LabResults(2, jj+1);
    feedPress = LabResults(3, jj+1) * 10;
    Flow      = LabResults(4, jj+1) * 1e-5;

    Z            = Compressibility(T0homog, feedPress, Parameters);
    rho          = rhoPB_Comp(T0homog, feedPress, Z, Parameters);
    enthalpy_rho = rho .* SpecificEnthalpy(T0homog, feedPress, Z, rho, Parameters);

    feedTemp  = T0homog   * ones(1, N_Time);
    feedPress = feedPress * ones(1, N_Time);
    feedFlow  = Flow      * ones(1, N_Time);

    uu = [feedTemp', feedPress', feedFlow'];

    % Initial state
    x0 = [C0fluid';
          C0solid * bed_mask;
          enthalpy_rho * ones(size(bed_mask));
          feedPress(1);
          0];

    % Baseline prediction with optimized parameters
    Parameters_opt = Parameters;
    for i = 1:numel(which_k)
        Parameters_opt{which_k(i)} = KOUT(i);
    end

    U_baseline = [uu'; repmat(cell2mat(Parameters_opt), 1, N_Time)];
    X_baseline_all = F_sens_accum(x0, U_baseline);
    X_baseline = [x0, X_baseline_all];
    Yield_baseline = diff(X_baseline(Nx, N_Sample));

    % Normalize
    max_data_diff = max(data_diff);
    if max_data_diff <= 1e-9
        warning('Trial %d has near-zero derivative; using unity scale to avoid division by zero.', jj);
        max_data_diff = 1;
    end
    Yield_baseline_norm = Yield_baseline ./ max_data_diff;
    data_diff_norm = data_diff ./ max_data_diff;
    residuals_baseline = Yield_baseline_norm - data_diff_norm;

    % Store residuals
    N_obs_trial = length(residuals_baseline);
    residuals_all(obs_idx+1:obs_idx+N_obs_trial) = full(residuals_baseline');

    % Compute sensitivities for each parameter (finite differences)
    for p = 1:Nk
        delta_p = max(abs(KOUT(p)) * rel_delta, min_delta);
        k_pert = KOUT;
        k_pert(p) = k_pert(p) + delta_p;

        Parameters_pert = Parameters;
        for i = 1:numel(which_k)
            Parameters_pert{which_k(i)} = k_pert(i);
        end

        U_pert = [uu'; repmat(cell2mat(Parameters_pert), 1, N_Time)];
        X_pert_all = F_sens_accum(x0, U_pert);
        X_pert = [x0, X_pert_all];
        Yield_pert = diff(X_pert(Nx, N_Sample));

        Yield_pert_norm = Yield_pert ./ max_data_diff;
        residuals_pert = Yield_pert_norm - data_diff_norm;

        % Sensitivity: d(residuals)/d(k_p)
        sensitivity = (residuals_pert' - residuals_baseline') / delta_p;
        J_matrix(obs_idx+1:obs_idx+N_obs_trial, p) = full(sensitivity);
    end

    obs_idx = obs_idx + N_obs_trial;
end

fprintf('Jacobian computation completed in %.2f seconds.\n', toc);

%% Compute Variance-Covariance Matrix
RSS = residuals_all' * residuals_all;
sigma2 = RSS / (N_observations - Nk);
fprintf('Residual variance (sigma^2): %.6e\n', sigma2);
fprintf('Residual standard deviation: %.6e\n', sqrt(sigma2));

JtJ = J_matrix' * J_matrix;
cond_number = cond(JtJ);
fprintf('Condition number of J''*J: %.2e\n', cond_number);

if cond_number > 1e10
    warning('J''*J is poorly conditioned. Covariance estimates may be unreliable.');
end

if cond_number > 1e8
    reg_term = 1e-8 * trace(JtJ) / Nk * eye(Nk);
    Cov_k = sigma2 * inv(JtJ + reg_term);
    fprintf('Added regularization for numerical stability.\n');
else
    Cov_k = sigma2 * inv(JtJ);
end

std_errors = sqrt(diag(Cov_k));
std_matrix = diag(1 ./ sqrt(diag(Cov_k)));
Corr_k = std_matrix * Cov_k * std_matrix;

%% Display Parameter Statistics
fprintf('\n=== Parameter Estimates with Confidence Intervals ===\n');
fprintf('%-10s %12s %12s %12s %12s\n', 'Parameter', 'Estimate', 'Std Error', '95% CI Low', '95% CI High');
fprintf('%s\n', repmat('-', 1, 70));

alpha = 0.05;
df = N_observations - Nk;
t_crit = tinv(1 - alpha/2, df);

param_names = {
    'k_w_0';
    'a_w';
    'b_w';
    'n'
};

for i = 1:Nk
    ci_low = KOUT(i) - t_crit * std_errors(i);
    ci_high = KOUT(i) + t_crit * std_errors(i);
    fprintf('%-10s %12.6f %12.6f %12.6f %12.6f\n', ...
        param_names{i}, KOUT(i), std_errors(i), ci_low, ci_high);
end

fprintf('\n=== Correlation Matrix (Top 10 Highest Correlations) ===\n');
[row, col] = find(triu(abs(Corr_k), 1) > 0.7);  % Threshold: |r| > 0.7
if ~isempty(row)
    corr_values = zeros(length(row), 1);
    for i = 1:length(row)
        corr_values(i) = Corr_k(row(i), col(i));
    end
    [~, sort_idx] = sort(abs(corr_values), 'descend');

    fprintf('%-10s %-10s %12s\n', 'Param 1', 'Param 2', 'Correlation');
    fprintf('%s\n', repmat('-', 1, 40));
    for i = 1:min(10, length(sort_idx))
        idx = sort_idx(i);
        fprintf('%-10s %-10s %12.4f\n', ...
            param_names{row(idx)}, param_names{col(idx)}, corr_values(idx));
    end
else
    fprintf('No strong correlations (|r| > 0.7) found.\n');
end

%% Save Statistics to workspace variables
results_struct.KOUT = KOUT;
results_struct.std_errors = std_errors;
results_struct.Cov_k = Cov_k;
results_struct.Corr_k = Corr_k;
results_struct.param_names = param_names;
results_struct.RSS = RSS;
results_struct.sigma2 = sigma2;
results_struct.df = df;
results_struct.N_observations = N_observations;

T_stats = table(param_names, KOUT, std_errors, ...
    KOUT - t_crit*std_errors, KOUT + t_crit*std_errors, ...
    'VariableNames', {'Parameter', 'Estimate', 'StdError', 'CI_Low', 'CI_High'});

fprintf('\nStatistics ready in workspace variables (results_struct, T_stats).\n');
