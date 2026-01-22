function [opt_conditions, D_grid, details] = optimal_experiment_design(results, criterion)
% OPTIMAL_EXPERIMENT_DESIGN
% Identifies optimal operating conditions for model discrimination based on
% various design criteria.
%
% Inputs:
%   results   - Structure from model_discrimination_analysis.m containing:
%               T_range, P_range, F_range, D_integrated, D_max, etc.
%   criterion - Design criterion: 'max_diff', 'max_integrated', 't_ratio',
%               'hunter_reiner', 'divergence'
%
% Outputs:
%   opt_conditions - [T_opt, P_opt, F_opt] optimal operating conditions
%   D_grid         - Discrimination metric grid used for optimization
%   details        - Structure with additional analysis details

if nargin < 2
    criterion = 'max_integrated';
end

fprintf('=== Optimal Experimental Design ===\n');
fprintf('Criterion: %s\n\n', criterion);

%% Extract data from results
T_range = results.T_range;
P_range = results.P_range;
F_range = results.F_range;

n_T = length(T_range);
n_P = length(P_range);
n_F = length(F_range);

%% Compute design criterion
switch lower(criterion)

    case 'max_diff'
        % Maximum pointwise difference
        D_grid = results.D_max;
        criterion_name = 'Maximum Pointwise Difference';

    case 'max_integrated'
        % Integrated absolute difference (default)
        D_grid = results.D_integrated;
        criterion_name = 'Integrated Absolute Difference';

    case 't_ratio'
        % T-statistic: |Y_power - Y_linear| / sqrt(var_power + var_linear)
        % Requires uncertainty information
        if isfield(results, 'Std_power') && isfield(results, 'Std_linear')
            diff_final = abs(results.Y_final_power - results.Y_final_linear);
            std_pooled = sqrt(results.Std_power.^2 + results.Std_linear.^2);
            D_grid = diff_final ./ max(std_pooled, 1e-8);
        else
            warning('Uncertainty data not available, using integrated difference');
            D_grid = results.D_integrated;
        end
        criterion_name = 'T-Ratio (Difference/Uncertainty)';

    case 'hunter_reiner'
        % Hunter-Reiner criterion for model discrimination
        % D = sum_i [ (y1_i - y2_i)^2 / (var1_i + var2_i) ]
        % Approximated here using final yields and integrated differences
        D_grid = results.D_integrated .* results.D_max;
        D_grid = D_grid / max(D_grid(:));  % Normalize
        criterion_name = 'Hunter-Reiner Criterion';

    case 'divergence'
        % Kullback-Leibler style divergence between model predictions
        % Approximated as relative difference weighted by magnitude
        Y_power  = results.Y_final_power;
        Y_linear = results.Y_final_linear;
        Y_mean   = (Y_power + Y_linear) / 2;
        D_grid   = abs(Y_power - Y_linear) ./ max(Y_mean, 1e-8);
        criterion_name = 'Relative Divergence';

    case 'extraction_phase'
        % Focus on transition phase (30-70% extraction)
        % Weight by time of maximum difference
        t_max = results.t_max_diff;
        % Penalize very early (<30 min) or very late (>300 min) max differences
        weight = exp(-((t_max - 150)/100).^2);  % Gaussian centered at 150 min
        D_grid = results.D_integrated .* weight;
        criterion_name = 'Extraction Phase Weighted';

    otherwise
        error('Unknown criterion: %s', criterion);
end

%% Find optimal conditions
[D_max_val, idx_max] = max(D_grid(:));
[i_T_opt, i_P_opt, i_F_opt] = ind2sub([n_T, n_P, n_F], idx_max);

T_opt = T_range(i_T_opt);
P_opt = P_range(i_P_opt);
F_opt = F_range(i_F_opt);

opt_conditions = [T_opt, P_opt, F_opt];

fprintf('Optimal Conditions (%s):\n', criterion_name);
fprintf('  Temperature: %.1f K (%.1f °C)\n', T_opt, T_opt - 273);
fprintf('  Pressure:    %.1f bar\n', P_opt);
fprintf('  Flow rate:   %.2e m³/s (%.1f g/min)\n', F_opt, F_opt * 1e5 * 60);
fprintf('  Criterion value: %.4f\n\n', D_max_val);

%% Find top 10 conditions
D_flat = D_grid(:);
[D_sorted, idx_sorted] = sort(D_flat, 'descend');

fprintf('Top 10 Operating Conditions:\n');
fprintf('%-5s %-10s %-10s %-12s %-12s\n', 'Rank', 'T [°C]', 'P [bar]', 'F [g/min]', criterion_name(1:min(12,end)));
fprintf('%s\n', repmat('-', 1, 55));

top_conditions = zeros(10, 4);
for rank = 1:min(10, length(idx_sorted))
    [i_T, i_P, i_F] = ind2sub([n_T, n_P, n_F], idx_sorted(rank));
    top_conditions(rank, :) = [T_range(i_T)-273, P_range(i_P), F_range(i_F)*1e5*60, D_sorted(rank)];
    fprintf('%-5d %-10.1f %-10.1f %-12.2f %-12.4f\n', ...
        rank, T_range(i_T)-273, P_range(i_P), F_range(i_F)*1e5*60, D_sorted(rank));
end

%% Sensitivity analysis: how much does criterion change near optimum?
fprintf('\n=== Sensitivity Analysis Near Optimum ===\n');

% Partial derivatives (finite differences on grid)
if i_T_opt > 1 && i_T_opt < n_T
    dD_dT = (D_grid(i_T_opt+1, i_P_opt, i_F_opt) - D_grid(i_T_opt-1, i_P_opt, i_F_opt)) / ...
            (T_range(i_T_opt+1) - T_range(i_T_opt-1));
else
    dD_dT = NaN;
end

if i_P_opt > 1 && i_P_opt < n_P
    dD_dP = (D_grid(i_T_opt, i_P_opt+1, i_F_opt) - D_grid(i_T_opt, i_P_opt-1, i_F_opt)) / ...
            (P_range(i_P_opt+1) - P_range(i_P_opt-1));
else
    dD_dP = NaN;
end

if i_F_opt > 1 && i_F_opt < n_F
    dD_dF = (D_grid(i_T_opt, i_P_opt, i_F_opt+1) - D_grid(i_T_opt, i_P_opt, i_F_opt-1)) / ...
            (F_range(i_F_opt+1) - F_range(i_F_opt-1));
else
    dD_dF = NaN;
end

fprintf('Local sensitivities (d(Criterion)/d(Variable)):\n');
fprintf('  dD/dT: %.4f per K\n', dD_dT);
fprintf('  dD/dP: %.4f per bar\n', dD_dP);
fprintf('  dD/dF: %.4f per m³/s\n', dD_dF);

% Normalized sensitivities
D_val = D_grid(i_T_opt, i_P_opt, i_F_opt);
fprintf('\nElasticity (normalized sensitivity):\n');
if ~isnan(dD_dT)
    fprintf('  (dD/dT)*(T/D): %.2f\n', dD_dT * T_opt / D_val);
end
if ~isnan(dD_dP)
    fprintf('  (dD/dP)*(P/D): %.2f\n', dD_dP * P_opt / D_val);
end
if ~isnan(dD_dF)
    fprintf('  (dD/dF)*(F/D): %.2f\n', dD_dF * F_opt / D_val);
end

%% Robustness: how flat is the criterion near optimum?
fprintf('\n=== Robustness of Optimal Conditions ===\n');

% Find conditions within 95% of maximum
threshold_95 = 0.95 * D_max_val;
idx_robust = D_grid >= threshold_95;
n_robust = sum(idx_robust(:));
pct_robust = 100 * n_robust / numel(D_grid);

fprintf('Conditions within 95%% of optimum: %d (%.1f%% of grid)\n', n_robust, pct_robust);

if pct_robust > 20
    fprintf('  -> Optimum is FLAT (many near-optimal conditions)\n');
    fprintf('     Practical implication: Operating conditions are flexible\n');
elseif pct_robust > 5
    fprintf('  -> Optimum is MODERATE (some flexibility)\n');
else
    fprintf('  -> Optimum is SHARP (precise conditions required)\n');
    fprintf('     Practical implication: Operating conditions are critical\n');
end

% Range of robust conditions
[i_T_rob, i_P_rob, i_F_rob] = ind2sub([n_T, n_P, n_F], find(idx_robust));
if ~isempty(i_T_rob)
    fprintf('\nRobust operating ranges (95%% of optimal):\n');
    fprintf('  T: %.1f - %.1f °C\n', min(T_range(i_T_rob))-273, max(T_range(i_T_rob))-273);
    fprintf('  P: %.1f - %.1f bar\n', min(P_range(i_P_rob)), max(P_range(i_P_rob)));
    fprintf('  F: %.1f - %.1f g/min\n', min(F_range(i_F_rob))*1e5*60, max(F_range(i_F_rob))*1e5*60);
end

%% Store details
details.criterion_name = criterion_name;
details.D_max_val = D_max_val;
details.top_conditions = top_conditions;
details.idx_opt = [i_T_opt, i_P_opt, i_F_opt];
details.sensitivity.dD_dT = dD_dT;
details.sensitivity.dD_dP = dD_dP;
details.sensitivity.dD_dF = dD_dF;
details.robustness.n_robust = n_robust;
details.robustness.pct_robust = pct_robust;
details.robustness.idx_robust = idx_robust;

%% Visualization
figure('Name', sprintf('Optimal Design - %s', criterion_name), 'Position', [100 100 1400 500]);

% 2D slice at optimal F
subplot(1, 3, 1);
D_slice_F = squeeze(D_grid(:, :, i_F_opt))';
imagesc(T_range - 273, P_range, D_slice_F);
hold on;
plot(T_opt - 273, P_opt, 'wo', 'MarkerSize', 15, 'LineWidth', 3);
plot(T_opt - 273, P_opt, 'kx', 'MarkerSize', 12, 'LineWidth', 2);
colorbar;
xlabel('Temperature [°C]');
ylabel('Pressure [bar]');
title(sprintf('T-P plane at F=%.1f g/min', F_opt*1e5*60));
set(gca, 'YDir', 'normal');
colormap(hot);

% 2D slice at optimal P
subplot(1, 3, 2);
D_slice_P = squeeze(D_grid(:, i_P_opt, :))';
imagesc(T_range - 273, F_range*1e5*60, D_slice_P);
hold on;
plot(T_opt - 273, F_opt*1e5*60, 'wo', 'MarkerSize', 15, 'LineWidth', 3);
plot(T_opt - 273, F_opt*1e5*60, 'kx', 'MarkerSize', 12, 'LineWidth', 2);
colorbar;
xlabel('Temperature [°C]');
ylabel('Flow Rate [g/min]');
title(sprintf('T-F plane at P=%.0f bar', P_opt));
set(gca, 'YDir', 'normal');
colormap(hot);

% 2D slice at optimal T
subplot(1, 3, 3);
D_slice_T = squeeze(D_grid(i_T_opt, :, :))';
imagesc(P_range, F_range*1e5*60, D_slice_T);
hold on;
plot(P_opt, F_opt*1e5*60, 'wo', 'MarkerSize', 15, 'LineWidth', 3);
plot(P_opt, F_opt*1e5*60, 'kx', 'MarkerSize', 12, 'LineWidth', 2);
colorbar;
xlabel('Pressure [bar]');
ylabel('Flow Rate [g/min]');
title(sprintf('P-F plane at T=%.0f°C', T_opt-273));
set(gca, 'YDir', 'normal');
colormap(hot);

sgtitle(sprintf('Optimal Experimental Design: %s', criterion_name), 'FontSize', 14);

fprintf('\n=== Design Complete ===\n');

end
