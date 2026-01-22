function plot_discrimination_results(results, save_figures)
% PLOT_DISCRIMINATION_RESULTS
% Generates comprehensive visualizations for model discrimination analysis.
%
% Inputs:
%   results      - Structure from model_discrimination_analysis.m
%   save_figures - (optional) Boolean to save figures to files (default: false)

if nargin < 2
    save_figures = false;
end

%% Extract data
T_range = results.T_range;
P_range = results.P_range;
F_range = results.F_range;
D_integrated = results.D_integrated;
D_max = results.D_max;
t_max_diff = results.t_max_diff;
Y_final_power = results.Y_final_power;
Y_final_linear = results.Y_final_linear;

n_T = length(T_range);
n_P = length(P_range);
n_F = length(F_range);

%% Color settings
cmap_hot = hot(256);
cmap_cool = cool(256);
cmap_parula = parula(256);

%% ========================================================================
%  FIGURE 1: Overview Heatmaps
%  ========================================================================
fig1 = figure('Name', 'Model Discrimination Overview', 'Position', [50 50 1600 900]);

% Select flow rates to display
F_indices = round(linspace(1, n_F, min(6, n_F)));

for idx = 1:length(F_indices)
    i_F = F_indices(idx);
    subplot(2, 3, idx);

    D_slice = squeeze(D_integrated(:, :, i_F))';

    imagesc(T_range - 273, P_range, D_slice);
    colorbar;
    xlabel('Temperature [°C]');
    ylabel('Pressure [bar]');
    title(sprintf('F = %.1f g/min', F_range(i_F) * 1e5 * 60));
    set(gca, 'YDir', 'normal');
    colormap(gca, cmap_hot);

    % Mark maximum
    [~, idx_max] = max(D_slice(:));
    [i_P_max, i_T_max] = ind2sub(size(D_slice), idx_max);
    hold on;
    plot(T_range(i_T_max) - 273, P_range(i_P_max), 'co', 'MarkerSize', 12, 'LineWidth', 2);
    hold off;
end

sgtitle('Integrated Model Discrimination ∫|Y_{power} - Y_{linear}| dt', 'FontSize', 14, 'FontWeight', 'bold');

if save_figures
    saveas(fig1, 'fig_discrimination_overview.png');
    saveas(fig1, 'fig_discrimination_overview.fig');
end

%% ========================================================================
%  FIGURE 2: Maximum Difference Analysis
%  ========================================================================
fig2 = figure('Name', 'Maximum Difference Analysis', 'Position', [100 100 1400 500]);

% Find optimal flow index
[~, idx_opt] = max(D_integrated(:));
[~, ~, i_F_opt] = ind2sub([n_T, n_P, n_F], idx_opt);

% Maximum difference magnitude
subplot(1, 3, 1);
D_max_slice = squeeze(D_max(:, :, i_F_opt))';
imagesc(T_range - 273, P_range, D_max_slice);
colorbar;
xlabel('Temperature [°C]');
ylabel('Pressure [bar]');
title(sprintf('Max |ΔY| at F=%.1f g/min', F_range(i_F_opt)*1e5*60));
set(gca, 'YDir', 'normal');
colormap(gca, cmap_hot);

% Time of maximum difference
subplot(1, 3, 2);
t_max_slice = squeeze(t_max_diff(:, :, i_F_opt))';
imagesc(T_range - 273, P_range, t_max_slice);
cb = colorbar;
ylabel(cb, 'Time [min]');
xlabel('Temperature [°C]');
ylabel('Pressure [bar]');
title('Time of Maximum Difference');
set(gca, 'YDir', 'normal');
colormap(gca, cmap_parula);

% Final yield difference
subplot(1, 3, 3);
diff_final = squeeze(Y_final_power(:, :, i_F_opt) - Y_final_linear(:, :, i_F_opt))';
imagesc(T_range - 273, P_range, diff_final);
cb = colorbar;
ylabel(cb, 'Yield [g]');
xlabel('Temperature [°C]');
ylabel('Pressure [bar]');
title('Final Yield Difference (Power - Linear)');
set(gca, 'YDir', 'normal');
% Use diverging colormap for signed values
n_colors = 256;
cmap_div = [linspace(0,1,n_colors/2)', linspace(0,1,n_colors/2)', ones(n_colors/2,1);
            ones(n_colors/2,1), linspace(1,0,n_colors/2)', linspace(1,0,n_colors/2)'];
colormap(gca, cmap_div);
caxis([-max(abs(diff_final(:))), max(abs(diff_final(:)))]);

sgtitle('Maximum Difference Analysis', 'FontSize', 14, 'FontWeight', 'bold');

if save_figures
    saveas(fig2, 'fig_max_difference_analysis.png');
    saveas(fig2, 'fig_max_difference_analysis.fig');
end

%% ========================================================================
%  FIGURE 3: Operating Variable Effects
%  ========================================================================
fig3 = figure('Name', 'Operating Variable Effects', 'Position', [150 150 1400 400]);

% Marginal effects (averaging over other variables)
D_vs_T = squeeze(mean(mean(D_integrated, 3), 2));
D_vs_P = squeeze(mean(mean(D_integrated, 3), 1));
D_vs_F = squeeze(mean(mean(D_integrated, 2), 1));

% Also compute conditional effects at optimum
[~, i_T_opt, i_P_opt, i_F_opt] = find_optimum(D_integrated);
D_vs_T_cond = squeeze(D_integrated(:, i_P_opt, i_F_opt));
D_vs_P_cond = squeeze(D_integrated(i_T_opt, :, i_F_opt));
D_vs_F_cond = squeeze(D_integrated(i_T_opt, i_P_opt, :));

% Temperature effect
subplot(1, 3, 1);
yyaxis left;
plot(T_range - 273, D_vs_T, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'MarkerSize', 6);
ylabel('Mean Discrimination');
yyaxis right;
plot(T_range - 273, D_vs_T_cond, 'r--s', 'LineWidth', 1.5, 'MarkerFaceColor', 'r', 'MarkerSize', 5);
ylabel('At Optimal P,F');
xlabel('Temperature [°C]');
title('Effect of Temperature');
legend('Marginal', 'Conditional', 'Location', 'best');
grid on;

% Pressure effect
subplot(1, 3, 2);
yyaxis left;
plot(P_range, D_vs_P, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'MarkerSize', 6);
ylabel('Mean Discrimination');
yyaxis right;
plot(P_range, D_vs_P_cond, 'r--s', 'LineWidth', 1.5, 'MarkerFaceColor', 'r', 'MarkerSize', 5);
ylabel('At Optimal T,F');
xlabel('Pressure [bar]');
title('Effect of Pressure');
legend('Marginal', 'Conditional', 'Location', 'best');
grid on;

% Flow rate effect
subplot(1, 3, 3);
yyaxis left;
plot(F_range * 1e5 * 60, D_vs_F, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'MarkerSize', 6);
ylabel('Mean Discrimination');
yyaxis right;
plot(F_range * 1e5 * 60, D_vs_F_cond, 'r--s', 'LineWidth', 1.5, 'MarkerFaceColor', 'r', 'MarkerSize', 5);
ylabel('At Optimal T,P');
xlabel('Flow Rate [g/min]');
title('Effect of Flow Rate');
legend('Marginal', 'Conditional', 'Location', 'best');
grid on;

sgtitle('Effect of Operating Variables on Model Discrimination', 'FontSize', 14, 'FontWeight', 'bold');

if save_figures
    saveas(fig3, 'fig_operating_variable_effects.png');
    saveas(fig3, 'fig_operating_variable_effects.fig');
end

%% ========================================================================
%  FIGURE 4: 3D Surface Plot
%  ========================================================================
fig4 = figure('Name', '3D Discrimination Surface', 'Position', [200 200 1000 800]);

[T_mesh, P_mesh] = meshgrid(T_range - 273, P_range);

% Surface at optimal flow
D_surface = squeeze(D_integrated(:, :, i_F_opt))';

subplot(2, 2, [1 2]);
surf(T_mesh, P_mesh, D_surface, 'EdgeAlpha', 0.3);
xlabel('Temperature [°C]');
ylabel('Pressure [bar]');
zlabel('Discrimination');
title(sprintf('Discrimination Surface at F = %.1f g/min', F_range(i_F_opt)*1e5*60));
colorbar;
colormap(cmap_hot);
view([-30, 30]);
lighting gouraud;
camlight('headlight');

% Contour plot
subplot(2, 2, 3);
contourf(T_mesh, P_mesh, D_surface, 15);
xlabel('Temperature [°C]');
ylabel('Pressure [bar]');
title('Contour Plot');
colorbar;
colormap(gca, cmap_hot);

% Find and mark optimal point
[D_max_val, idx_max] = max(D_surface(:));
[i_P_max, i_T_max] = ind2sub(size(D_surface), idx_max);
hold on;
plot(T_range(i_T_max) - 273, P_range(i_P_max), 'ko', 'MarkerSize', 12, 'MarkerFaceColor', 'c', 'LineWidth', 2);
text(T_range(i_T_max) - 273 + 2, P_range(i_P_max), sprintf(' Optimal\n (%.0f°C, %.0f bar)', ...
    T_range(i_T_max)-273, P_range(i_P_max)), 'FontSize', 10);
hold off;

% Discrimination vs density (derived from P, T)
subplot(2, 2, 4);
% Compute approximate density for each (T, P) combination
rho_grid = zeros(n_T, n_P);
for i_T = 1:n_T
    for i_P = 1:n_P
        Z = Compressibility(T_range(i_T), P_range(i_P), []);  % Simplified
        % Approximate density using ideal gas + compressibility
        R = 8.314e-5;  % m³·bar/(mol·K)
        MW = 0.044;    % kg/mol for CO2
        rho_grid(i_T, i_P) = P_range(i_P) * MW / (Z * R * T_range(i_T));
    end
end

D_at_opt_F = squeeze(D_integrated(:, :, i_F_opt));
scatter(rho_grid(:), D_at_opt_F(:), 30, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Density [kg/m³]');
ylabel('Discrimination');
title('Discrimination vs Fluid Density');
grid on;

% Fit trend line
p = polyfit(rho_grid(:), D_at_opt_F(:), 2);
rho_fit = linspace(min(rho_grid(:)), max(rho_grid(:)), 100);
D_fit = polyval(p, rho_fit);
hold on;
plot(rho_fit, D_fit, 'r-', 'LineWidth', 2);
hold off;
legend('Data', 'Quadratic Fit', 'Location', 'best');

sgtitle('3D Analysis of Model Discrimination', 'FontSize', 14, 'FontWeight', 'bold');

if save_figures
    saveas(fig4, 'fig_3d_discrimination.png');
    saveas(fig4, 'fig_3d_discrimination.fig');
end

%% ========================================================================
%  FIGURE 5: Trajectory Comparison at Key Points
%  ========================================================================
fig5 = figure('Name', 'Trajectory Comparison', 'Position', [250 100 1400 800]);

if isfield(results, 'Y_trajectories_power') && isfield(results, 'Y_trajectories_linear')

    % Select 4 representative conditions
    % 1. Maximum discrimination
    [~, idx1] = max(D_integrated(:));
    [i1_T, i1_P, i1_F] = ind2sub([n_T, n_P, n_F], idx1);

    % 2. Minimum discrimination (where models agree)
    [~, idx2] = min(D_integrated(:));
    [i2_T, i2_P, i2_F] = ind2sub([n_T, n_P, n_F], idx2);

    % 3. High T, Low P corner
    i3_T = n_T; i3_P = 1; i3_F = ceil(n_F/2);

    % 4. Low T, High P corner
    i4_T = 1; i4_P = n_P; i4_F = ceil(n_F/2);

    conditions = {
        [i1_T, i1_P, i1_F], 'Max Discrimination';
        [i2_T, i2_P, i2_F], 'Min Discrimination';
        [i3_T, i3_P, i3_F], 'High T, Low P';
        [i4_T, i4_P, i4_F], 'Low T, High P'
    };

    Time = linspace(0, 600, size(results.Y_trajectories_power{1,1,1}, 2));

    for k = 1:4
        idx = conditions{k, 1};
        name = conditions{k, 2};

        Y_power = results.Y_trajectories_power{idx(1), idx(2), idx(3)};
        Y_linear = results.Y_trajectories_linear{idx(1), idx(2), idx(3)};

        if isempty(Y_power) || isempty(Y_linear)
            continue;
        end

        subplot(2, 4, k);
        plot(Time, Y_power, 'b-', 'LineWidth', 2);
        hold on;
        plot(Time, Y_linear, 'r--', 'LineWidth', 2);
        hold off;
        xlabel('Time [min]');
        ylabel('Yield [g]');
        title(sprintf('%s\nT=%.0f°C, P=%.0f bar, F=%.1f', name, ...
            T_range(idx(1))-273, P_range(idx(2)), F_range(idx(3))*1e5*60));
        if k == 1
            legend('Power', 'Linear', 'Location', 'southeast');
        end
        grid on;

        subplot(2, 4, k + 4);
        diff_traj = Y_power - Y_linear;
        plot(Time, diff_traj, 'k-', 'LineWidth', 2);
        xlabel('Time [min]');
        ylabel('ΔY [g]');
        title('Difference (Power - Linear)');
        grid on;
        hold on;
        yline(0, 'r--');
        hold off;
    end

    sgtitle('Yield Trajectories at Different Operating Conditions', 'FontSize', 14, 'FontWeight', 'bold');
else
    text(0.5, 0.5, 'Trajectory data not available', 'HorizontalAlignment', 'center', ...
        'FontSize', 14, 'Units', 'normalized');
end

if save_figures
    saveas(fig5, 'fig_trajectory_comparison.png');
    saveas(fig5, 'fig_trajectory_comparison.fig');
end

%% ========================================================================
%  FIGURE 6: Summary Statistics
%  ========================================================================
fig6 = figure('Name', 'Summary Statistics', 'Position', [300 150 1000 600]);

% Histogram of discrimination values
subplot(2, 2, 1);
histogram(D_integrated(:), 30, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'k');
xlabel('Integrated Discrimination');
ylabel('Frequency');
title('Distribution of Discrimination Values');
hold on;
xline(mean(D_integrated(:)), 'r--', 'LineWidth', 2);
xline(median(D_integrated(:)), 'g--', 'LineWidth', 2);
legend('', 'Mean', 'Median');
grid on;

% Histogram of time of max difference
subplot(2, 2, 2);
histogram(t_max_diff(:), 30, 'FaceColor', [0.9 0.6 0.3], 'EdgeColor', 'k');
xlabel('Time of Max Difference [min]');
ylabel('Frequency');
title('When Models Differ Most');
grid on;

% Scatter: Max diff vs Integrated diff
subplot(2, 2, 3);
scatter(D_max(:), D_integrated(:), 20, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Maximum Difference');
ylabel('Integrated Difference');
title('Max vs Integrated Discrimination');
grid on;
% Add correlation
r = corrcoef(D_max(:), D_integrated(:));
text(0.05, 0.95, sprintf('r = %.3f', r(1,2)), 'Units', 'normalized', ...
    'FontSize', 12, 'FontWeight', 'bold');

% Box plot by flow rate
subplot(2, 2, 4);
D_by_F = reshape(D_integrated, [], n_F);
boxplot(D_by_F, 'Labels', arrayfun(@(x) sprintf('%.1f', x*1e5*60), F_range, 'UniformOutput', false));
xlabel('Flow Rate [g/min]');
ylabel('Integrated Discrimination');
title('Discrimination Distribution by Flow Rate');
grid on;

sgtitle('Summary Statistics of Model Discrimination', 'FontSize', 14, 'FontWeight', 'bold');

if save_figures
    saveas(fig6, 'fig_summary_statistics.png');
    saveas(fig6, 'fig_summary_statistics.fig');
end

fprintf('Plotting complete.\n');
if save_figures
    fprintf('Figures saved to current directory.\n');
end

end

%% Helper function
function [D_max, i_T, i_P, i_F] = find_optimum(D)
    [D_max, idx] = max(D(:));
    [i_T, i_P, i_F] = ind2sub(size(D), idx);
end
