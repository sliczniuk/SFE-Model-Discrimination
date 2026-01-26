function plot_discrimination_results(results, varargin)
% PLOT_DISCRIMINATION_RESULTS Visualize model discrimination results
%
% Generates diagnostic plots for trajectory-based model discrimination,
% including divergence metrics, distribution evolution, and confidence bands.
%
% Syntax:
%   plot_discrimination_results(results)
%   plot_discrimination_results(results, 'Name', Value, ...)
%
% Inputs:
%   results - Struct from compute_discrimination_metrics containing:
%             .Time, .Y_power_valid, .Y_linear_valid, .metrics, etc.
%
% Optional Name-Value Pairs:
%   'Figures'     - Cell array of figure names to plot (default: 'all')
%                   Options: 'divergence', 'divergence_cumulative', 'distribution',
%                            'probability', 'final_yield', 'trajectories'
%   'SaveFigs'    - Save figures to disk (default: false)
%   'SavePath'    - Path for saved figures (default: current directory)
%   'FileFormat'  - Format for saved figures (default: 'png')
%
% Example:
%   [~, ~, results] = compute_discrimination_metrics(303, 150, 5e-5, 5, 600);
%   plot_discrimination_results(results);
%   plot_discrimination_results(results, 'Figures', {'divergence', 'distribution'});

%% Parse inputs
p = inputParser;
addRequired(p, 'results', @isstruct);
addParameter(p, 'Figures', 'all', @(x) ischar(x) || iscell(x));
addParameter(p, 'SaveFigs', false, @islogical);
addParameter(p, 'SavePath', '.', @ischar);
addParameter(p, 'FileFormat', 'png', @ischar);
parse(p, results, varargin{:});

figures_to_plot = p.Results.Figures;
save_figs = p.Results.SaveFigs;
save_path = p.Results.SavePath;
file_format = p.Results.FileFormat;

if ischar(figures_to_plot) && strcmp(figures_to_plot, 'all')
    figures_to_plot = {'divergence', 'divergence_cumulative', 'distribution', ...
                       'probability', 'final_yield', 'trajectories'};
end

%% Extract data from results
Time_full = results.Time;
Y_power_valid = results.Y_power_valid;
Y_linear_valid = results.Y_linear_valid;
Y_power_nom = results.Y_power_nom;
Y_linear_nom = results.Y_linear_nom;
metrics = results.metrics;
n_valid = results.n_valid;
T0 = results.T0;
P0 = results.P0;
F0 = results.F0;
ExtractionTime = results.ExtractionTime;
n_time_full = length(Time_full);

%% Figure: Trajectory ensemble with confidence bands
if ismember('trajectories', figures_to_plot)
    fig_traj = figure('Name', 'Trajectory Ensemble with Parameter Uncertainty', 'Position', [100 100 1200 800]);

    subplot(2, 2, 1);
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

    if save_figs
        saveas(fig_traj, fullfile(save_path, ['trajectories.' file_format]));
    end
end

%% Figure: Time-pointwise divergence metrics
if ismember('divergence', figures_to_plot)
    fig_div = figure('Name', 'Divergence Metrics Over Time', 'Position', [150 150 1200 400]);

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

    if save_figs
        saveas(fig_div, fullfile(save_path, ['divergence.' file_format]));
    end
end

%% Figure: Divergence metrics integrated (cumulative) over time
if ismember('divergence_cumulative', figures_to_plot)
    fig_div_cum = figure('Name', 'Divergence Metrics Integrated Over Time', 'Position', [150 150 1200 400]);

    subplot(1, 3, 1);
    plot(Time_full, cumsum(metrics.js_divergence), 'k-', 'LineWidth', 2);
    hold on;
    xline(metrics.js_max_time, 'r--', sprintf('Max=%.3f', metrics.js_max));
    xlabel('Time [min]');
    ylabel('Cumulative JS Divergence [nats]');
    title('Jensen-Shannon Divergence');
    grid on;

    subplot(1, 3, 2);
    plot(Time_full, cumsum(metrics.kl_power_linear), 'b-', 'LineWidth', 2, 'DisplayName', 'KL(P||L)');
    hold on;
    plot(Time_full, cumsum(metrics.kl_linear_power), 'r-', 'LineWidth', 2, 'DisplayName', 'KL(L||P)');
    plot(Time_full, cumsum(metrics.kl_linear_power + metrics.kl_power_linear), 'k-', 'LineWidth', 2, 'DisplayName', 'KL(L||P) + KL(P||L)');
    xlabel('Time [min]');
    ylabel('Cumulative KL Divergence [nats]');
    title('KL Divergence');
    legend('Location', 'best');
    grid on;

    subplot(1, 3, 3);
    plot(Time_full, cumsum(metrics.ks_stat), 'g-', 'LineWidth', 2);
    hold on;
    xline(metrics.ks_max_time, 'r--', sprintf('Max=%.3f', metrics.ks_max));
    xlabel('Time [min]');
    ylabel('Cumulative KS Statistic');
    title('Kolmogorov-Smirnov Statistic');
    grid on;

    sgtitle('Divergence Metrics Integrated Over Time', 'FontSize', 14);

    if save_figs
        saveas(fig_div_cum, fullfile(save_path, ['divergence_cumulative.' file_format]));
    end
end

%% Figure: Distribution evolution at selected times
if ismember('distribution', figures_to_plot)
    fig_dist = figure('Name', 'Distribution Evolution', 'Position', [200 200 1400 600]);

    t_select = [2, round(n_time_full/5), round(2*n_time_full/5), round(3*n_time_full/5), ...
                round(4*n_time_full/5), n_time_full];
    t_select = unique(max(t_select, 2));

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

    if save_figs
        saveas(fig_dist, fullfile(save_path, ['distribution.' file_format]));
    end
end

%% Figure: Probability of Power > Linear
if ismember('probability', figures_to_plot)
    fig_prob = figure('Name', 'Probability Power Greater', 'Position', [250 250 800 500]);

    plot(Time_full, metrics.prob_power_greater * 100, 'k-', 'LineWidth', 2);
    hold on;
    yline(50, 'r--', 'No Difference', 'LineWidth', 1.5);
    yline(95, 'g--', '95%');
    yline(5, 'g--', '5%');
    fill([Time_full(1), Time_full(end), Time_full(end), Time_full(1)], [45 45 55 55], ...
        'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

    xlabel('Time [min]');
    ylabel('P(Power > Linear) [%]');
    title('Probability that Power Model Predicts Higher Yield');
    ylim([0 100]);
    grid on;

    if save_figs
        saveas(fig_prob, fullfile(save_path, ['probability.' file_format]));
    end
end

%% Figure: Final yield comparison
if ismember('final_yield', figures_to_plot)
    fig_final = figure('Name', 'Final Yield Comparison', 'Position', [300 300 1000 400]);

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

    if save_figs
        saveas(fig_final, fullfile(save_path, ['final_yield.' file_format]));
    end
end

end
