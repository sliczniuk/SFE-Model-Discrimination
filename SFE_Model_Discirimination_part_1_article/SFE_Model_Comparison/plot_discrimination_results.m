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
%   'Figures'        - Cell array of figure names to plot (default: 'all')
%                      Yield-based: 'divergence_cumulative', 'distribution',
%                                   'probability', 'final_yield', 'trajectories'
%                      Rate-based:  'distribution_rate',
%                                   'rate_comparison'
%   'SaveFigs'       - Save figures to disk (default: false)
%   'SavePath'       - Path for saved figures (default: current directory)
%   'FileFormat'     - Format for saved figures (default: 'png')
%   'SaveNameSuffix' - Custom suffix appended to filenames (default: '')
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
addParameter(p, 'SaveNameSuffix', '', @ischar);
parse(p, results, varargin{:});

figures_to_plot = p.Results.Figures;
save_figs = p.Results.SaveFigs;
save_path = p.Results.SavePath;
file_format = p.Results.FileFormat;
save_name_suffix = p.Results.SaveNameSuffix;

if ischar(figures_to_plot) && strcmp(figures_to_plot, 'all')
    figures_to_plot = {'divergence_cumulative', 'distribution', ...
                       'probability', 'final_yield', 'trajectories', ...
                       'distribution_rate', ...
                       'rate_comparison'};
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

% Extract rate data if available
has_rate_data = isfield(results, 'Time_rate') && isfield(results, 'Rate_power_valid');
if has_rate_data
    Time_rate = results.Time_rate;
    Rate_power_valid = results.Rate_power_valid;
    Rate_linear_valid = results.Rate_linear_valid;
    Rate_power_nom = results.Rate_power_nom;
    Rate_linear_nom = results.Rate_linear_nom;
    n_time_rate = length(Time_rate);
end

%% Figure: Trajectory ensemble with confidence bands
if ismember('trajectories', figures_to_plot)
    fig_traj = figure('Name', 'Trajectory Ensemble with Parameter Uncertainty', 'Position', [100 100 800 800]);

    subplot(2, 1, 1);
    fill([Time_full, fliplr(Time_full)], [metrics.ci95_power(1,:), fliplr(metrics.ci95_power(2,:))], ...
        'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Power 95$\%$ CI');
    hold on;
    fill([Time_full, fliplr(Time_full)], [metrics.ci95_linear(1,:), fliplr(metrics.ci95_linear(2,:))], ...
        'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Linear 95$\%$ CI');
    plot(Time_full, metrics.mean_power, 'b-', 'LineWidth', 2, 'DisplayName', 'Mean Power');
    plot(Time_full, metrics.mean_linear, 'r-', 'LineWidth', 2, 'DisplayName', 'Mean Linear');
    xlabel('Time [min]');
    ylabel('Yield [g]');
    title('Mean Trajectories with 95$\%$ CI');
    legend('Location', 'southeast');
    grid on;

    subplot(2, 1, 2);
    plot(Time_rate, metrics.mean_rate_power, 'b-', 'LineWidth', 2, 'DisplayName', 'Mean Power');
    hold on;
    plot(Time_rate, metrics.mean_rate_linear, 'r-', 'LineWidth', 2, 'DisplayName', 'Mean Linear');
    xlabel('Time [min]');
    ylabel('Rate [g/min]');
    title('Mean Extraction Rates');
    legend('Location', 'northeast');
    grid on;
    set(gca,'fontsize', 14) 

    %sgtitle(sprintf('Trajectory Ensemble: T=%.0fK, P=%.0fbar, F=%.1e m3/s (N=%d)', T0, P0, F0, n_valid), 'FontSize', 12);

    if save_figs
        saveas(fig_traj, fullfile(save_path, ['trajectories' save_name_suffix '.' file_format]));
    end
end

%% Figure: Distribution evolution at selected times
if ismember('distribution', figures_to_plot)
    fig_dist = figure('Name', 'Distribution & KS Evolution', 'Position', [200 200 1100 1400]);

    t_select = [3, round(n_time_full/4), round(2*n_time_full/4), round(3*n_time_full/4), n_time_full];
    t_select = unique(max(t_select, 2));

    for j = 1:length(t_select)
        i_t = t_select(j);
        y_p = Y_power_valid(:, i_t);
        y_l = Y_linear_valid(:, i_t);

        edges = linspace(min([y_p; y_l])*0.95, max([y_p; y_l])*1.05, 30);

        % Histogram (left column)
        subplot(length(t_select), 2, 2*j-1);
        histogram(y_p, edges, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'Normalization', 'pdf', 'DisplayName', 'Power');
        hold on;
        histogram(y_l, edges, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'Normalization', 'pdf', 'DisplayName', 'Linear');
        title(sprintf('t = %.0f min, JS=%.3f', Time_full(i_t), metrics.js_divergence(i_t)));
        xlabel('Yield [g]');
        ylabel('PDF');
        if j == length(t_select)
            legend('Location', 'northeast');
        end
        grid on;

        % KS / CDF (right column)
        subplot(length(t_select), 2, 2*j);
        [f_p, x_p] = ecdf(y_p);
        [f_l, x_l] = ecdf(y_l);
        plot(x_p, f_p, 'b-', 'LineWidth', 2, 'DisplayName', 'Power');
        hold on;
        plot(x_l, f_l, 'r-', 'LineWidth', 2, 'DisplayName', 'Linear');
        xlabel('Yield [g]');
        ylabel('CDF');
        if j == length(t_select)
            legend('Location', 'southeast');
        end
        title(sprintf('KS = %.3f', metrics.ks_stat(i_t)));
        grid on;
        set(gca,'fontsize', 14) 
    end

    if save_figs
        saveas(fig_dist, fullfile(save_path, ['distribution' save_name_suffix '.' file_format]));
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
    ylabel('P(Power $\leq$ Linear) [$\%$]');
    title('Probability that Power Model Predicts Higher Yield');
    ylim([0 100]);
    grid on;
    set(gca,'fontsize', 14) 

    if save_figs
        saveas(fig_prob, fullfile(save_path, ['probability' save_name_suffix '.' file_format]));
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
    set(gca,'fontsize', 14) 

    %sgtitle(sprintf('Final Yield Comparison at t = %.0f min', ExtractionTime), 'FontSize', 14);

    if save_figs
        saveas(fig_final, fullfile(save_path, ['final_yield' save_name_suffix '.' file_format]));
    end
end

%% ========== RATE-BASED FIGURES ==========

%% Figure: Rate distribution evolution
if ismember('distribution_rate', figures_to_plot) && has_rate_data
    fig_dist_rate = figure('Name', 'Rate Distribution & KS Evolution', 'Position', [200 200 1100 1400]);

    t_select = [3, round(n_time_rate/4), round(2*n_time_rate/4), round(3*n_time_rate/4), n_time_rate];
    t_select = unique(max(t_select, 1));

    for j = 1:length(t_select)
        i_t = t_select(j);
        r_p = Rate_power_valid(:, i_t);
        r_l = Rate_linear_valid(:, i_t);

        edges = linspace(min([r_p; r_l])*0.95, max([r_p; r_l])*1.05, 30);

        % Histogram (left column)
        subplot(length(t_select), 2, 2*j-1);
        histogram(r_p, edges, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'Normalization', 'pdf', 'DisplayName', 'Power');
        hold on;
        histogram(r_l, edges, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'Normalization', 'pdf', 'DisplayName', 'Linear');
        title(sprintf('t = %.0f min, JS$_{rate}$ = %.3f', Time_rate(i_t), metrics.js_rate(i_t)));
        xlabel('Rate [g/min]');
        ylabel('PDF');
        if j == length(t_select)
            legend('Location', 'northeast');
        end
        grid on;

        % KS / CDF (right column)
        subplot(length(t_select), 2, 2*j);
        [f_p, x_p] = ecdf(r_p);
        [f_l, x_l] = ecdf(r_l);
        plot(x_p, f_p, 'b-', 'LineWidth', 2, 'DisplayName', 'Power');
        hold on;
        plot(x_l, f_l, 'r-', 'LineWidth', 2, 'DisplayName', 'Linear');
        xlabel('Rate [g/min]');
        ylabel('CDF');
        if j == length(t_select)
            legend('Location', 'southeast');
        end
        title(sprintf('KS$_{rate}$ = %.3f', metrics.ks_rate(i_t)), 'Interpreter','latex');
        grid on;
        set(gca,'fontsize', 14) 
    end

    if save_figs
        saveas(fig_dist_rate, fullfile(save_path, ['distribution_rate' save_name_suffix '.' file_format]));
    end
end

%% Figure: Yield vs Rate comparison
if ismember('rate_comparison', figures_to_plot) && has_rate_data
    fig_comp = figure('Name', 'Yield vs Rate Divergence Comparison', 'Position', [200 200 1000 800]);

    subplot(2, 2, 1);
    plot(Time_full, metrics.js_divergence, 'b-', 'LineWidth', 2, 'DisplayName', 'Yield-based');
    hold on;
    plot(Time_rate, metrics.js_rate, 'r-', 'LineWidth', 2, 'DisplayName', 'Rate-based');
    xlabel('Time [min]');
    ylabel('JS Divergence [nats]');
    title('JS Divergence: Yield vs Rate');
    %legend('Location', 'best');
    grid on;
    set(gca,'fontsize', 14) 

    subplot(2, 2, 2);
    plot(Time_full, metrics.ks_stat, 'b-', 'LineWidth', 2, 'DisplayName', 'Yield-based');
    hold on;
    plot(Time_rate, metrics.ks_rate, 'r-', 'LineWidth', 2, 'DisplayName', 'Rate-based');
    xlabel('Time [min]');
    ylabel('KS Statistic');
    title('KS Statistic: Yield vs Rate');
    %legend('Location', 'best');
    grid on;
    set(gca,'fontsize', 14) 

    subplot(2, 2, 3);
    plot(Time_full, metrics.auc, 'b-', 'LineWidth', 2, 'DisplayName', 'Yield-based');
    hold on;
    plot(Time_rate, metrics.auc_rate, 'r-', 'LineWidth', 2, 'DisplayName', 'Rate-based');
    yline(0.5, 'k--', 'No Discrimination', 'HandleVisibility','off');
    xlabel('Time [min]');
    ylabel('AUC');
    title('AUC: Yield vs Rate');
    legend('Location', 'best', 'Box','off');
    ylim([0.4, 1]);
    grid on;
    set(gca,'fontsize', 14) 

    subplot(2, 2, 4);
    plot(Time_full, metrics.kl_power_linear, 'b-', 'LineWidth', 2, 'DisplayName', 'Yield KL(P||L)');
    hold on;
    plot(Time_full, metrics.kl_linear_power, 'b--', 'LineWidth', 2, 'DisplayName', 'Yield KL(L||P)');
    plot(Time_rate, metrics.kl_rate_power_linear, 'r-', 'LineWidth', 2, 'DisplayName', 'Rate KL(P||L)');
    plot(Time_rate, metrics.kl_rate_linear_power, 'r--', 'LineWidth', 2, 'DisplayName', 'Rate KL(L||P)');
    xlabel('Time [min]');
    ylabel('KL Divergence [nats]');
    title('KL Divergence: Yield vs Rate');
    grid on;
    set(gca,'fontsize', 14) 
    legend('Location', 'north', 'Box','off','NumColumns',2, 'FontSize', 10);

    if save_figs
        saveas(fig_comp, fullfile(save_path, ['rate_comparison' save_name_suffix '.' file_format]));
    end
end

end
