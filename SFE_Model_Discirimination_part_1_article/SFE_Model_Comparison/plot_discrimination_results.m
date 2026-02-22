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
%                      Rate-based:  'distribution_rate', 'roc',
%                                   'rate_comparison', 'rate_comparison_ci'
%                      'roc' and 'rate_comparison_ci' require N_noiseCI>0 in compute step for CI bands.
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
                       'distribution_rate', 'roc', ...
                       'rate_comparison', 'rate_comparison_ci'};
end

%% Extract data from results
Time_full = results.Time;
Y_power_valid = results.Y_power_valid;
Y_linear_valid = results.Y_linear_valid;
Y_power_nom = results.Y_power_nom;
Y_linear_nom = results.Y_linear_nom;
metrics = results.metrics;
n_valid = results.n_valid_cum;
T0 = results.T0;
P0 = results.P0;
F0 = results.F0;
ExtractionTime = results.ExtractionTime;
n_time_full = length(Time_full);

% Extract noise-inflated observation arrays (for distribution diagnostics)
Y_power_obs  = results.Y_power_obs;
Y_linear_obs = results.Y_linear_obs;

% Extract rate data if available
has_rate_data = isfield(results, 'Time_rate') && isfield(results, 'Rate_power_valid');
if has_rate_data
    Time_rate = results.Time_rate;
    Rate_power_valid  = results.Rate_power_valid;
    Rate_linear_valid = results.Rate_linear_valid;
    Rate_power_nom    = results.Rate_power_nom;
    Rate_linear_nom   = results.Rate_linear_nom;
    Rate_power_obs    = results.Rate_power_obs;
    Rate_linear_obs   = results.Rate_linear_obs;
    n_time_rate = length(Time_rate);
    % Rate sampling window for axis labels (fallback for old results structs)
    if isfield(results, 'dt_rate')
        dt_rate = results.dt_rate;
    else
        dt_rate = results.Time_rate(1) * 2;   % approximate from first midpoint
    end
    rate_lbl = sprintf('$\\Delta Y_{%.0f\\,\\mathrm{min}}$ [g]', dt_rate);
    % Window edge time axes for stair-step plots
    Time_rate_end   = Time_rate + dt_rate/2;   % [min]  right edge of each dt_rate window
    Time_rate_start = Time_rate - dt_rate/2;   % [min]  left  edge of each dt_rate window
    % Extended x-vector for stairs: appends right edge of last window so the
    % final bar is drawn completely (MATLAB stairs otherwise drops the last step).
    Time_rate_stairs = [Time_rate_start(:); Time_rate_end(end)];  % (:) ensures column regardless of Time_rate orientation
end

% Guard: CI fields present only when N_noiseCI > 0 was used in compute step
has_metric_ci = isfield(metrics, 'ci95_js_cum');

% Variables needed for ROC CI draws (reuse same sigma/seed as compute CI block)
has_roc_ci = has_metric_ci && isfield(results, 'N_noiseCI') && isfield(results, 'Seed');
if has_roc_ci
    roc_N_noiseCI     = results.N_noiseCI;
    roc_Seed          = results.Seed;
    roc_sigma_pow_cum  = results.sigma_empirical_power_cum;
    roc_sigma_lin_cum  = results.sigma_empirical_linear_cum;
    roc_sigma_pow_diff = results.sigma_empirical_power_diff;
    roc_sigma_lin_diff = results.sigma_empirical_linear_diff;
end

%% Set global default font size (applies to all axes uniformly)
set(0, 'DefaultAxesFontSize', 14);

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
    legend('Location', 'northoutside', 'NumColumns', 2, 'Box', 'off');
    grid on;

    subplot(2, 1, 2);
    [xs_p_lo, ys_p_lo] = stairs(Time_rate_stairs, [metrics.ci95_rate_power(1,:), metrics.ci95_rate_power(1,end)]);
    [xs_p_hi, ys_p_hi] = stairs(Time_rate_stairs, [metrics.ci95_rate_power(2,:), metrics.ci95_rate_power(2,end)]);
    fill([xs_p_lo; flipud(xs_p_hi)], [ys_p_lo; flipud(ys_p_hi)], ...
        'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Power 95$\%$ CI');
    hold on;
    [xs_l_lo, ys_l_lo] = stairs(Time_rate_stairs, [metrics.ci95_rate_linear(1,:), metrics.ci95_rate_linear(1,end)]);
    [xs_l_hi, ys_l_hi] = stairs(Time_rate_stairs, [metrics.ci95_rate_linear(2,:), metrics.ci95_rate_linear(2,end)]);
    fill([xs_l_lo; flipud(xs_l_hi)], [ys_l_lo; flipud(ys_l_hi)], ...
        'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Linear 95$\%$ CI');
    stairs(Time_rate_stairs, [metrics.mean_rate_power,  metrics.mean_rate_power(end)],  'b-', 'LineWidth', 2, 'DisplayName', 'Mean Power');
    stairs(Time_rate_stairs, [metrics.mean_rate_linear, metrics.mean_rate_linear(end)], 'r-', 'LineWidth', 2, 'DisplayName', 'Mean Linear');
    xlabel('Time [min]');
    ylabel(rate_lbl, 'Interpreter', 'latex');
    title(sprintf('Mean Mass Increment with 95$\\%%$ CI ($\\Delta t = %.0f$ min)', dt_rate), 'Interpreter', 'latex');
    %legend('Location', 'northoutside', 'NumColumns', 2, 'Box', 'off');
    grid on;

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
        y_p = Y_power_obs(:, i_t);    % noise-inflated: parameter uncertainty + empirical sigma
        y_l = Y_linear_obs(:, i_t);

        edges = linspace(min([y_p; y_l])*0.95, max([y_p; y_l])*1.05, 30);

        % Histogram (left column)
        subplot(length(t_select), 2, 2*j-1);
        histogram(y_p, edges, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'Normalization', 'pdf', 'DisplayName', 'Power');
        hold on;
        histogram(y_l, edges, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'Normalization', 'pdf', 'DisplayName', 'Linear');
        title(sprintf('t = %.0f min, JS=%.3f', Time_full(i_t), metrics.js_divergence(i_t)));
        xlabel('Yield [g]');
        ylabel('PDF');
        if j == 1
            legend('Location', 'northoutside', 'NumColumns', 2, 'Box', 'off');
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
        if j == 1
            legend('Location', 'northoutside', 'NumColumns', 2, 'Box', 'off');
        end
        title(sprintf('KS = %.3f', metrics.ks_stat(i_t)));
        grid on;
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

    if save_figs
        saveas(fig_prob, fullfile(save_path, ['probability' save_name_suffix '.' file_format]));
    end
end

%% Figure: Final yield comparison
if ismember('final_yield', figures_to_plot)
    fig_final = figure('Name', 'Final Yield Comparison', 'Position', [300 300 1000 400]);

    subplot(1, 2, 1);
    Y_final_diff = Y_power_obs(:, end) - Y_linear_obs(:, end);  % noise-inflated
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
    scatter(Y_linear_obs(:, end), Y_power_obs(:, end), 20, 'filled', 'MarkerFaceAlpha', 0.3);  % noise-inflated
    hold on;
    lims = [min([Y_power_obs(:, end); Y_linear_obs(:, end)]), ...
            max([Y_power_obs(:, end); Y_linear_obs(:, end)])];
    plot(lims, lims, 'k--', 'LineWidth', 1.5);
    xlabel('Linear Final Yield [g]');
    ylabel('Power Final Yield [g]');
    title('Power vs Linear Final Yield');
    axis equal;
    xlim(lims);
    ylim(lims);
    grid on;

    %sgtitle(sprintf('Final Yield Comparison at t = %.0f min', ExtractionTime), 'FontSize', 14);

    if save_figs
        saveas(fig_final, fullfile(save_path, ['final_yield' save_name_suffix '.' file_format]));
    end
end

%% ========== RATE-BASED FIGURES ==========

%% Figure: Rate distribution evolution
if ismember('distribution_rate', figures_to_plot) && has_rate_data
    fig_dist_rate = figure('Name', 'Mass Increment Distribution & KS Evolution', 'Position', [200 200 1100 1400]);

    t_select = [3, round(n_time_rate/4), round(2*n_time_rate/4), round(3*n_time_rate/4), n_time_rate];
    t_select = unique(max(t_select, 1));

    for j = 1:length(t_select)
        i_t = t_select(j);
        r_p = Rate_power_obs(:, i_t);    % noise-inflated: parameter uncertainty + empirical sigma
        r_l = Rate_linear_obs(:, i_t);

        edges = linspace(min([r_p; r_l])*0.95, max([r_p; r_l])*1.05, 30);

        t_start_w = Time_rate(i_t) - dt_rate/2;
        t_end_w   = Time_rate(i_t) + dt_rate/2;

        % Histogram (left column)
        subplot(length(t_select), 2, 2*j-1);
        histogram(r_p, edges, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'Normalization', 'pdf', 'DisplayName', 'Power');
        hold on;
        histogram(r_l, edges, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'Normalization', 'pdf', 'DisplayName', 'Linear');
        title(sprintf('$t \\in [%.0f,\\,%.0f]$ min,  JS$_{\\Delta Y}$ = %.3f', ...
            t_start_w, t_end_w, metrics.js_rate(i_t)), 'Interpreter', 'latex');
        xlabel(rate_lbl, 'Interpreter', 'latex');
        ylabel('PDF');
        if j == 1
            legend('Location', 'northoutside', 'NumColumns', 2, 'Box', 'off');
        end
        grid on;

        % KS / CDF (right column)
        subplot(length(t_select), 2, 2*j);
        [f_p, x_p] = ecdf(r_p);
        [f_l, x_l] = ecdf(r_l);
        plot(x_p, f_p, 'b-', 'LineWidth', 2, 'DisplayName', 'Power');
        hold on;
        plot(x_l, f_l, 'r-', 'LineWidth', 2, 'DisplayName', 'Linear');
        xlabel(rate_lbl, 'Interpreter', 'latex');
        ylabel('CDF');
        if j == 1
            legend('Location', 'northoutside', 'NumColumns', 2, 'Box', 'off');
        end
        title(sprintf('$t \\in [%.0f,\\,%.0f]$ min,  KS$_{\\Delta Y}$ = %.3f', ...
            t_start_w, t_end_w, metrics.ks_rate(i_t)), 'Interpreter', 'latex');
        grid on;
    end

    if save_figs
        saveas(fig_dist_rate, fullfile(save_path, ['distribution_rate' save_name_suffix '.' file_format]));
    end
end

%% Figure: ROC Curves (cumulative yield + mass increment)
if ismember('roc', figures_to_plot)
    % Number of rows: 1 if no rate data, 2 if rate data available
    n_rows = 1 + has_rate_data;
    fig_roc = figure('Name', 'ROC Curves', 'Position', [200 200 1400 320*n_rows]);

    % Common FPR grid for CI band interpolation
    fpr_grid = linspace(0, 1, 200)';

    % Time-point selection — same logic as 'distribution' and 'distribution_rate'
    t_select_cum = unique(max([3, round(n_time_full/4), round(2*n_time_full/4), ...
                               round(3*n_time_full/4), n_time_full], 2));
    n_panels = length(t_select_cum);   % 5 panels

    %% Row 1: Cumulative yield ROC
    for j = 1:n_panels
        i_t = t_select_cum(j);
        y_p = Y_power_obs(:, i_t);
        y_l = Y_linear_obs(:, i_t);

        [fpr_main, tpr_main] = compute_roc_local(y_p, y_l);
        auc_val = metrics.auc(i_t);

        subplot(n_rows, n_panels, j);

        % CI band from N_noiseCI independent noise draws
        if has_roc_ci
            tpr_boot = zeros(roc_N_noiseCI, numel(fpr_grid));
            for b = 1:roc_N_noiseCI
                rng(roc_Seed + 100 + b);
                Y_p_b = max(0, Y_power_valid  + roc_sigma_pow_cum * randn(size(Y_power_valid)));
                Y_l_b = max(0, Y_linear_valid + roc_sigma_lin_cum * randn(size(Y_linear_valid)));
                [fpr_b, tpr_b] = compute_roc_local(Y_p_b(:, i_t), Y_l_b(:, i_t));
                tpr_boot(b, :) = interp1(fpr_b, tpr_b, fpr_grid, 'linear', 'extrap');
            end
            tpr_lo = max(0, prctile(tpr_boot, 2.5,  1));
            tpr_hi = min(1, prctile(tpr_boot, 97.5, 1));
            fill([fpr_grid; flipud(fpr_grid)], [tpr_lo'; flipud(tpr_hi')], ...
                 'b', 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            hold on;
        end

        plot(fpr_main, tpr_main, 'b-', 'LineWidth', 2, 'HandleVisibility', 'off');
        hold on;
        plot([0 1], [0 1], 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
        xlabel('FPR');
        ylabel('TPR');
        title(sprintf('$t = %.0f$ min,  AUC$=%.2f$', Time_full(i_t), auc_val), 'Interpreter', 'latex');
        axis square; grid on; xlim([0 1]); ylim([0 1]);
    end

    %% Row 2: Mass increment ROC (only if rate data available)
    if has_rate_data
        t_select_rate = unique(max([3, round(n_time_rate/4), round(2*n_time_rate/4), ...
                                    round(3*n_time_rate/4), n_time_rate], 1));
        % Pad to n_panels if fewer rate time points exist
        while length(t_select_rate) < n_panels
            t_select_rate(end+1) = t_select_rate(end); %#ok<AGROW>
        end
        t_select_rate = t_select_rate(1:n_panels);

        for j = 1:n_panels
            i_t = t_select_rate(j);
            r_p = Rate_power_obs(:, i_t);
            r_l = Rate_linear_obs(:, i_t);

            [fpr_main, tpr_main] = compute_roc_local(r_p, r_l);
            auc_val = metrics.auc_rate(i_t);

            t_start_w = Time_rate(i_t) - dt_rate/2;
            t_end_w   = Time_rate(i_t) + dt_rate/2;

            subplot(n_rows, n_panels, n_panels + j);

            % CI band from N_noiseCI draws on rate increments
            if has_roc_ci
                tpr_boot = zeros(roc_N_noiseCI, numel(fpr_grid));
                for b = 1:roc_N_noiseCI
                    rng(roc_Seed + 200 + b);
                    R_p_b = max(0, Rate_power_valid  + roc_sigma_pow_diff * randn(size(Rate_power_valid)));
                    R_l_b = max(0, Rate_linear_valid + roc_sigma_lin_diff * randn(size(Rate_linear_valid)));
                    [fpr_b, tpr_b] = compute_roc_local(R_p_b(:, i_t), R_l_b(:, i_t));
                    tpr_boot(b, :) = interp1(fpr_b, tpr_b, fpr_grid, 'linear', 'extrap');
                end
                tpr_lo = max(0, prctile(tpr_boot, 2.5,  1));
                tpr_hi = min(1, prctile(tpr_boot, 97.5, 1));
                fill([fpr_grid; flipud(fpr_grid)], [tpr_lo'; flipud(tpr_hi')], ...
                     'r', 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');
                hold on;
            end

            plot(fpr_main, tpr_main, 'r-', 'LineWidth', 2, 'HandleVisibility', 'off');
            hold on;
            plot([0 1], [0 1], 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
            xlabel('FPR');
            ylabel('TPR');
            title(sprintf('$t\\in[%.0f,\\,%.0f]$ min,  AUC$=%.2f$', t_start_w, t_end_w, auc_val), ...
                  'Interpreter', 'latex');
            axis square; grid on; xlim([0 1]); ylim([0 1]);
        end
    end

    sgtitle(sprintf('ROC Curves  (T=%.0fK, P=%.0fbar, F=%.1e m$^3$/s)', ...
        T0, P0, F0), 'Interpreter', 'latex', 'FontSize', 14);

    if save_figs
        saveas(fig_roc, fullfile(save_path, ['roc' save_name_suffix '.' file_format]));
    end
end

%% Figure: Yield vs Rate comparison
if ismember('rate_comparison', figures_to_plot) && has_rate_data
    fig_comp = figure('Name', 'Yield vs Rate Divergence Comparison', 'Position', [200 200 1000 800]);

    rate_win_str = sprintf('$\\Delta Y_{%.0f\\,\\mathrm{min}}$', dt_rate);

    subplot(2, 2, 1);
    plot(Time_full, metrics.js_divergence, 'b-', 'LineWidth', 2, 'DisplayName', 'Yield');
    hold on;
    stairs(Time_rate_stairs, [metrics.js_rate, metrics.js_rate(end)], 'r-', 'LineWidth', 2, 'DisplayName', ['Rate (', rate_win_str, ')']);
    xlabel('Time [min]');
    ylabel('JS Divergence [nats]');
    %title('JS Divergence: Yield vs Rate', 'Interpreter', 'latex');
    legend('Location', 'northoutside', 'NumColumns', 2, 'Box', 'off', 'Interpreter', 'latex');
    grid on;

    subplot(2, 2, 2);
    plot(Time_full, metrics.ks_stat, 'b-', 'LineWidth', 2, 'DisplayName', 'Yield');
    hold on;
    stairs(Time_rate_stairs, [metrics.ks_rate, metrics.ks_rate(end)], 'r-', 'LineWidth', 2, 'DisplayName', ['Rate (', rate_win_str, ')']);
    xlabel('Time [min]');
    ylabel('KS Statistic');
    %title('KS Statistic: Yield vs Rate', 'Interpreter', 'latex');
    legend('Location', 'northoutside', 'NumColumns', 2, 'Box', 'off', 'Interpreter', 'latex');
    grid on;

    subplot(2, 2, 3);
    % Symmetrize AUC for display: AUC* = max(AUC, 1-AUC) in [0.5, 1].
    % Stored metrics remain directional; symmetrization is plot-only.
    auc_sym      = max(metrics.auc,      1 - metrics.auc);
    auc_rate_sym = max(metrics.auc_rate, 1 - metrics.auc_rate);
    % Discrimination quality zone boundary lines (no shading)
    hold on;
    yline(0.7, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.8, 'HandleVisibility', 'off');
    yline(0.8, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 0.8, 'HandleVisibility', 'off');
    yline(0.9, '--', 'Color', [0.2 0.2 0.2], 'LineWidth', 0.8, 'HandleVisibility', 'off');
    plot(Time_full, auc_sym, 'b-', 'LineWidth', 2, 'DisplayName', 'Yield');
    stairs(Time_rate_stairs, [auc_rate_sym, auc_rate_sym(end)], 'r-', 'LineWidth', 2, 'DisplayName', ['Rate (', rate_win_str, ')']);
    xlabel('Time [min]');
    ylabel('$\mathrm{AUC}$', 'Interpreter', 'latex');
    %title('AUC: Yield vs Rate', 'Interpreter', 'latex');
    legend('Location', 'northoutside', 'NumColumns', 2, 'Box', 'off', 'Interpreter', 'latex');
    ylim([0.5, 1]);
    grid on;

    subplot(2, 2, 4);
    plot(Time_full, metrics.kl_power_linear, 'b-', 'LineWidth', 2, 'DisplayName', 'Yield $\mathrm{KL}(\mathcal{P}\|\mathcal{L})$');
    hold on;
    plot(Time_full, metrics.kl_linear_power, 'b--', 'LineWidth', 2, 'DisplayName', 'Yield $\mathrm{KL}(\mathcal{L}\|\mathcal{P})$');
    stairs(Time_rate_stairs, [metrics.kl_rate_power_linear, metrics.kl_rate_power_linear(end)], 'r-',  'LineWidth', 2, 'DisplayName', ['Rate $\mathrm{KL}(\mathcal{P}\|\mathcal{L})$ (', rate_win_str, ')']);
    stairs(Time_rate_stairs, [metrics.kl_rate_linear_power, metrics.kl_rate_linear_power(end)], 'r--', 'LineWidth', 2, 'DisplayName', ['Rate $\mathrm{KL}(\mathcal{L}\|\mathcal{P})$ (', rate_win_str, ')']);
    xlabel('Time [min]');
    ylabel('KL Divergence [nats]');
    %title('KL Divergence: Yield vs Rate', 'Interpreter', 'latex');
    grid on;
    legend('Location', 'northoutside', 'NumColumns', 2, 'Box', 'off', 'Interpreter', 'latex');

    if save_figs
        saveas(fig_comp, fullfile(save_path, ['rate_comparison' save_name_suffix '.' file_format]));
    end
end

%% Figure: Discrimination Metrics with Noise CI bands
if ismember('rate_comparison_ci', figures_to_plot) && has_rate_data && has_metric_ci
    fig_ci = figure('Name', 'Discrimination Metrics with Noise CI', 'Position', [200 200 1000 800]);

    rate_win_str = sprintf('$\\Delta Y_{%.0f\\,\\mathrm{min}}$', dt_rate);

    % Helper: build stair-polygon vertices for fill from lower/upper arrays
    % Returns concatenated [xs; flipud(xs_lo)] compatible with fill()
    % Usage: [xs_lo,ys_lo]=stairs(x,lo); [xs_hi,ys_hi]=stairs(x,hi);
    %        fill([xs_lo;flipud(xs_hi)],[ys_lo;flipud(ys_hi)], ...)

    % ---- Subplot 1: JS Divergence ----
    subplot(2, 2, 1);
    % Cumulative CI fill
    fill([Time_full, fliplr(Time_full)], ...
         [metrics.ci95_js_cum(1,:), fliplr(metrics.ci95_js_cum(2,:))], ...
         'b', 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'DisplayName', 'Yield CI 95\%');
    hold on;
    % Rate CI fill (stair polygon)
    [xs_lo, ys_lo] = stairs(Time_rate_stairs, [metrics.ci95_js_rate(1,:), metrics.ci95_js_rate(1,end)]);
    [xs_hi, ys_hi] = stairs(Time_rate_stairs, [metrics.ci95_js_rate(2,:), metrics.ci95_js_rate(2,end)]);
    fill([xs_lo; flipud(xs_hi)], [ys_lo; flipud(ys_hi)], ...
         'r', 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'DisplayName', [rate_win_str, ' CI 95\%']);
    % Metric curves on top — mean across N_noiseCI draws (smooth, noise-cancelled)
    plot(Time_full, metrics.mean_ci_js_cum, 'b-', 'LineWidth', 2, 'DisplayName', 'Yield');
    stairs(Time_rate_stairs, [metrics.mean_ci_js_rate, metrics.mean_ci_js_rate(end)], 'r-', 'LineWidth', 2, 'DisplayName', rate_win_str);
    xlabel('Time [min]');
    ylabel('JS Divergence [nats]');
    %title('JS Divergence with Noise CI', 'Interpreter', 'latex');
    legend('Location', 'northoutside', 'NumColumns', 2, 'Box', 'off', 'Interpreter', 'latex');
    grid on;

    % ---- Subplot 2: KS Statistic ----
    subplot(2, 2, 2);
    fill([Time_full, fliplr(Time_full)], ...
         [metrics.ci95_ks_cum(1,:), fliplr(metrics.ci95_ks_cum(2,:))], ...
         'b', 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'DisplayName', 'Yield CI 95\%');
    hold on;
    [xs_lo, ys_lo] = stairs(Time_rate_stairs, [metrics.ci95_ks_rate(1,:), metrics.ci95_ks_rate(1,end)]);
    [xs_hi, ys_hi] = stairs(Time_rate_stairs, [metrics.ci95_ks_rate(2,:), metrics.ci95_ks_rate(2,end)]);
    fill([xs_lo; flipud(xs_hi)], [ys_lo; flipud(ys_hi)], ...
         'r', 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'DisplayName', [rate_win_str, ' CI 95\%']);
    plot(Time_full, metrics.mean_ci_ks_cum, 'b-', 'LineWidth', 2, 'DisplayName', 'Yield');
    stairs(Time_rate_stairs, [metrics.mean_ci_ks_rate, metrics.mean_ci_ks_rate(end)], 'r-', 'LineWidth', 2, 'DisplayName', rate_win_str);
    xlabel('Time [min]');
    ylabel('KS Statistic');
    %title('KS Statistic with Noise CI', 'Interpreter', 'latex');
    legend('Location', 'northoutside', 'NumColumns', 2, 'Box', 'off', 'Interpreter', 'latex');
    grid on;

    % ---- Subplot 3: AUC ----
    subplot(2, 2, 3);
    % Symmetrize AUC for display: AUC* = max(AUC, 1-AUC) in [0.5, 1].
    % Mean curves
    mean_auc_cum_sym  = max(metrics.mean_ci_auc_cum,  1 - metrics.mean_ci_auc_cum);
    mean_auc_rate_sym = max(metrics.mean_ci_auc_rate, 1 - metrics.mean_ci_auc_rate);
    % CI bounds: apply symmetrization element-wise, then re-sort (lo<=hi) because
    % max(x, 1-x) can swap which percentile is larger when AUC crosses 0.5.
    ci_auc_cum_raw  = max(metrics.ci95_auc_cum,  1 - metrics.ci95_auc_cum);
    ci_auc_rate_raw = max(metrics.ci95_auc_rate, 1 - metrics.ci95_auc_rate);
    ci_auc_cum_lo   = min(ci_auc_cum_raw,  [], 1);
    ci_auc_cum_hi   = max(ci_auc_cum_raw,  [], 1);
    ci_auc_rate_lo  = min(ci_auc_rate_raw, [], 1);
    ci_auc_rate_hi  = max(ci_auc_rate_raw, [], 1);
    % Discrimination quality zone boundary lines (no shading)
    hold on;
    yline(0.7, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.8, 'HandleVisibility', 'off');
    yline(0.8, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 0.8, 'HandleVisibility', 'off');
    yline(0.9, '--', 'Color', [0.2 0.2 0.2], 'LineWidth', 0.8, 'HandleVisibility', 'off');
    % CI fills (symmetrized)
    fill([Time_full, fliplr(Time_full)], ...
         [ci_auc_cum_lo, fliplr(ci_auc_cum_hi)], ...
         'b', 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'DisplayName', 'Yield CI 95\%');
    [xs_lo, ys_lo] = stairs(Time_rate_stairs, [ci_auc_rate_lo, ci_auc_rate_lo(end)]);
    [xs_hi, ys_hi] = stairs(Time_rate_stairs, [ci_auc_rate_hi, ci_auc_rate_hi(end)]);
    fill([xs_lo; flipud(xs_hi)], [ys_lo; flipud(ys_hi)], ...
         'r', 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'DisplayName', [rate_win_str, ' CI 95\%']);
    % Metric curves (symmetrized)
    plot(Time_full, mean_auc_cum_sym, 'b-', 'LineWidth', 2, 'DisplayName', 'Yield');
    stairs(Time_rate_stairs, [mean_auc_rate_sym, mean_auc_rate_sym(end)], 'r-', 'LineWidth', 2, 'DisplayName', rate_win_str);
    xlabel('Time [min]');
    ylabel('$\mathrm{AUC}$', 'Interpreter', 'latex');
    %title('AUC with Noise CI', 'Interpreter', 'latex');
    legend('Location', 'northoutside', 'NumColumns', 2, 'Box', 'off', 'Interpreter', 'latex');
    ylim([0.5, 1]);
    grid on;

    % ---- Subplot 4: KL Divergence ----
    subplot(2, 2, 4);
    % Cumulative CI fills (two directions, same blue shade)
    fill([Time_full, fliplr(Time_full)], ...
         [metrics.ci95_kl_pl_cum(1,:), fliplr(metrics.ci95_kl_pl_cum(2,:))], ...
         'b', 'FaceAlpha', 0.20, 'EdgeColor', 'none', 'DisplayName', 'Yield CI 95\%');
    hold on;
    fill([Time_full, fliplr(Time_full)], ...
         [metrics.ci95_kl_lp_cum(1,:), fliplr(metrics.ci95_kl_lp_cum(2,:))], ...
         'b', 'FaceAlpha', 0.20, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    % Rate CI fills (two directions, red shade)
    [xs_lo, ys_lo] = stairs(Time_rate_stairs, [metrics.ci95_kl_pl_rate(1,:), metrics.ci95_kl_pl_rate(1,end)]);
    [xs_hi, ys_hi] = stairs(Time_rate_stairs, [metrics.ci95_kl_pl_rate(2,:), metrics.ci95_kl_pl_rate(2,end)]);
    fill([xs_lo; flipud(xs_hi)], [ys_lo; flipud(ys_hi)], ...
         'r', 'FaceAlpha', 0.20, 'EdgeColor', 'none', 'DisplayName', [rate_win_str, ' CI 95\%']);
    [xs_lo, ys_lo] = stairs(Time_rate_stairs, [metrics.ci95_kl_lp_rate(1,:), metrics.ci95_kl_lp_rate(1,end)]);
    [xs_hi, ys_hi] = stairs(Time_rate_stairs, [metrics.ci95_kl_lp_rate(2,:), metrics.ci95_kl_lp_rate(2,end)]);
    fill([xs_lo; flipud(xs_hi)], [ys_lo; flipud(ys_hi)], ...
         'r', 'FaceAlpha', 0.20, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    % Metric curves — mean across N_noiseCI draws (smooth, noise-cancelled)
    plot(Time_full, metrics.mean_ci_kl_pl_cum, 'b-',  'LineWidth', 2, 'DisplayName', 'Yield $\mathrm{KL}(\mathcal{P}\|\mathcal{L})$');
    plot(Time_full, metrics.mean_ci_kl_lp_cum, 'b--', 'LineWidth', 2, 'DisplayName', 'Yield $\mathrm{KL}(\mathcal{L}\|\mathcal{P})$');
    stairs(Time_rate_stairs, [metrics.mean_ci_kl_pl_rate, metrics.mean_ci_kl_pl_rate(end)], 'r-',  'LineWidth', 2, 'DisplayName', [rate_win_str, ' $\mathrm{KL}(\mathcal{P}\|\mathcal{L})$']);
    stairs(Time_rate_stairs, [metrics.mean_ci_kl_lp_rate, metrics.mean_ci_kl_lp_rate(end)], 'r--', 'LineWidth', 2, 'DisplayName', [rate_win_str, ' $\mathrm{KL}(\mathcal{L}\|\mathcal{P})$']);
    xlabel('Time [min]');
    ylabel('KL Divergence [nats]');
    %title('KL Divergence with Noise CI', 'Interpreter', 'latex');
    legend('Location', 'northoutside', 'NumColumns', 2, 'Box', 'off', 'Interpreter', 'latex');
    grid on;

    sgtitle(sprintf('Discrimination Metrics with Noise CI  (T=%.0fK, P=%.0fbar, F=%.1e m$^3$/s)', ...
        T0, P0, F0), 'Interpreter', 'latex', 'FontSize', 14);

    if save_figs
        saveas(fig_ci, fullfile(save_path, ['rate_comparison_ci' save_name_suffix '.' file_format]));
    end
end

end  % main function

%% ========== LOCAL HELPER FUNCTIONS ==========

function [fpr, tpr] = compute_roc_local(y_p, y_l)
% COMPUTE_ROC_LOCAL  Empirical ROC curve for two-sample discrimination.
%
%   [FPR, TPR] = COMPUTE_ROC_LOCAL(Y_P, Y_L)
%
%   Y_P : column vector of "positive" class samples (Power model)
%   Y_L : column vector of "negative" class samples (Linear model)
%
%   At threshold T, a sample is classified as "Power" (positive) if value >= T.
%     TPR(T) = mean(y_p >= T)  — fraction of Power samples correctly identified
%     FPR(T) = mean(y_l >= T)  — fraction of Linear samples misclassified
%
%   Returns FPR and TPR sorted so that (FPR, TPR) traces the ROC from (0,0) to (1,1).
%   The area under the resulting curve equals the Mann-Whitney AUC = P(y_p > y_l).

    % Unique thresholds: sweep from +Inf down; add sentinel -Inf to close at (1,1)
    thresholds = [-inf; unique([y_p; y_l], 'sorted')];
    thresholds = flipud(thresholds);   % descending: starts at highest → (0,0), ends at -inf → (1,1)

    n = numel(thresholds);
    tpr = zeros(n, 1);
    fpr = zeros(n, 1);
    for k = 1:n
        T = thresholds(k);
        tpr(k) = mean(y_p >= T);
        fpr(k) = mean(y_l >= T);
    end

    % Deduplicate FPR values (required for interp1 in CI boot loop).
    % Multiple thresholds can yield the same FPR when no y_l samples fall
    % between consecutive thresholds.  Keep the entry with the highest TPR
    % at each unique FPR level ('last' because fpr is non-decreasing and
    % tpr is also non-decreasing along the threshold sweep).
    [fpr, ia] = unique(fpr, 'last');
    tpr = tpr(ia);
    % Guarantee (0,0) is the first point (it should already be there from the
    % -Inf sentinel threshold, but deduplication can occasionally remove it).
    if fpr(1) ~= 0
        fpr = [0; fpr];
        tpr = [0; tpr];
    end
end
