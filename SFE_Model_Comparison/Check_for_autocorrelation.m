function [] = Check_for_autocorrelation(results)

%% Autocorrelation Analysis for Rate Trajectories
% Check if independence assumption is valid for trajectory-level metrics
% If autocorrelation dies quickly (lag 2-3), product of KDE probabilities is justified

fprintf('\n=============================================================================\n');
fprintf('   AUTOCORRELATION ANALYSIS FOR RATE TRAJECTORIES \n');
fprintf('=============================================================================\n\n');

% Use the last computed results (or compute fresh)
Rate_power = results.Rate_power_valid;
Rate_linear = results.Rate_linear_valid;
Time_rate = results.Time_rate;
n_valid = results.n_valid;

% Add small epsilon to avoid log(0)
eps_val = 1e-12;
log_Rate_power = log(abs(Rate_power) + eps_val);
log_Rate_linear = log(abs(Rate_linear) + eps_val);

% Compute autocorrelation for each trajectory, then average
n_lags = min(20, floor(size(Rate_power, 2) / 4));

acf_power_raw = zeros(n_valid, n_lags + 1);
acf_linear_raw = zeros(n_valid, n_lags + 1);
acf_power_log = zeros(n_valid, n_lags + 1);
acf_linear_log = zeros(n_valid, n_lags + 1);

for i = 1:n_valid
    % Raw rates
    acf_power_raw(i, :) = autocorr(Rate_power(i, :), 'NumLags', n_lags);
    acf_linear_raw(i, :) = autocorr(Rate_linear(i, :), 'NumLags', n_lags);

    % Log-transformed rates
    acf_power_log(i, :) = autocorr(log_Rate_power(i, :), 'NumLags', n_lags);
    acf_linear_log(i, :) = autocorr(log_Rate_linear(i, :), 'NumLags', n_lags);
end

% Average ACF across trajectories
mean_acf_power_raw = mean(acf_power_raw, 1);
mean_acf_linear_raw = mean(acf_linear_raw, 1);
mean_acf_power_log = mean(acf_power_log, 1);
mean_acf_linear_log = mean(acf_linear_log, 1);

% 95% CI for white noise: +/- 1.96/sqrt(n_time)
n_time_rate = size(Rate_power, 2);
ci_bound = 1.96 / sqrt(n_time_rate);

% Find lag at which ACF drops below CI (approximate independence)
lag_threshold_power_raw = find(abs(mean_acf_power_raw(2:end)) < ci_bound, 1);
lag_threshold_linear_raw = find(abs(mean_acf_linear_raw(2:end)) < ci_bound, 1);
lag_threshold_power_log = find(abs(mean_acf_power_log(2:end)) < ci_bound, 1);
lag_threshold_linear_log = find(abs(mean_acf_linear_log(2:end)) < ci_bound, 1);

fprintf('Autocorrelation Analysis Results:\n');
fprintf('  95%% CI bound for white noise: +/- %.4f\n\n', ci_bound);
fprintf('  Raw Rates:\n');
fprintf('    Power model:  ACF drops below CI at lag %d\n', lag_threshold_power_raw);
fprintf('    Linear model: ACF drops below CI at lag %d\n', lag_threshold_linear_raw);
fprintf('  Log-Transformed Rates:\n');
fprintf('    Power model:  ACF drops below CI at lag %d\n', lag_threshold_power_log);
fprintf('    Linear model: ACF drops below CI at lag %d\n', lag_threshold_linear_log);

% Interpretation
if isempty(lag_threshold_power_log) || lag_threshold_power_log > 5
    fprintf('\n  WARNING: Log-rates show persistent autocorrelation.\n');
    fprintf('           Consider using multivariate Gaussian approach for trajectory-level metrics.\n');
else
    fprintf('\n  Log-rates show low autocorrelation (dies by lag %d).\n', max([lag_threshold_power_log, lag_threshold_linear_log]));
    fprintf('  Independence assumption is approximately valid for product of KDE probabilities.\n');
end

%% Plot ACF
figure('Name', 'Rate Trajectory Autocorrelation', 'Position', [100 100 1000 800]);

subplot(2, 2, 1);
bar(0:n_lags, mean_acf_power_raw, 'FaceColor', [0.2 0.4 0.8]);
hold on;
yline(ci_bound, 'r--', 'LineWidth', 1.5);
yline(-ci_bound, 'r--', 'LineWidth', 1.5);
xlabel('Lag');
ylabel('ACF');
title('Power Model - Raw Rates');
xlim([-0.5, n_lags + 0.5]);
grid on;

subplot(2, 2, 2);
bar(0:n_lags, mean_acf_linear_raw, 'FaceColor', [0.8 0.2 0.2]);
hold on;
yline(ci_bound, 'r--', 'LineWidth', 1.5);
yline(-ci_bound, 'r--', 'LineWidth', 1.5);
xlabel('Lag');
ylabel('ACF');
title('Linear Model - Raw Rates');
xlim([-0.5, n_lags + 0.5]);
grid on;

subplot(2, 2, 3);
bar(0:n_lags, mean_acf_power_log, 'FaceColor', [0.2 0.4 0.8]);
hold on;
yline(ci_bound, 'r--', 'LineWidth', 1.5);
yline(-ci_bound, 'r--', 'LineWidth', 1.5);
xlabel('Lag');
ylabel('ACF');
title('Power Model - Log Rates');
xlim([-0.5, n_lags + 0.5]);
grid on;

subplot(2, 2, 4);
bar(0:n_lags, mean_acf_linear_log, 'FaceColor', [0.8 0.2 0.2]);
hold on;
yline(ci_bound, 'r--', 'LineWidth', 1.5);
yline(-ci_bound, 'r--', 'LineWidth', 1.5);
xlabel('Lag');
ylabel('ACF');
title('Linear Model - Log Rates');
xlim([-0.5, n_lags + 0.5]);
grid on;

sgtitle(sprintf('Autocorrelation Analysis (T=30°C, P=200 bar, F=3.33 g/s, n=%d trajectories)', n_valid));

print('autocorrelation_analysis.png', '-dpng', '-r300');


end