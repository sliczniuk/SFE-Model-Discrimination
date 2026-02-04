function residual_diagnostics(residuals_all, sigma2, Corr_k, param_names, cond_number)
% RESIDUAL_DIAGNOSTICS plots residual checks and reports VIFs.

diagfig = figure('Name', 'Residual Diagnostics', 'Position', [150 150 1000 800]);

% Residuals vs fitted values
subplot(2,2,1)
plot(residuals_all, 'o', 'MarkerSize', 4);
hold on;
yline(0, 'r--', 'LineWidth', 1.5);
yline(2*sqrt(sigma2), 'b--', 'LineWidth', 1);
yline(-2*sqrt(sigma2), 'b--', 'LineWidth', 1);
hold off;
xlabel('Observation Index');
ylabel('Standardized Residuals');
title('Residual Plot');
grid on;
legend('Residuals', 'Zero Line', '$\pm2\sigma$', 'Location', 'best');

% Histogram of residuals
subplot(2,2,2)
histogram(residuals_all, 20, 'Normalization', 'pdf');
hold on;
x_norm = linspace(min(residuals_all), max(residuals_all), 100);
y_norm = normpdf(x_norm, 0, sqrt(sigma2));
plot(x_norm, y_norm, 'r-', 'LineWidth', 2);
hold off;
xlabel('Residuals');
ylabel('Probability Density');
title('Residual Distribution');
legend('Observed', 'Normal', 'Location', 'best');
grid on;

% Q-Q plot
subplot(2,2,3)
qqplot(residuals_all);
title('Normal Q-Q Plot');
grid on;

% ACF of residuals
subplot(2,2,4)
N_observations = numel(residuals_all);
max_lag = min(20, floor(N_observations/4));
[acf_vals, lags] = autocorr(residuals_all, max_lag);
stem(lags, acf_vals, 'filled');
xlabel('Lag');
ylabel('Autocorrelation');
title('Autocorrelation Function of Residuals');
grid on;
hold on;
conf_bound = 1.96/sqrt(N_observations);
yline(conf_bound, 'r--', 'LineWidth', 1);
yline(-conf_bound, 'r--', 'LineWidth', 1);
hold off;

fprintf('\n=== Diagnostic Summary ===\n');
fprintf('Residual normality test (Jarque-Bera):\n');
[h_jb, p_jb] = jbtest(residuals_all);
fprintf('  p-value: %.4f (reject normality if p < 0.05)\n', p_jb);
if h_jb
    fprintf('  WARNING: Residuals may not be normally distributed.\n');
else
    fprintf('  Residuals appear normally distributed.\n');
end

fprintf('\nResidual autocorrelation test (Ljung-Box):\n');
[h_lb, p_lb] = lbqtest(residuals_all);
fprintf('  p-value: %.4f (reject independence if p < 0.05)\n', p_lb);
if h_lb
    fprintf('  WARNING: Residuals may be autocorrelated.\n');
else
    fprintf('  Residuals appear independent.\n');
end

fprintf('\nVariance inflation factors (VIF):\n');
if cond_number < 1e6
    VIF = diag(inv(Corr_k));
    fprintf('  Max VIF: %.2f (VIF > 10 indicates multicollinearity)\n', max(VIF));
    high_vif_idx = find(VIF > 10);
    if ~isempty(high_vif_idx)
        fprintf('  Parameters with high VIF:\n');
        for i = 1:length(high_vif_idx)
            fprintf('    %s: VIF = %.2f\n', param_names{high_vif_idx(i)}, VIF(high_vif_idx(i)));
        end
    end
else
    fprintf('  VIF calculation skipped (correlation matrix poorly conditioned)\n');
end

fprintf('\n=== Analysis Complete ===\n');

end
