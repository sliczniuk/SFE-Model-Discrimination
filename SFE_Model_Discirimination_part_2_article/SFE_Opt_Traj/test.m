%% Single-run wrapper for trajectory optimization
% Uses the refactored solver and plots the seeded initial guess vs result.

startup;
addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');

%%
seed = 1;
result = solve_opt_traj_once(seed, true);

fprintf('seed=%d | success=%d | j=% .6e | iter=%g | status=%s\n', ...
    result.seed, result.success, result.j, result.iter_count, result.status);
if ~result.success && ~isempty(result.error_message)
    fprintf('Solver error (debug values returned): %s\n', result.error_message);
end

%%
figure;
subplot(3,1,1)
hold on
stairs(result.Time, [result.feedTemp0, result.feedTemp0(end)] - 273, 'LineWidth', 2)
stairs(result.Time, [result.feedTemp,  result.feedTemp(end)]  - 273, 'LineWidth', 2)
hold off
xlabel('Time min')
ylabel('T C')
legend('Initial guess', 'Optimized', 'Location', 'best')

subplot(3,1,2)
hold on
stairs(result.Time, [result.feedFlow0, result.feedFlow0(end)], 'LineWidth', 2)
stairs(result.Time, [result.feedFlow,  result.feedFlow(end)],  'LineWidth', 2)
hold off
xlabel('Time min')
ylabel('F kg/s')
legend('Initial guess', 'Optimized', 'Location', 'best')

subplot(3,1,3)
hold on
x = result.yieldTime(:)';
p_lo = result.yieldPower  - result.yieldPowerCI;
p_hi = result.yieldPower  + result.yieldPowerCI;
l_lo = result.yieldLinear - result.yieldLinearCI;
l_hi = result.yieldLinear + result.yieldLinearCI;

fill([x, fliplr(x)], [p_lo, fliplr(p_hi)], [0.20, 0.45, 0.85], ...
    'FaceAlpha', 0.15, 'EdgeColor', 'none');
fill([x, fliplr(x)], [l_lo, fliplr(l_hi)], [0.90, 0.35, 0.20], ...
    'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(x, result.yieldPower,  '-', 'Color', [0.10, 0.30, 0.75], 'LineWidth', 2);
plot(x, result.yieldLinear, '-', 'Color', [0.75, 0.20, 0.10], 'LineWidth', 2);
hold off
xlabel('Time min')
ylabel('Yield')
legend('Power 95% CI', 'Linear 95% CI', 'Power model', 'Linear model', 'Location', 'best')
title('Optimized yield curves with predictive 95% CI')
