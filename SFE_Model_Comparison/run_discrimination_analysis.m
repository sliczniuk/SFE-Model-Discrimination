%% Run Discrimination Analysis
% Example script demonstrating the use of modular discrimination functions.
%
% This script shows how to:
%   1. Compute discrimination metrics at a single operating point
%   2. Generate diagnostic visualizations
%   3. Scan across multiple operating conditions
%
% Functions used:
%   - compute_discrimination_metrics: Core computation function
%   - plot_discrimination_results: Visualization function

%% Initialization
startup;
addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');

fprintf('=============================================================================\n');
fprintf('   MODEL DISCRIMINATION ANALYSIS - MODULAR VERSION                          \n');
fprintf('=============================================================================\n\n');

%{
%% Example 1: Single Operating Point Analysis
fprintf('--- Example 1: Single Operating Point ---\n\n');

% Define operating conditions
T = 35 + 273;    % Temperature [K] (30 C)
P = 200;         % Pressure [bar]
F = 5e-5;        % Flow rate [m3/s]
timeStep = 30;    % Time step [min]
finalTime = 600; % Extraction time [min]

% Compute discrimination metrics
[max_KS, integrated_JS, results] = compute_discrimination_metrics(T, P, F, timeStep, finalTime, ...
    'N_MC', 500, 'Verbose', true);

% Print summary
fprintf('\n--- Summary ---\n');
fprintf('Max KS Statistic: %.4f (at t=%.0f min)\n', max_KS, results.metrics.ks_max_time);
fprintf('Integrated JS Divergence: %.4f nats*min\n', integrated_JS);
fprintf('Final yield difference: %.4f +/- %.4f g\n', ...
    results.metrics.final_diff_mean, results.metrics.final_diff_std);

% Generate plots
%plot_discrimination_results(results, 'Figures', {'divergence', 'distribution', 'final_yield'});
%}
%%

timeStep = 30;    % Time step [min]
finalTime = 600; % Extraction time [min]

T = linspace(30,40,10)+273;
F = linspace(3.3,6.7,10)*1e-5;

[T_grid,F_grid] = meshgrid(T,F);

T_grid = T_grid(:);
F_grid = F_grid(:);

%% Example 2: Multiple Operating Conditions (Pressure Sweep)
fprintf('\n--- Example 2: Pressure Sweep ---\n\n');

pressures = [100];
n_pressures = length(pressures);

sweep_results = struct();
sweep_results.P = pressures;
sweep_results.max_KS = zeros(1, n_pressures);
sweep_results.integrated_JS = zeros(1, n_pressures);

for i = 1:n_pressures

    JS_grid = nan( size(T_grid) );
    JS_grid = JS_grid(:);

    for j = 1:numel(T_grid)

        T_i = T_grid(j);
        F_i = F_grid(j);
        
        fprintf('Processing P = %d bar, T = %d [C], F = %d g/s...\n', pressures(i), T_i-273, F_i*1e5);

        [max_KS_i, int_JS_i] = compute_discrimination_metrics(T_i, pressures(i), F_i, timeStep, finalTime, ...
            'N_MC', 50, 'Verbose', false);

        JS_grid(j) = int_JS_i;

    end

    T_cordinate = reshape(T_grid(:),numel(F),numel(T));
    F_cordinate = reshape(F_grid(:),numel(F),numel(T));
    JS_cordinate = reshape(JS_grid(:),numel(F),numel(T));

    figure()
    pcolor(T_cordinate,F_cordinate,JS_cordinate)

    figure()
    imagesc(T_grid,F_grid,JS_grid)

    %sweep_results.max_KS(i) = max_KS_i;
    %sweep_results.integrated_JS(i) = int_JS_i;
end

% Plot sweep results
%{
figure('Name', 'Pressure Sweep Results', 'Position', [100 100 800 400]);

subplot(1, 2, 1);
bar(pressures, sweep_results.max_KS);
xlabel('Pressure [bar]');
ylabel('Max KS Statistic');
title('Maximum KS vs Pressure');
grid on;

subplot(1, 2, 2);
bar(pressures, sweep_results.integrated_JS);
xlabel('Pressure [bar]');
ylabel('Integrated JS [nats*min]');
title('Integrated JS Divergence vs Pressure');
grid on;

sgtitle(sprintf('Discrimination Metrics vs Pressure (T=%.0fK, F=%.1e m3/s)', T, F));

fprintf('\n--- Sweep Complete ---\n');
fprintf('Pressure [bar]   Max KS    Int JS\n');
fprintf('------------------------------------\n');
for i = 1:n_pressures
    fprintf('%8d        %.4f    %.4f\n', pressures(i), sweep_results.max_KS(i), sweep_results.integrated_JS(i));
end

%% Save results
%save('discrimination_analysis_results.mat', 'results', 'sweep_results');
%fprintf('\nResults saved to discrimination_analysis_results.mat\n');

fprintf('\n=== Analysis Complete ===\n');
%}