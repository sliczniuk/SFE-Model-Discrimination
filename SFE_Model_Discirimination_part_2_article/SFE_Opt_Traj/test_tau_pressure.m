%% test_tau_pressure.m
%
% Tests the pressure relaxation dynamics for a range of tau_P values.
% Solves only the scalar pressure ODE:
%
%   dP/dt = (P_in - P) / tau_P
%
% for a step change and a ramp input, comparing how tau_P shapes
% the pressure trajectory and the pressure-work term dP/dt.
%
% This script is self-contained — it does not call modelSFE or any
% other model function.
%
% Outputs:
%   Figure 1 — Pressure trajectory P(t) for step input
%   Figure 2 — Pressure rate dP/dt(t) for step input (energy source term)
%   Figure 3 — Pressure trajectory P(t) for ramp input
%   Figure 4 — Pressure rate dP/dt(t) for ramp input
%   Figure 5 — Cumulative pressure work integral (total energy deposited)
%
% Reference: Section 3.2.6 of dissertation

clear; clc; close all;

%% =========================================================
%  Simulation parameters
%% =========================================================

% Pressure setpoints [bar]
P0    = 150;    % initial pressure
P_in  = 160;    % target pressure (step change of 10 bar)

% Total simulation time [s]
t_end = 3600;   % 1 hour, representative of extraction duration

% Tau values to compare [s]
% Includes: current implementation (600 s = 10 min),
%           recommended value (30 s),
%           and bracketing values for sensitivity
tau_values  = [10, 30, 60, 120, 600];
tau_labels  = {'$\tau$ = 10 s', '$\tau$ = 30 s', '$\tau$ = 60 s', ...
               '$\tau$ = 120 s', '$\tau$ = 600 s'};

% Colours for each tau (colorblind-friendly palette)
colours = [
    0.00, 0.45, 0.70;   % blue       — 10 s
    0.00, 0.62, 0.45;   % green      — 30 s  (recommended)
    0.94, 0.67, 0.00;   % yellow     — 60 s
    0.84, 0.37, 0.00;   % orange     — 120 s
    0.80, 0.10, 0.10;   % red        — 600 s (current)
];

% Line styles: recommended value (30 s) is solid bold, others dashed
line_styles = {'--', '-', '--', '--', ':'};
line_widths = [1.5, 1.5, 1.5, 1.5, 1.5];

% Bed porosity (for pressure-work term phi * dP/dt)
phi = 0.4;

% Ramp duration for ramp-input scenario [s]
t_ramp = 300;   % pressure ramp over 5 minutes

%% =========================================================
%  Analytical solution for step input
%  P(t) = P_in + (P0 - P_in) * exp(-t/tau)
%  dPdt(t) = (P_in - P0)/tau * exp(-t/tau)
%% =========================================================

t = linspace(0, t_end, 10000)';

% --- Figure 1: Pressure trajectory — step input ---
fig1 = figure('Name', 'Pressure trajectory (step input)', ...
              'Units', 'centimeters', 'Position', [2 2 18 10]);
hold on; box on; grid on;

for k = 1:numel(tau_values)
    tau = tau_values(k);
    P_t = P_in + (P0 - P_in) .* exp(-t ./ tau);
    plot(t/60, P_t, ...
        'Color',     colours(k,:),    ...
        'LineStyle', line_styles{k},  ...
        'LineWidth', line_widths(k));
end

% Reference lines
yline(P_in, 'k--', 'LineWidth', 1);
yline(P0,   'k:',  'LineWidth', 1);

% Mark 63.2% point for each tau (time constant definition)
for k = 1:numel(tau_values)
    tau = tau_values(k);
    P_63 = P0 + 0.632 * (P_in - P0);
    if tau/60 <= t_end/60
        plot(tau/60, P_63, 'o', 'Color', colours(k,:), ...
             'MarkerSize', 6, 'MarkerFaceColor', colours(k,:));
    end
end

xlabel('Time  [min]', 'FontSize', 11);
ylabel('Pressure  [bar]', 'FontSize', 11);
%title('Pressure relaxation — step input', 'FontSize', 12);
legend(tau_labels, 'Location', 'northoutside', 'FontSize', 12, NumColumns=5, Box='off');
xlim([0 t_end/60]);
ylim([P0-1, P_in+1]);
set(gca, 'FontSize', 16);
print('P_dynamics.png','-dpng', '-r500'); close all

% Inset: zoom on first 10 minutes to show differences
%{
axes('Position', [0.55 0.20 0.32 0.38]);
hold on; box on; grid on;
t_zoom = linspace(0, 600, 2000)';
for k = 1:numel(tau_values)
    tau = tau_values(k);
    P_t = P_in + (P0 - P_in) .* exp(-t_zoom ./ tau);
    plot(t_zoom/60, P_t, ...
        'Color',     colours(k,:),    ...
        'LineStyle', line_styles{k},  ...
        'LineWidth', line_widths(k));
end
yline(P_in, 'k--', 'LineWidth', 0.8);
xlabel('Time [min]', 'FontSize', 8);
ylabel('P [bar]',    'FontSize', 8);
title('First 10 min', 'FontSize', 8);
xlim([0 10]); ylim([P0-2, P_in+2]);
set(gca, 'FontSize', 8);

% --- Figure 2: dP/dt — step input (energy source term) ---
fig2 = figure('Name', 'dP/dt (step input) — energy source term', ...
              'Units', 'centimeters', 'Position', [2 14 18 10]);
hold on; box on; grid on;

for k = 1:numel(tau_values)
    tau  = tau_values(k);
    dPdt = ((P_in - P0) / tau) .* exp(-t ./ tau);
    plot(t/60, phi .* dPdt, ...
        'Color',     colours(k,:),    ...
        'LineStyle', line_styles{k},  ...
        'LineWidth', line_widths(k));
end

yline(0, 'k-', 'LineWidth', 0.8);
xlabel('Time  [min]', 'FontSize', 11);
ylabel('\phi \cdot dP/dt  [bar/s]', 'FontSize', 11);
title('Pressure-work source term  \phi \cdot \partial P/\partial t — step input', ...
      'FontSize', 12);
legend(tau_labels, 'Location', 'northeast', 'FontSize', 9);
xlim([0 t_end/60]);
set(gca, 'FontSize', 10);

% Inset: zoom on first 5 minutes
axes('Position', [0.40 0.35 0.45 0.40]);
hold on; box on; grid on;
t_zoom2 = linspace(0, 300, 2000)';
for k = 1:numel(tau_values)
    tau  = tau_values(k);
    dPdt = ((P_in - P0) / tau) .* exp(-t_zoom2 ./ tau);
    plot(t_zoom2/60, phi .* dPdt, ...
        'Color',     colours(k,:),    ...
        'LineStyle', line_styles{k},  ...
        'LineWidth', line_widths(k));
end
xlabel('Time [min]', 'FontSize', 8);
ylabel('\phi \cdot dP/dt', 'FontSize', 8);
title('First 5 min', 'FontSize', 8);
xlim([0 5]);
set(gca, 'FontSize', 8);

%% =========================================================
%  Ramp input scenario
%  P_in(t) = P0 + (P_in - P0) * min(t/t_ramp, 1)
%  Analytical ODE solution via convolution (numerical)
%% =========================================================

% Solve numerically using ode45 for each tau
ode_opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);

t_span = [0, t_end];
P_ramp_solutions = cell(numel(tau_values), 1);
t_ode            = cell(numel(tau_values), 1);

for k = 1:numel(tau_values)
    tau = tau_values(k);

    % Ramp input function
    P_in_ramp = @(t_now) P0 + (P_in - P0) .* min(t_now ./ t_ramp, 1);

    % ODE: dP/dt = (P_in(t) - P) / tau
    ode_rhs = @(t_now, P_now) (P_in_ramp(t_now) - P_now) ./ tau;

    [t_sol, P_sol] = ode45(ode_rhs, t_span, P0, ode_opts);

    t_ode{k}            = t_sol;
    P_ramp_solutions{k} = P_sol;
end

% --- Figure 3: Pressure trajectory — ramp input ---
fig3 = figure('Name', 'Pressure trajectory (ramp input)', ...
              'Units', 'centimeters', 'Position', [22 2 18 10]);
hold on; box on; grid on;

% Plot ideal ramp reference
t_ref  = linspace(0, t_end, 10000)';
P_ref  = P0 + (P_in - P0) .* min(t_ref ./ t_ramp, 1);
plot(t_ref/60, P_ref, 'k-', 'LineWidth', 1.2, 'DisplayName', ...
     sprintf('Ideal ramp (t_{ramp} = %d s)', t_ramp));

for k = 1:numel(tau_values)
    plot(t_ode{k}/60, P_ramp_solutions{k}, ...
        'Color',     colours(k,:),    ...
        'LineStyle', line_styles{k},  ...
        'LineWidth', line_widths(k),  ...
        'DisplayName', tau_labels{k});
end

yline(P_in, 'k--', 'P_{in} = 200 bar', ...
      'LabelHorizontalAlignment', 'left', 'LineWidth', 1);

xlabel('Time  [min]', 'FontSize', 11);
ylabel('Pressure  [bar]', 'FontSize', 11);
title(sprintf('Pressure relaxation — ramp input (t_{ramp} = %d s)', t_ramp), ...
      'FontSize', 12);
legend('Location', 'southeast', 'FontSize', 9);
xlim([0 t_end/60]);
ylim([P0 - 5, P_in + 5]);
set(gca, 'FontSize', 10);

% Inset: zoom on ramp region
axes('Position', [0.55 0.20 0.32 0.38]);
hold on; box on; grid on;
for k = 1:numel(tau_values)
    idx = t_ode{k} <= 600;
    plot(t_ode{k}(idx)/60, P_ramp_solutions{k}(idx), ...
        'Color',     colours(k,:),    ...
        'LineStyle', line_styles{k},  ...
        'LineWidth', line_widths(k));
end
t_ref_zoom = linspace(0, 600, 1000)';
P_ref_zoom = P0 + (P_in - P0) .* min(t_ref_zoom ./ t_ramp, 1);
plot(t_ref_zoom/60, P_ref_zoom, 'k-', 'LineWidth', 1.2);
xlabel('Time [min]', 'FontSize', 8);
ylabel('P [bar]',    'FontSize', 8);
title('Ramp region', 'FontSize', 8);
xlim([0 10]); ylim([P0-2, P_in+2]);
set(gca, 'FontSize', 8);

% --- Figure 4: dP/dt — ramp input ---
fig4 = figure('Name', 'dP/dt (ramp input) — energy source term', ...
              'Units', 'centimeters', 'Position', [22 14 18 10]);
hold on; box on; grid on;

for k = 1:numel(tau_values)
    tau = tau_values(k);

    % Numerically differentiate the ODE solution
    t_sol = t_ode{k};
    P_sol = P_ramp_solutions{k};
    dPdt_sol = gradient(P_sol, t_sol);

    plot(t_sol/60, phi .* dPdt_sol, ...
        'Color',     colours(k,:),    ...
        'LineStyle', line_styles{k},  ...
        'LineWidth', line_widths(k),  ...
        'DisplayName', tau_labels{k});
end

% Ideal ramp dP/dt
dPdt_ideal_during = (P_in - P0) / t_ramp;
plot([0, t_ramp/60], phi .* [dPdt_ideal_during, dPdt_ideal_during], ...
     'k-', 'LineWidth', 1.2, 'DisplayName', 'Ideal ramp');
plot([t_ramp/60, t_end/60], [0, 0], 'k-', 'LineWidth', 1.2, ...
     'HandleVisibility', 'off');

yline(0, 'k-', 'LineWidth', 0.8, 'HandleVisibility', 'off');
xlabel('Time  [min]', 'FontSize', 11);
ylabel('\phi \cdot dP/dt  [bar/s]', 'FontSize', 11);
title('Pressure-work source term  \phi \cdot \partial P/\partial t — ramp input', ...
      'FontSize', 12);
legend('Location', 'northeast', 'FontSize', 9);
xlim([0 t_end/60]);
set(gca, 'FontSize', 10);

%% =========================================================
%  Figure 5: Cumulative pressure work — step input
%  Integral of phi * dP/dt from 0 to t
%  Should converge to phi * Delta_P for all tau
%% =========================================================

fig5 = figure('Name', 'Cumulative pressure work', ...
              'Units', 'centimeters', 'Position', [44 2 18 10]);
hold on; box on; grid on;

t_fine = linspace(0, t_end, 50000)';

for k = 1:numel(tau_values)
    tau  = tau_values(k);
    dPdt = ((P_in - P0) / tau) .* exp(-t_fine ./ tau);

    % Cumulative integral using trapezoidal rule
    cum_work = cumtrapz(t_fine, phi .* dPdt);

    plot(t_fine/60, cum_work, ...
        'Color',     colours(k,:),    ...
        'LineStyle', line_styles{k},  ...
        'LineWidth', line_widths(k));
end

% Theoretical asymptote: phi * Delta_P
yline(phi * (P_in - P0), 'k--', ...
      sprintf('\\phi \\cdot \\Delta P = %.0f bar', phi*(P_in-P0)), ...
      'LabelHorizontalAlignment', 'left', 'LineWidth', 1.2);

xlabel('Time  [min]', 'FontSize', 11);
ylabel('\int \phi \cdot dP/dt \, dt  [bar]', 'FontSize', 11);
title('Cumulative pressure-work energy deposition — step input', 'FontSize', 12);
legend(tau_labels, 'Location', 'southeast', 'FontSize', 9);
xlim([0 t_end/60]);
ylim([0, phi*(P_in-P0)*1.15]);
set(gca, 'FontSize', 10);

%% =========================================================
%  Summary table printed to console
%% =========================================================

fprintf('\n');
fprintf('%-30s  %8s  %12s  %12s  %12s\n', ...
    'Scenario', 'tau [s]', 't_95%% [s]', ...
    'Peak dPdt [bar/s]', 'Delta_P_work [bar]');
fprintf('%s\n', repmat('-', 1, 80));

for k = 1:numel(tau_values)
    tau         = tau_values(k);
    t_95        = -tau * log(0.05);          % time to reach 95% of setpoint
    peak_dPdt   = (P_in - P0) / tau;        % peak of dP/dt source term
    delta_work  = phi * (P_in - P0);        % total energy (same for all)

    fprintf('%-30s  %8.0f  %12.1f  %12.4f  %12.1f\n', ...
        tau_labels{k}, tau, t_95, peak_dPdt, delta_work);
end

fprintf('\n');
fprintf('Note: total cumulative pressure work is identical for all tau.\n');
fprintf('      tau affects only the temporal distribution of energy deposition.\n\n');

%% =========================================================
%  Time-scale comparison
%% =========================================================

fprintf('--- Time-scale comparison ---\n');
tau_conv  = 1000;   % convective residence time [s], order of magnitude
tau_eq    = 10;     % LTE equilibration time [s]
tau_ext   = 3600;   % total extraction time [s]

fprintf('  LTE equilibration time:       tau_eq   = %5.0f s\n', tau_eq);
fprintf('  Convective residence time:    tau_conv  = %5.0f s\n', tau_conv);
fprintf('  Total extraction time:        t_f       = %5.0f s\n', tau_ext);
fprintf('  Defensible window for tau_P:  (%d, %d) s\n\n', tau_eq, tau_conv/10);

for k = 1:numel(tau_values)
    tau = tau_values(k);
    Pi  = tau / tau_conv;
    if tau > tau_eq && tau < tau_conv/10
        status = 'WITHIN defensible window';
    elseif tau <= tau_eq
        status = 'TOO FAST — violates LTE';
    else
        status = 'TOO SLOW — violates quasi-static P assumption';
    end
    fprintf('  tau = %4.0f s  (Pi = tau/tau_conv = %.3f)  ->  %s\n', ...
            tau, Pi, status);
end
fprintf('\n');

%}