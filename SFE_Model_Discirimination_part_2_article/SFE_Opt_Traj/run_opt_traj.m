%% Initialization
startup;
addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
import casadi.*

fprintf('=============================================================================\n');
fprintf('   Optimal trajectory for the model discrimination \n');
fprintf('=============================================================================\n\n');

%% Optimizer Settings - IMPROVED CONVERGENCE
nlp_opts = struct;
nlp_opts.ipopt.max_iter              = 50;           % Increased from 50
%nlp_opts.ipopt.max_cpu_time          = Time_max * 3600;
nlp_opts.ipopt.tol                   = 1e-7;          % Convergence tolerance
nlp_opts.ipopt.acceptable_tol        = 1e-5;          % Backup tolerance
nlp_opts.ipopt.acceptable_iter       = 10;            % Accept after 10 iterations
nlp_opts.ipopt.hessian_approximation = 'limited-memory';
nlp_opts.ipopt.limited_memory_max_history = 20;
nlp_opts.ipopt.print_level           = 5;             % Suppress IPOPT iteration output
nlp_opts.print_time                  = false;          % Suppress CasADi timing summary table

fprintf('=== Optimizer Configuration ===\n');
fprintf('Max iterations: %d\n', nlp_opts.ipopt.max_iter);
fprintf('Convergence tolerance: %.0e\n', nlp_opts.ipopt.tol);
%fprintf('Max CPU time: %.1f hours\n\n', Time_max);

opti = casadi.Opti();
opti.solver('ipopt', nlp_opts);

%% Set up the simulation
timeStep  = 5;  % Time step [min]
finalTime = 600; % Extraction time [min]
Time      = 0 : timeStep: finalTime;

N_starts = 1; %pool.NumWorkers;

pressures_bar = [200];   % extraction pressures to optimise

sigma2_cases = [2.45e-2, 1.386e-3, 1.007e-2]; % Mean empirical sigma as given in the report

% Decision-variable bounds (defined here so they can be used in the MX expressions below)
T_lb = 303;    T_ub = 313;
F_lb = 3.3e-5; F_ub = 6.7e-5;

%% Load parameters and data
Parameters_table = readtable('Parameters.csv');
Parameters       = num2cell(Parameters_table{:,3});
LabResults       = xlsread('dataset_2.xlsx');

which_k = (0:9) + 44;     % Indices of parameters to fit (44-53)
Nk      = numel(which_k);  % 10 parameters (4 Power + 6 Linear)
k1      = opti.parameter(4);
k2      = opti.parameter(6);
k       = [k1; k2];
P_param = opti.parameter(1);   % extraction pressure [bar] — swept externally

k1_val  = cell2mat(Parameters((0:3) + 44) );
k2_val  = cell2mat(Parameters((4:9) + 44) );

%% Sample Time Matching
%SAMPLE = LabResults(6:19, 1);
%SAMPLE = SAMPLE(2:end);
SAMPLE = [15, 30, 60, 120, 240, 360, 600]

sample_tol = 1e-3;
N_Sample   = zeros(size(SAMPLE));
for ii = 1:numel(SAMPLE)
    [delta, idx] = min(abs(Time - SAMPLE(ii)));
    if delta > sample_tol
        error('Sample time mismatch at index %d (delta=%.3g min)', ii, delta);
    end
    N_Sample(ii) = idx;
end

%% Parameter covariance matrices
% Power model covariance (4x4): [k_w0, a_w, b_w, n_k]
% Cumulative yield
Cov_power_cum = [
    1.0035e-02,  1.1795e-02,  1.8268e-03,  2.5611e-02;
    1.1795e-02,  5.6469e-02,  3.1182e-03,  2.8266e-02;
    1.8268e-03,  3.1182e-03,  5.7241e-03,  6.4459e-03;
    2.5611e-02,  2.8266e-02,  6.4459e-03,  7.1744e-02
    ];

% Differentiated yield
Cov_power_diff = [
    3.2963e-03,  1.2094e-03, -2.5042e-03,  6.8414e-03;
    1.2094e-03,  1.0981e-01, -5.7125e-04,  2.3381e-03;
    -2.5042e-03, -5.7125e-04,  1.3915e-02, -2.7301e-04;
    6.8414e-03,  2.3381e-03, -2.7301e-04,  3.8686e-02
    ];

% Normalised differentiated yield
Cov_power_norm = [
    2.9603e-03,  6.8345e-03,  8.7769e-07,  5.6113e-03;
    6.8345e-03,  7.7672e-02,  2.0806e-03,  1.3146e-03;
    8.7769e-07,  2.0806e-03,  8.2066e-03, -2.9670e-04;
    5.6113e-03,  1.3146e-03, -2.9670e-04,  3.1794e-02
    ];

% Linear model covariance (6x6): [D_i(0), D_i(Re), D_i(F), Ups(0), Ups(Re), Ups(F)]
% Cumulative yield
Cov_linear_cum = [
    2.7801e-02,  3.5096e-02, -6.9596e-03,  7.1573e-02,  1.0992e-02, -1.2661e-02;
    3.5096e-02,  6.8482e-01, -5.0531e-02, -4.8187e-02,  3.9209e-01, -2.6206e-02;
    -6.9596e-03, -5.0531e-02,  4.5693e-03, -7.7054e-03, -1.5915e-02,  3.3012e-03;
    7.1573e-02, -4.8187e-02, -7.7054e-03,  2.9254e-01,  6.5758e-02, -4.6300e-02;
    1.0992e-02,  3.9209e-01, -1.5915e-02,  6.5758e-02,  2.9506e+00, -1.3133e-01;
    -1.2661e-02, -2.6206e-02,  3.3012e-03, -4.6300e-02, -1.3133e-01,  1.2975e-02
    ];

% Differentiated yield
Cov_linear_diff = [
    2.2178e-02,  1.0828e-02, -4.3832e-03,  4.3992e-02,  4.4695e-03, -7.4634e-03;
    1.0828e-02,  4.3513e-01, -2.4832e-02,  3.0423e-03,  6.7633e-01, -3.3289e-02;
    -4.3832e-03, -2.4832e-02,  2.1282e-03, -7.3766e-03, -3.2298e-02,  2.8884e-03;
    4.3992e-02,  3.0423e-03, -7.3766e-03,  3.1429e-01, -6.2258e-02, -4.7085e-02;
    4.4695e-03,  6.7633e-01, -3.2298e-02, -6.2258e-02,  6.1032e+00, -2.4975e-01;
    -7.4634e-03, -3.3289e-02,  2.8884e-03, -4.7085e-02, -2.4975e-01,  1.8474e-02
    ];

% Normalised differentiated yield
Cov_linear_norm = [
    1.2717e-02,  2.8660e-02, -4.0477e-03,  2.6437e-02,  1.8916e-02, -5.8348e-03;
    2.8660e-02,  4.4243e-01, -3.1689e-02,  2.2351e-02,  6.5357e-01, -4.0257e-02;
    -4.0477e-03, -3.1689e-02,  2.6805e-03, -5.9892e-03, -3.8209e-02,  3.3147e-03;
    2.6437e-02,  2.2351e-02, -5.9892e-03,  2.0283e-01, -7.3693e-02, -3.1444e-02;
    1.8916e-02,  6.5357e-01, -3.8209e-02, -7.3693e-02,  5.9933e+00, -2.6958e-01;
    -5.8348e-03, -4.0257e-02,  3.3147e-03, -3.1444e-02, -2.6958e-01,  1.8713e-02
    ];

% Nominal parameter values
theta_power  = [1.222524; 4.308414; 0.972739; 3.428618];
theta_linear = [0.19; -8.188; 0.62; 3.158; 11.922; -0.6868];

%% Setup simulation infrastructure
m_total = 3.0;
before  = 0.04;
bed     = 0.92;

Time_in_sec     = (timeStep:timeStep:finalTime) * 60;
Time            = [0, Time_in_sec/60];
N_Time          = length(Time_in_sec);
timeStep_in_sec = timeStep * 60;

nstages = Parameters{1};
r       = Parameters{3};
epsi    = Parameters{4};
L       = Parameters{6};

nstagesbefore = 1:floor(before * nstages);
nstagesbed = nstagesbefore(end)+1 : nstagesbefore(end) + floor(bed * nstages);
nstagesafter = nstagesbed(end)+1 : nstages;

bed_mask = nan(nstages, 1);
bed_mask(nstagesbefore) = 0;
bed_mask(nstagesbed) = 1;
bed_mask(nstagesafter) = 0;

V_slice = (L/nstages) * pi * r^2;
V_bed = V_slice * numel(nstagesbed);

V_before_fluid = repmat(V_slice * numel(nstagesbefore) / numel(nstagesbefore), numel(nstagesbefore), 1);
V_bed_fluid = repmat(V_bed * (1-epsi) / numel(nstagesbed), numel(nstagesbed), 1);
V_after_fluid = repmat(V_slice * numel(nstagesafter) / numel(nstagesafter), numel(nstagesafter), 1);
V_fluid = [V_before_fluid; V_bed_fluid; V_after_fluid];

C0solid = m_total * 1e-3 / (V_bed * epsi);
Parameters{2} = C0solid;

m_fluid = zeros(1, nstages);
C0fluid = m_fluid * 1e-3 ./ V_fluid';

epsi_mask = epsi .* bed_mask;
one_minus_epsi_mask = 1 - epsi_mask;

Nx = 3 * nstages + 2;
Nu = 3 + numel(Parameters);

%% Yield case definitions
yield_cases = {'Cumulative', 'Differentiated', 'Normalised'};
Cov_power_cases  = {Cov_power_cum,  Cov_power_diff,  Cov_power_norm};
Cov_linear_cases = {Cov_linear_cum, Cov_linear_diff, Cov_linear_norm};

% Input vectors
% F is rescaled by ×1e-5 so its span [3.3, 6.7] is comparable to T's span [303, 313].
% L-BFGS span ratio: 10 : 3.4 ≈ 3  (vs 3×10^5 before scaling).
feedTemp  = opti.variable(N_Time)';                      % [303, 313] K  — direct physical
F_scaled  = opti.variable(N_Time)';                      % [3.3, 6.7]   — rescaled (×1e-5)
feedPress = P_param * ones(1, N_Time);                   % symbolic pressure — unchanged
feedFlow  = F_scaled * 1e-5;                             % [3.3e-5, 6.7e-5] kg/s — derived MX

T         = feedTemp(1);
P         = feedPress(1);
F         = feedFlow(1);

uu        = [feedTemp', feedPress', feedFlow'];

% Fluid properties
Z            = Compressibility(T, P, Parameters);
rho          = rhoPB_Comp(T, P, Z, Parameters);
enthalpy_rho = rho * SpecificEnthalpy(T, P, Z, rho, Parameters);

% Initial state
x0 = [C0fluid'; C0solid * bed_mask; enthalpy_rho * ones(nstages, 1); P; 0];

Parameters_sym          = MX(cell2mat(Parameters));
Parameters_sym(which_k) = k(1:numel(which_k));
U_base                  = [uu'; repmat(Parameters_sym, 1, N_Time)];

% Build integrators
f_power_nom       = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, ...
    'Power_model', epsi_mask, one_minus_epsi_mask);
F_power_nom       = buildIntegrator(f_power_nom, [Nx, Nu], timeStep_in_sec, 'cvodes');
F_accum_power_nom = F_power_nom.mapaccum('F_accum_power', N_Time);

f_linear_nom       = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, ...
    'Linear_model', epsi_mask, one_minus_epsi_mask);
F_linear_nom       = buildIntegrator(f_linear_nom, [Nx, Nu], timeStep_in_sec, 'cvodes');
F_accum_linear_nom = F_linear_nom.mapaccum('F_accum_linear', N_Time);

X_power_nom  = F_accum_power_nom(x0, U_base);
X_linear_nom = F_accum_linear_nom(x0, U_base);

% Extract cumulative yield at sample times
Y_cum_P_sym = [0, X_power_nom(Nx, :)];
Y_cum_P_sym = Y_cum_P_sym(N_Sample);
Y_cum_L_sym = [0, X_linear_nom(Nx, :)];
Y_cum_L_sym = Y_cum_L_sym(N_Sample);

% Differentiated yield (CasADi-compatible)
Y_diff_P_sym = Y_cum_P_sym(2:end) - Y_cum_P_sym(1:end-1);
Y_diff_L_sym = Y_cum_L_sym(2:end) - Y_cum_L_sym(1:end-1);

% Select symbolic output, data, and scaling based on case
Y_P_sym    = Y_cum_P_sym;
Y_L_sym    = Y_cum_L_sym;

% Jacobians
J_P_sym = jacobian(Y_P_sym, k1);
J_L_sym = jacobian(Y_L_sym, k2);

% Residuals
residuals = Y_P_sym - Y_L_sym;

% Predictive covariance and log-likelihood
n = numel(Y_P_sym);
I = MX.eye(n);

Sigma_r_P = sigma2_cases(1) * I + J_P_sym * Cov_power_cum  * J_P_sym';
Sigma_r_L = sigma2_cases(1) * I + J_L_sym * Cov_linear_cum * J_L_sym';

eps_reg = 1e-10;
Sigma_r_P = Sigma_r_P + eps_reg*I;
Sigma_r_L = Sigma_r_L + eps_reg*I;

j_1 = trace( Sigma_r_P * (Sigma_r_L\I) + Sigma_r_L * (Sigma_r_P\I) - 2*I );
j_2 = residuals * ((Sigma_r_P\I) + (Sigma_r_L\I)) * residuals';
%j   = j_1 + j_2;
j = residuals(end).^2 * 1e3;


% Box constraints — T in physical [K], F_scaled in ×1e-5 units [3.3, 6.7]
opti.subject_to(303 <= feedTemp <= 313);
opti.subject_to(3.3 <= F_scaled <= 6.7);

opti.set_value(k1, k1_val);
opti.set_value(k2, k2_val);

opti.minimize(-j);

%% Convert Opti problem to a CasADi Function for multi-start
% Decision variables passed as inputs become initial-guess arguments.
% Parameters (k1, k2) are baked in via opti.set_value above.
fprintf('Building solver function via opti.to_function ...\n');
solver_fn = opti.to_function('solver_fn', ...
    {feedTemp, F_scaled, P_param}, ...     % inputs:  physical T + scaled F initial guesses + pressure
    {feedTemp, F_scaled, j}, ...           % outputs: physical T + scaled F optimum + objective
    {'T_init', 'F_scaled_init', 'P_val'}, ...
    {'T_opt',  'F_scaled_opt',  'j_opt'});
fprintf('Solver function built successfully.\n');

%% Build a separate model-evaluation function (for post-processing plots)
% Full cumulative yield at ALL time steps (not just sample times)
Y_full_P_sym = [0, X_power_nom(Nx, :)];    % 1 x (N_Time+1)
Y_full_L_sym = [0, X_linear_nom(Nx, :)];   % 1 x (N_Time+1)

% Jacobians of full trajectory w.r.t. model parameters (for CI computation)
J_full_P_sym = jacobian(Y_full_P_sym, k1);  % (N_Time+1) x 4
J_full_L_sym = jacobian(Y_full_L_sym, k2);  % (N_Time+1) x 6

eval_fn = Function('eval_fn', ...
    {feedTemp, F_scaled, k1, k2, P_param}, ...
    {Y_full_P_sym, Y_full_L_sym, Y_P_sym, Y_L_sym, ...
     J_full_P_sym, J_full_L_sym, J_P_sym, J_L_sym}, ...
    {'T_in', 'F_scaled_in', 'k1_in', 'k2_in', 'P_in'}, ...
    {'Y_cum_P_full', 'Y_cum_L_full', 'Y_cum_P_sample', 'Y_cum_L_sample', ...
     'J_full_P', 'J_full_L', 'J_sample_P', 'J_sample_L'});

%% Pressure sweep configuration
N_press       = numel(pressures_bar);
best_j_all    = zeros(N_press, 1);           % best J value found at each pressure

%% Multi-start configuration
% Detect available parallel workers
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local');          % auto-detects available cores
end

% Suppress Java AWT initialisation on workers.
% CasADi can trigger an AWT call (for profiling/display) on worker threads,
% which fails because workers have no display.  Setting the headless property
% before the parfor prevents the ClassCastException.
parfevalOnAll(@() java.lang.System.setProperty('java.awt.headless','true'), 0);

fprintf('\n=== Multi-Start Configuration ===\n');
fprintf('Number of starts: %d\n', N_starts);
fprintf('Temperature bounds: [%.0f, %.0f] K\n', T_lb, T_ub);
fprintf('Flow rate bounds:   [%.2e, %.2e] kg/s\n', F_lb, F_ub);

%% Generate Latin Hypercube initial guesses
% Each row is a random trajectory: independent T(t), F(t) at every time step
rng(42);  % reproducibility
n_dim = 2 * N_Time;  % 40 temperatures + 40 flow rates

if exist('lhsdesign', 'file')
    X_lhs = lhsdesign(N_starts, n_dim);
else
    % Manual LHS fallback (no Statistics Toolbox required)
    X_lhs = zeros(N_starts, n_dim);
    for col = 1:n_dim
        perm = randperm(N_starts);
        X_lhs(:, col) = (perm' - rand(N_starts, 1)) / N_starts;
    end
end

% Rescale LHS samples from [0,1] to the decision-variable ranges.
feedTemp_all = T_lb + X_lhs(:, 1:N_Time) * (T_ub - T_lb);    % [303, 313] K
F_scaled_all = 3.3  + X_lhs(:, N_Time+1:end) * (6.7 - 3.3);  % [3.3, 6.7]  (×1e-5 = kg/s)

%% Pressure sweep loop
for p_idx = 1:N_press

P_val = pressures_bar(p_idx);
fprintf('\n%s\n', repmat('=', 1, 62));
fprintf('  Pressure: %d bar  (%d / %d)\n', P_val, p_idx, N_press);
fprintf('%s\n', repmat('=', 1, 62));

%% Parallel multi-start solve
fprintf('\nStarting %d parallel IPOPT solves ...\n', N_starts);
fprintf('%-9s %-8s %-16s %-10s  %s\n', 'Start', 'Status', 'J (Jeffreys)', 'Time [s]', 'ETA');
fprintf('%s\n', repmat('-', 1, 62));

% ---- DataQueue: each worker sends a struct ----
%  .idx       start index
%  .ok        true/false
%  .j_val     objective (or -Inf on failure)
%  .elapsed_s time taken by that start [s]
%  .msg       error message (on failure)
dq = parallel.pool.DataQueue;

% Mutable counters for the callback.
% containers.Map is a handle object, so the anonymous function captures
% a reference to the same map and can update it across calls.
n_workers = pool.NumWorkers;
ctr = containers.Map({'n_done','sum_time'}, {0, 0});

afterEach(dq, @(pkt) localProgress(pkt, ctr, n_workers, N_starts));

tic;

results_T = zeros(N_starts, N_Time);
results_F = zeros(N_starts, N_Time);
results_j = -Inf(N_starts, 1);
success   = false(N_starts, 1);

for i = 1:N_starts
    t_start = tic;
    pkt     = struct('idx', i, 'ok', false, 'j_val', -Inf, ...
                     'elapsed_s', 0, 'msg', '');
    try
        [T_opt, F_s_opt, j_opt] = solver_fn(feedTemp_all(i,:), F_scaled_all(i,:), P_val);
        results_T(i,:) = full(T_opt)';            % [K]    — already physical
        results_F(i,:) = full(F_s_opt)' * 1e-5;  % [kg/s] — unscale from [3.3,6.7]
        results_j(i)   = full(j_opt);
        success(i)     = true;
        pkt.ok         = true;
        pkt.j_val      = full(j_opt);
    catch ME
        success(i)     = false;
        results_j(i)   = -Inf;
        pkt.msg        = ME.message;
    end
    pkt.elapsed_s = toc(t_start);
    send(dq, pkt);
end

elapsed = toc;
fprintf('%s\n', repmat('-', 1, 62));
fprintf('All %d solves completed in %.1f s  (%.1f s/start wall-clock average).\n', ...
    N_starts, elapsed, elapsed / N_starts);

%% Result summary
fprintf('\n=== Multi-Start Results (%d starts) ===\n', N_starts);
fprintf('%5s | %10s | %14s\n', 'Start', 'Status', 'J (Jeffreys)');
fprintf('%s\n', repmat('-', 1, 37));
for i = 1:N_starts
    if success(i)
        status_str = 'OK';
    else
        status_str = 'FAIL';
    end
    fprintf('%5d | %10s | %14.6e\n', i, status_str, results_j(i));
end

%% Select best result
if any(success)
    obj_masked = results_j;
    obj_masked(~success) = -Inf;
    [best_j, best_idx] = max(obj_masked);
else
    [best_j, best_idx] = max(results_j);
    warning('No solver succeeded. Using best debug value.');
end

fprintf('\nBest start: #%d with J = %.6e  (%d/%d succeeded)\n', ...
    best_idx, best_j, sum(success), N_starts);

best_j_all(p_idx) = best_j;
K_out = [results_T(best_idx,:); results_F(best_idx,:)];

%% Plot best trajectory
%{
figure('Name', 'Optimal Trajectory (Best Start)', 'NumberTitle', 'off');

subplot(2,1,1)
stairs(Time, [K_out(1,:), K_out(1,end)] - 273, 'LineWidth', 2)
xlabel('Time [min]')
ylabel('T [C]')
title(sprintf('Best start #%d — %d bar:  J = %.4e', best_idx, P_val, best_j))
grid on

subplot(2,1,2)
stairs(Time, [K_out(2,:), K_out(2,end)] * 1e5, 'LineWidth', 2)
xlabel('Time [min]')
ylabel('F [kg/s  x 1e-5]')
grid on
%}

%% Plot all successful trajectories
figure('Name', sprintf('All Successful Trajectories — %d bar', P_val), 'NumberTitle', 'off');
colors = lines(N_starts);
colors(best_idx, :) = [0, 0, 0];

subplot(2,1,1); hold on;
for i = 1:N_starts
    if success(i)
        lw = 1; if i == best_idx, lw = 5; end
        stairs(Time, [results_T(i,:), results_T(i,end)] - 273, ...
            'Color', colors(i,:), 'LineWidth', lw)
    end
end
ylabel('T [C]'); xlabel('Time [min]'); grid on;
title('All successful starts (best = thick)')

subplot(2,1,2); hold on;
for i = 1:N_starts
    if success(i)
        lw = 1; if i == best_idx, lw = 5; end
        stairs(Time, [results_F(i,:), results_F(i,end)] * 1e5, ...
            'Color', colors(i,:), 'LineWidth', lw)
    end
end
ylabel('F [kg/s  x 1e-5]'); xlabel('Time [min]'); grid on;

%% Evaluate model outputs at the best trajectory
fprintf('\nEvaluating model outputs at best trajectory ...\n');
best_T = K_out(1,:);
best_F = K_out(2,:);

% Scale best_F from [kg/s] back to the F_scaled domain [3.3, 6.7] for eval_fn
best_F_scaled = best_F * 1e5;

[Y_cum_P_full, Y_cum_L_full, Y_cum_P_sample, Y_cum_L_sample, ...
 JJ_full_P, JJ_full_L, JJ_sample_P, JJ_sample_L] = ...
    eval_fn(best_T, best_F_scaled, k1_val, k2_val, P_val);

Y_cum_P_full   = full(Y_cum_P_full);
Y_cum_L_full   = full(Y_cum_L_full);
Y_cum_P_sample = full(Y_cum_P_sample);
Y_cum_L_sample = full(Y_cum_L_sample);
JJ_full_P      = full(JJ_full_P);      % (N_Time+1) x 4
JJ_full_L      = full(JJ_full_L);      % (N_Time+1) x 6
JJ_sample_P    = full(JJ_sample_P);    % N_sample x 4
JJ_sample_L    = full(JJ_sample_L);    % N_sample x 6

% Differentiated yield at sample times
Y_diff_P = Y_cum_P_sample(2:end) - Y_cum_P_sample(1:end-1);
Y_diff_L = Y_cum_L_sample(2:end) - Y_cum_L_sample(1:end-1);

% Sample times for plotting
SAMPLE_times = SAMPLE;
SAMPLE_diff_times = (SAMPLE(1:end-1) + SAMPLE(2:end)) / 2;  % midpoints

%% Compute 95% confidence intervals
% Predictive variance = parameter uncertainty + measurement noise:
%   Var(y_i(t)) = J_i(t,:) * Cov_theta_i * J_i(t,:)' + sigma^2
% CI half-width = 1.96 * sqrt(Var)
sigma2 = sigma2_cases(1);  % empirical measurement variance (cumulative)
z_95   = 1.96;

% --- Full trajectory CI (for smooth shaded bands) ---
% Power model: diag( J_full * Cov_power * J_full' ) + sigma2
var_full_P = sum((JJ_full_P * Cov_power_cum) .* JJ_full_P, 2)' + sigma2;
var_full_L = sum((JJ_full_L * Cov_linear_cum) .* JJ_full_L, 2)' + sigma2;

CI_full_P = z_95 * sqrt(var_full_P);  % 1 x (N_Time+1)
CI_full_L = z_95 * sqrt(var_full_L);

% --- Sample-time CI ---
var_sample_P = sum((JJ_sample_P * Cov_power_cum) .* JJ_sample_P, 2)' + sigma2;
var_sample_L = sum((JJ_sample_L * Cov_linear_cum) .* JJ_sample_L, 2)' + sigma2;

CI_sample_P = z_95 * sqrt(var_sample_P);
CI_sample_L = z_95 * sqrt(var_sample_L);

% --- Differentiated yield CI ---
% diff(Y) = Y(k+1) - Y(k), so J_diff(k,:) = J(k+1,:) - J(k,:)
% Cov(diff_Y) = J_diff * Cov_theta_diff * J_diff' + sigma2_diff
% Uses differentiated-yield parameter covariances (estimated from diff residuals)
JJ_diff_P = JJ_sample_P(2:end,:) - JJ_sample_P(1:end-1,:);
JJ_diff_L = JJ_sample_L(2:end,:) - JJ_sample_L(1:end-1,:);

sigma2_diff = sigma2_cases(2);  % empirical variance for differentiated yield
var_diff_P = sum((JJ_diff_P * Cov_power_diff) .* JJ_diff_P, 2)' + sigma2_diff;
var_diff_L = sum((JJ_diff_L * Cov_linear_diff) .* JJ_diff_L, 2)' + sigma2_diff;

CI_diff_P = z_95 * sqrt(var_diff_P);
CI_diff_L = z_95 * sqrt(var_diff_L);

fprintf('CI computed: full trajectory (%d pts), sample (%d pts), diff (%d pts)\n', ...
    numel(CI_full_P), numel(CI_sample_P), numel(CI_diff_P));

%% Plot model outputs: Cumulative yield with CI
figure('Name', sprintf('Model Outputs - Cumulative Yield — %d bar', P_val), 'NumberTitle', 'off');

% Shaded CI bands (plot first so they appear behind lines)
fill([Time, fliplr(Time)], ...
     [Y_cum_P_full + CI_full_P, fliplr(Y_cum_P_full - CI_full_P)], ...
     'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
hold on;
fill([Time, fliplr(Time)], ...
     [Y_cum_L_full + CI_full_L, fliplr(Y_cum_L_full - CI_full_L)], ...
     'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% Model predictions
plot(Time, Y_cum_P_full, 'b-', 'LineWidth', 2, 'DisplayName', 'Power model');
plot(Time, Y_cum_L_full, 'r--', 'LineWidth', 2, 'DisplayName', 'Linear model');

% Sample markers with error bars
errorbar(SAMPLE_times, Y_cum_P_sample, CI_sample_P, 'bo', ...
    'MarkerSize', 5, 'MarkerFaceColor', 'b', 'LineWidth', 1, ...
    'HandleVisibility', 'off');
errorbar(SAMPLE_times, Y_cum_L_sample, CI_sample_L, 'rs', ...
    'MarkerSize', 5, 'MarkerFaceColor', 'r', 'LineWidth', 1, ...
    'HandleVisibility', 'off');

xlabel('Time [min]')
ylabel('Cumulative yield [g]')
title(sprintf('Model predictions — %d bar  (J = %.4e)', P_val, best_j))
legend('Location', 'southeast')
grid on

%% Plot model outputs: Differentiated yield with CI
figure('Name', sprintf('Model Outputs - Differentiated Yield — %d bar', P_val), 'NumberTitle', 'off');

bar_width = 0.35;
bar_x = 1:numel(Y_diff_P);

bar(bar_x - bar_width/2, Y_diff_P, bar_width, 'FaceColor', [0.2 0.4 0.8], ...
    'DisplayName', 'Power model'); hold on;
bar(bar_x + bar_width/2, Y_diff_L, bar_width, 'FaceColor', [0.8 0.2 0.2], ...
    'DisplayName', 'Linear model');

% Error bars on top of bars
errorbar(bar_x - bar_width/2, Y_diff_P, CI_diff_P, 'k.', ...
    'LineWidth', 1.2, 'HandleVisibility', 'off');
errorbar(bar_x + bar_width/2, Y_diff_L, CI_diff_L, 'k.', ...
    'LineWidth', 1.2, 'HandleVisibility', 'off');

xlabel('Sample interval')
ylabel('Differentiated yield [g]')
title(sprintf('Interval-wise yield — %d bar', P_val))
legend('Location', 'northeast')
grid on

%% Plot model outputs: Cumulative yield + operating conditions (combined)
figure('Name', sprintf('Model Outputs and Operating Conditions — %d bar', P_val), 'NumberTitle', 'off');

% Top: cumulative yield with CI
subplot(3,1,1)
fill([Time, fliplr(Time)], ...
     [Y_cum_P_full + CI_full_P, fliplr(Y_cum_P_full - CI_full_P)], ...
     'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
hold on;
fill([Time, fliplr(Time)], ...
     [Y_cum_L_full + CI_full_L, fliplr(Y_cum_L_full - CI_full_L)], ...
     'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(Time, Y_cum_P_full, 'b-', 'LineWidth', 2, 'DisplayName', 'Power');
plot(Time, Y_cum_L_full, 'r--', 'LineWidth', 2, 'DisplayName', 'Linear');
ylabel('Cumulative yield [g]')
legend('Location', 'southeast')
title(sprintf('Best start #%d — %d bar:  J = %.4e', best_idx, P_val, best_j))
grid on

% Middle: temperature
subplot(3,1,2)
stairs(Time, [best_T, best_T(end)] - 273, 'k-', 'LineWidth', 2)
ylabel('T [C]')
grid on

% Bottom: flow rate
subplot(3,1,3)
stairs(Time, [best_F, best_F(end)] * 1e5, 'k-', 'LineWidth', 2)
xlabel('Time [min]')
ylabel('F [kg/s x 1e-5]')
grid on

%% Save results for this pressure
save_name = sprintf('multistart_results_%dbar.mat', P_val);
save(save_name, ...
    'results_T', 'results_F', 'results_j', 'success', ...
    'best_idx', 'best_j', 'K_out', 'P_val', ...
    'feedTemp_all', 'F_scaled_all', 'T_lb', 'T_ub', 'F_lb', 'F_ub', 'N_starts', 'nlp_opts', ...
    'Y_cum_P_full', 'Y_cum_L_full', 'Y_cum_P_sample', 'Y_cum_L_sample', ...
    'Y_diff_P', 'Y_diff_L', ...
    'CI_full_P', 'CI_full_L', 'CI_sample_P', 'CI_sample_L', ...
    'CI_diff_P', 'CI_diff_L', ...
    'var_full_P', 'var_full_L');
fprintf('Results saved to %s\n', save_name);

end  % for p_idx = 1:N_press

%% Summary: best Jeffreys divergence vs extraction pressure
fprintf('\n=== Pressure Sweep Summary ===\n');
fprintf('%12s | %14s\n', 'Pressure [bar]', 'J_best');
fprintf('%s\n', repmat('-', 1, 30));
for p_idx = 1:N_press
    fprintf('%14d | %14.6e\n', pressures_bar(p_idx), best_j_all(p_idx));
end

figure('Name', 'Jeffreys Divergence vs Pressure', 'NumberTitle', 'off');
plot(pressures_bar, best_j_all, 'ko-', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', 'k');
xlabel('Extraction pressure [bar]')
ylabel('J_{best} (Jeffreys divergence)')
title('Optimal model-discrimination criterion vs extraction pressure')
grid on

%% -----------------------------------------------------------------------
%  Local helper functions  (must appear after all script statements)
%  -----------------------------------------------------------------------
function localProgress(pkt, ctr, n_workers, N_starts)
% Called by afterEach(dq, ...) on the client thread every time a worker
% sends a progress packet.  ctr is a containers.Map (handle object), so
% mutations here are visible across calls.

    % --- Update mutable counters ---
    ctr('n_done')   = ctr('n_done')   + 1;
    ctr('sum_time') = ctr('sum_time') + pkt.elapsed_s;

    n_done   = ctr('n_done');
    sum_time = ctr('sum_time');

    % --- Estimate remaining wall-clock time ---
    mean_t      = sum_time / n_done;          % avg serial time per start [s]
    n_remaining = N_starts - n_done;
    eta_s       = (n_remaining / n_workers) * mean_t;  % parallel ETA [s]

    % --- Format ETA string ---
    if n_remaining == 0
        eta_str = 'done';
    elseif eta_s < 60
        eta_str = sprintf('~%.0f s',   eta_s);
    elseif eta_s < 3600
        eta_str = sprintf('~%.1f min', eta_s / 60);
    else
        eta_str = sprintf('~%.1f h',   eta_s / 3600);
    end

    % --- Format status / objective ---
    if pkt.ok
        status_str = 'OK';
        j_str      = sprintf('%+.6e', pkt.j_val);
    else
        status_str = 'FAIL';
        j_str      = sprintf('%-16s', '-Inf');
    end

    fprintf('Start %2d  %6s  %16s  %6.1f s   ETA %-12s  (%d/%d done)\n', ...
        pkt.idx, status_str, j_str, pkt.elapsed_s, eta_str, n_done, N_starts);

    if ~pkt.ok && ~isempty(pkt.msg)
        fprintf('          >> %s\n', pkt.msg);
    end
end





















