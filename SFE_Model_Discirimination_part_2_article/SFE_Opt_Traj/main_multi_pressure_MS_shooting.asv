%% Initialization
startup;
%initParPool

fprintf('=============================================================================\n');
fprintf('   Optimal trajectory for model discrimination (parfor multistart)\n');
fprintf('=============================================================================\n\n');

%% Run configuration
N_seeds      = 1;
N_workers    = 6;
n_init_knots = 10;

sim_dt    = 10;    % [min]  ODE integration / simulation step
TF_dt     = 10;    % [min]  T and F change every TF_dt min
P_dt      = 20;    % [min]  P changes every P_dt min
finalTime = 600;   % total experiment duration [min]

%casadi_path  = '/scratch/work/sliczno1/SFE-Model-Discrimination/SFE_Model_Discirimination_part_2_article/SFE_Opt_Traj/casadi_folder';
casadi_path  = 'C:\Dev\casadi-3.6.3-windows64-matlab2018b';

% Directory for incremental per-seed saves
%save_dir = 'seed_results';
%if ~exist(save_dir, 'dir')
%    mkdir(save_dir);
%end

%% Optimizer Settings
nlp_opts = struct;
nlp_opts.ipopt.max_iter              = 15;
nlp_opts.ipopt.tol                   = 1e-7;
nlp_opts.ipopt.acceptable_tol        = 1e-5;
nlp_opts.ipopt.acceptable_iter       = 10;
nlp_opts.ipopt.hessian_approximation      = 'limited-memory';
nlp_opts.ipopt.nlp_scaling_max_gradient   = 10;   % scale NLP if initial gradient exceeds this
nlp_opts.ipopt.limited_memory_init_val    = 0.01; % small initial H^{-1} → conservative first step
nlp_opts.ipopt.print_level                = 5;
nlp_opts.print_time                       = 0;
nlp_opts.ipopt.mu_strategy = 'adaptive';
%nlp_opts.ipopt.obj_scaling_factor = 0.01;
nlp_opts.ipopt.bound_push = 0.1;
%nlp_opts.ad_weight     = 0;   % 0 = pure forward AD, 1 = pure reverse AD (default)
%nlp_opts.ad_weight_sp  = 0;   % same for sparsity pattern

fprintf('=== Run Configuration ===\n');
fprintf('Seeds: %d | Workers: %d | Knots: %d | Max iter: %d\n\n', ...
    N_seeds, N_workers, n_init_knots, nlp_opts.ipopt.max_iter);
%fprintf('Incremental save directory: %s\n\n', save_dir);

%% Load parameters and data
addpath(casadi_path);

Parameters_table = readtable('Parameters.csv');
Parameters       = num2cell(Parameters_table{:,3});
LabResults       = readmatrix('dataset_2.xlsx');

which_k = (0:9) + 44;
k1_val  = cell2mat(Parameters((0:3) + 44));
k2_val  = cell2mat(Parameters((4:9) + 44));

%% Time grid and sample matching
assert(mod(TF_dt, sim_dt) == 0, 'TF_dt must be a multiple of sim_dt');
assert(mod(P_dt,  sim_dt) == 0, 'P_dt must be a multiple of sim_dt');
assert(mod(finalTime, TF_dt) == 0, 'finalTime must be divisible by TF_dt');
assert(mod(finalTime, P_dt)  == 0, 'finalTime must be divisible by P_dt');

Time_in_sec     = (sim_dt:sim_dt:finalTime) * 60;
Time            = [0, Time_in_sec/60];
N_sim           = length(Time_in_sec);      % total simulation steps
timeStep_in_sec = sim_dt * 60;

% T/F knots: one decision per TF_dt interval, expanded to simulation grid via ZOH
N_TF          = round(finalTime / TF_dt);
TF_steps_per  = round(TF_dt / sim_dt);
TF_knot_index = ceil((1:N_sim) / TF_steps_per);   % 1 x N_sim

SAMPLE = LabResults(6:19, 1);
SAMPLE = SAMPLE(2:end);

sample_tol = 1e-3;
N_Sample   = zeros(size(SAMPLE));
for ii = 1:numel(SAMPLE)
    [delta, idx] = min(abs(Time - SAMPLE(ii)));
    if delta > sample_tol
        error('Sample time mismatch at index %d (delta=%.3g min)', ii, delta);
    end
    N_Sample(ii) = idx;
end

%% Pressure switching grid
N_P_knots    = round(finalTime / P_dt);
P_steps_per  = round(P_dt / sim_dt);
P_knot_index = ceil((1:N_sim) / P_steps_per);   % 1 x N_sim

fprintf('Discretisation: sim_dt=%d min | TF_dt=%d min (%d knots) | P_dt=%d min (%d knots)\n\n', ...
    sim_dt, TF_dt, N_TF, P_dt, N_P_knots);

% Fix 4: Tune L-BFGS history to effective DOF (controls only after equality constraints)
nlp_opts.ipopt.limited_memory_max_history = N_TF + N_TF + N_P_knots;
fprintf('L-BFGS history = %d (= %d T-knots + %d F-knots + %d P-knots)\n\n', ...
    N_TF + N_TF + N_P_knots, N_TF, N_TF, N_P_knots);

% Fix 5: Use TF_dt-granularity shooting nodes if all sample times align with TF_dt.
% This reduces state variables from 2*Nx*N_sim to 2*Nx*N_TF (3x fewer when TF_dt=3*sim_dt).
use_TF_shooting = all(mod(SAMPLE, TF_dt) < sample_tol);
if use_TF_shooting
    N_shot        = N_TF;
    shot_dt_sec   = TF_dt * 60;
    TF_P_knot_idx = ceil((1:N_TF) / round(P_dt / TF_dt));   % maps each TF step to a P knot
    Time_TF       = [0, TF_dt : TF_dt : finalTime];
    N_Sample_shot = zeros(size(SAMPLE));
    for ii_s = 1:numel(SAMPLE)
        [~, N_Sample_shot(ii_s)] = min(abs(Time_TF - SAMPLE(ii_s)));
    end
    fprintf('Fix 5: coarse shooting — %d TF-nodes (was %d sim-nodes, %.0fx reduction)\n\n', ...
        N_TF, N_sim, N_sim / N_TF);
else
    N_shot        = N_sim;
    shot_dt_sec   = timeStep_in_sec;
    N_Sample_shot = N_Sample;
    fprintf('Fix 5: fine shooting — %d sim-nodes (sample times not TF_dt-aligned)\n\n', N_sim);
end

% Fix 2: thread-parallel map (uses std::thread; 'openmp' requires WITH_OPENMP=ON)
n_threads = N_workers;

%% Covariance matrices
Cov_power_cum = [
    1.0035e-02,  1.1795e-02,  1.8268e-03,  2.5611e-02;
    1.1795e-02,  5.6469e-02,  3.1182e-03,  2.8266e-02;
    1.8268e-03,  3.1182e-03,  5.7241e-03,  6.4459e-03;
    2.5611e-02,  2.8266e-02,  6.4459e-03,  7.1744e-02
    ];

Cov_power_diff = [
    3.2963e-03,  1.2094e-03, -2.5042e-03,  6.8414e-03;
    1.2094e-03,  1.0981e-01, -5.7125e-04,  2.3381e-03;
    -2.5042e-03, -5.7125e-04,  1.3915e-02, -2.7301e-04;
    6.8414e-03,  2.3381e-03, -2.7301e-04,  3.8686e-02
    ];

Cov_power_norm = [
    2.9603e-03,  6.8345e-03,  8.7769e-07,  5.6113e-03;
    6.8345e-03,  7.7672e-02,  2.0806e-03,  1.3146e-03;
    8.7769e-07,  2.0806e-03,  8.2066e-03, -2.9670e-04;
    5.6113e-03,  1.3146e-03, -2.9670e-04,  3.1794e-02
    ];

Cov_linear_cum = [
    2.7801e-02,  3.5096e-02, -6.9596e-03,  7.1573e-02,  1.0992e-02, -1.2661e-02;
    3.5096e-02,  6.8482e-01, -5.0531e-02, -4.8187e-02,  3.9209e-01, -2.6206e-02;
    -6.9596e-03, -5.0531e-02,  4.5693e-03, -7.7054e-03, -1.5915e-02,  3.3012e-03;
    7.1573e-02, -4.8187e-02, -7.7054e-03,  2.9254e-01,  6.5758e-02, -4.6300e-02;
    1.0992e-02,  3.9209e-01, -1.5915e-02,  6.5758e-02,  2.9506e+00, -1.3133e-01;
    -1.2661e-02, -2.6206e-02,  3.3012e-03, -4.6300e-02, -1.3133e-01,  1.2975e-02
    ];

Cov_linear_diff = [
    2.2178e-02,  1.0828e-02, -4.3832e-03,  4.3992e-02,  4.4695e-03, -7.4634e-03;
    1.0828e-02,  4.3513e-01, -2.4832e-02,  3.0423e-03,  6.7633e-01, -3.3289e-02;
    -4.3832e-03, -2.4832e-02,  2.1282e-03, -7.3766e-03, -3.2298e-02,  2.8884e-03;
    4.3992e-02,  3.0423e-03, -7.3766e-03,  3.1429e-01, -6.2258e-02, -4.7085e-02;
    4.4695e-03,  6.7633e-01, -3.2298e-02, -6.2258e-02,  6.1032e+00, -2.4975e-01;
    -7.4634e-03, -3.3289e-02,  2.8884e-03, -4.7085e-02, -2.4975e-01,  1.8474e-02
    ];

Cov_linear_norm = [
    1.2717e-02,  2.8660e-02, -4.0477e-03,  2.6437e-02,  1.8916e-02, -5.8348e-03;
    2.8660e-02,  4.4243e-01, -3.1689e-02,  2.2351e-02,  6.5357e-01, -4.0257e-02;
    -4.0477e-03, -3.1689e-02,  2.6805e-03, -5.9892e-03, -3.8209e-02,  3.3147e-03;
    2.6437e-02,  2.2351e-02, -5.9892e-03,  2.0283e-01, -7.3693e-02, -3.1444e-02;
    1.8916e-02,  6.5357e-01, -3.8209e-02, -7.3693e-02,  5.9933e+00, -2.6958e-01;
    -5.8348e-03, -4.0257e-02,  3.3147e-03, -3.1444e-02, -2.6958e-01,  1.8713e-02
    ];

sigma2_cases = [2.45e-2, 1.386e-3, 1.007e-2];

%% Simulation geometry
m_total = 3.0;
before  = 0.04;
bed     = 0.92;

nstages = Parameters{1};
r       = Parameters{3};
epsi    = Parameters{4};
L       = Parameters{6};

nstagesbefore = 1:floor(before * nstages);
nstagesbed    = nstagesbefore(end)+1 : nstagesbefore(end) + floor(bed * nstages);
nstagesafter  = nstagesbed(end)+1 : nstages;

bed_mask = nan(nstages, 1);
bed_mask(nstagesbefore) = 0;
bed_mask(nstagesbed)    = 1;
bed_mask(nstagesafter)  = 0;

V_slice = (L/nstages) * pi * r^2;
V_bed   = V_slice * numel(nstagesbed);

V_before_fluid = repmat(V_slice * numel(nstagesbefore) / numel(nstagesbefore), numel(nstagesbefore), 1);
V_bed_fluid    = repmat(V_bed * (1-epsi) / numel(nstagesbed), numel(nstagesbed), 1);
V_after_fluid  = repmat(V_slice * numel(nstagesafter) / numel(nstagesafter), numel(nstagesafter), 1);
V_fluid        = [V_before_fluid; V_bed_fluid; V_after_fluid];

C0solid = m_total * 1e-3 / (V_bed * epsi);
Parameters{2} = C0solid;

m_fluid = zeros(1, nstages);
C0fluid = m_fluid * 1e-3 ./ V_fluid';

epsi_mask           = epsi .* bed_mask;
one_minus_epsi_mask = 1 - epsi_mask;

Nx = 3 * nstages + 2;
Nu = 3 + numel(Parameters);

%% Bounds and normalization
% Temperature
T_min  = 303;     T_max  = 313;
T_mid  = 0.5 * (T_min + T_max);
T_half = 0.5 * (T_max - T_min);

% Flow
F_min  = 3.3e-5;  F_max  = 6.7e-5;
F_mid  = 0.5 * (F_min + F_max);
F_half = 0.5 * (F_max - F_min);

% Pressure
P_min  = 100;     P_max  = 200;
P_mid  = 0.5 * (P_min + P_max);
P_half = 0.5 * (P_max - P_min);

fprintf('Decision variable bounds:\n');
fprintf('  Temperature : [%.1f, %.1f] K\n',    T_min, T_max);
fprintf('  Flow rate   : [%.2e, %.2e] m3/s\n', F_min, F_max);
fprintf('  Pressure    : [%.1f, %.1f] bar  (%d knots, ZOH every %d min)\n\n', ...
    P_min, P_max, N_P_knots, P_dt);

%% Pre-generate all initial guesses via Latin Hypercube Sampling
%
% Temperature and flow: N_seeds x n_init_knots each  (free at every step)
% Pressure:            N_seeds x N_P_knots           (one value per 30-min interval)
%
% All three inputs are sampled jointly in a single LHS design so that the
% combined input space is covered uniformly across seeds.
%
% Column layout (unit [0,1] before scaling):
%   cols 1               : n_init_knots    -> temperature knots
%   cols n+1             : 2*n_init_knots  -> flow knots
%   cols 2*n+1           : 2*n+N_P_knots  -> pressure interval values

n_lhs_cols = 2 * n_init_knots + N_P_knots;

rng(0);   % master seed — governs LHS values; per-seed rng governs knot positions
lhs = lhsdesign(N_seeds, n_lhs_cols, 'criterion', 'maximin', 'iterations', 20);

z0_mat         = zeros(2 * N_TF + N_P_knots, N_seeds);
feedTemp0_all  = zeros(N_TF,    N_seeds);
feedFlow0_all  = zeros(N_TF,    N_seeds);
feedPress0_all = zeros(N_P_knots, N_seeds);   % one value per 30-min interval

for s = 1:N_seeds
    rng(s);   % governs knot *positions* for T and F only
    interior_pool = 2:(N_TF - 1);
    n_interior    = n_init_knots - 2;
    pick          = randperm(numel(interior_pool), n_interior);
    interior_idx  = sort(interior_pool(pick));
    init_knot_idx = [1, interior_idx, N_TF];

    % Scale LHS columns to physical ranges
    temp_knots  = T_min + (T_max - T_min) * lhs(s,                  1 :   n_init_knots);
    flow_knots  = F_min + (F_max - F_min) * lhs(s,   n_init_knots + 1 : 2*n_init_knots);
    %press_vals  = P_min + (P_max - P_min) * lhs(s, 2*n_init_knots + 1 : 2*n_init_knots + N_P_knots);
    %press_vals   = P_min + (P_max - P_min) * rand();
    press_vals = 130;

    % Interpolate T and F knots onto the T/F decision grid
    feedTemp_0  = interp1(init_knot_idx, temp_knots, 1:N_TF, 'linear');
    feedFlow_0  = interp1(init_knot_idx, flow_knots, 1:N_TF, 'linear');

    % Pressure initial guess: one scalar per interval (LHS-sampled, no interpolation needed)
    feedPress_0 = press_vals * ones(1, N_P_knots);   % 1 x N_P_knots  (from LHS over [P_min, P_max])

    % Store normalized versions
    z0_mat(:, s) = [(feedTemp_0  - T_mid)  / T_half, ...
                    (feedFlow_0  - F_mid)  / F_half, ...
                    (feedPress_0 - P_mid)  / P_half]';

    feedTemp0_all(:,  s) = feedTemp_0(:);
    feedFlow0_all(:,  s) = feedFlow_0(:);
    feedPress0_all(:, s) = feedPress_0(:);
end

fprintf('Initial guess summary across %d seeds (LHS):\n', N_seeds);
fprintf('  Temp  mean %.2f K,   std %.2f K\n',    mean(feedTemp0_all(:)),  std(feedTemp0_all(:)));
fprintf('  Flow  mean %.2e,     std %.2e\n',       mean(feedFlow0_all(:)),  std(feedFlow0_all(:)));
fprintf('  Press mean %.2f bar, std %.2f bar\n\n', mean(feedPress0_all(:)), std(feedPress0_all(:)));

%% Parallel pool
pool = gcp('nocreate');
if isempty(pool)
    parpool('local', N_workers);
end
parfevalOnAll(@() addpath(casadi_path), 0);

%% Parallel multistart
clear local_save_progress;
dq = parallel.pool.DataQueue;
afterEach(dq, @(msg) local_save_progress(msg, N_seeds, save_dir));

t_par = tic;
fprintf('Solving %d seeds on %d workers...\n', N_seeds, N_workers);
results = cell(1, N_seeds);

for s = 1:N_seeds
    t_seed_start = tic;

    % Default failure result
    r = struct('seed', s, 'success', false, 'j_initial', NaN, 'j', NaN, ...
        'j_1', NaN, 'j_2', NaN, ...
        'feedTemp',    NaN(1, N_TF),     ...
        'feedFlow',    NaN(1, N_TF),     ...
        'feedPress',   NaN(1, N_sim),    ...   % full time-expanded profile
        'feedPressKnots', NaN(1, N_P_knots), ... % the N_P_knots optimised values
        'feedTemp0',   feedTemp0_all(:, s)',  ...
        'feedFlow0',   feedFlow0_all(:, s)',  ...
        'feedPress0',  feedPress0_all(:, s)');

    try
        import casadi.*

        %% --- Build CasADi problem ---
        opti = casadi.Opti();
        opti.solver('ipopt', nlp_opts);

        k1 = opti.parameter(4);
        k2 = opti.parameter(6);
        k  = [k1; k2];

        % --- Temperature and flow: one decision per TF_dt interval ---
        zFeedTemp = opti.variable(1, N_TF);
        zFeedFlow = opti.variable(1, N_TF);

        feedTemp = T_mid + T_half * zFeedTemp;   % 1 x N_TF
        feedFlow = F_mid + F_half * zFeedFlow;   % 1 x N_TF

        % Expand to simulation grid via ZOH index map
        feedTempSim = feedTemp(TF_knot_index);   % 1 x N_sim
        feedFlowSim = feedFlow(TF_knot_index);   % 1 x N_sim

        % --- Pressure: one decision per P_dt interval ---
        zFeedPressKnots = opti.variable(1, N_P_knots);

        feedPressKnots = P_mid + P_half * zFeedPressKnots;   % 1 x N_P_knots physical
        feedPressSim   = feedPressKnots(P_knot_index);        % 1 x N_sim  via index broadcast

        % --- Initial condition uses values at t=0 (first knot of each) ---
        T  = feedTempSim(1);
        P  = feedPressSim(1);

        Z            = Compressibility(T, P, Parameters);
        rho          = rhoPB_Comp(T, P, Z, Parameters);
        enthalpy_rho = rho * SpecificEnthalpy(T, P, Z, rho, Parameters);

        % enthalpy_rho stored scaled by 1/1e4 to match modelSFE ENTHALPY_SCALE=1e4
        x0 = [C0fluid'; C0solid * bed_mask; (enthalpy_rho / 1e4) * ones(nstages, 1); P; 0];

        Parameters_sym          = MX(cell2mat(Parameters));
        Parameters_sym(which_k) = k(1:numel(which_k));

        % Fix 5: build U_base at shooting-node granularity (TF_dt or sim_dt)
        if use_TF_shooting
            feedPressTF = feedPressKnots(TF_P_knot_idx);          % 1 x N_TF
            uu_shot     = [feedTemp', feedPressTF', feedFlow'];    % N_TF x 3
        else
            uu_shot = [feedTempSim', feedPressSim', feedFlowSim']; % N_sim x 3
        end
        U_base = [uu_shot'; repmat(Parameters_sym, 1, N_shot)];   % Nu x N_shot

        % Fix 2: integrators at shooting horizon; thread-parallel map (std::thread)
        f_power_nom  = @(x, u) modelSFE(x, u, bed_mask, shot_dt_sec, ...
            'Power_model', epsi_mask, one_minus_epsi_mask);
        F_power_nom  = buildIntegrator(f_power_nom, [Nx, Nu], shot_dt_sec, 'cvodes');
        F_power_map  = F_power_nom.map(N_shot, 'thread', n_threads);

        f_linear_nom = @(x, u) modelSFE(x, u, bed_mask, shot_dt_sec, ...
            'Linear_model', epsi_mask, one_minus_epsi_mask);
        F_linear_nom = buildIntegrator(f_linear_nom, [Nx, Nu], shot_dt_sec, 'cvodes');
        F_linear_map = F_linear_nom.map(N_shot, 'thread', n_threads);

        % --- Multiple-shooting state decision variables ---
        % X_P(:,k) / X_L(:,k) = state at END of shooting step k (k = 1..N_shot)
        X_P = opti.variable(Nx, N_shot);
        X_L = opti.variable(Nx, N_shot);

        % Continuity constraints: X(:,k) = F( X(:,k-1), U(:,k) )
        % For k=1 the "previous state" is x0 (depends on first control knot).
        % Stack previous states: [x0 | X(:, 1:end-1)]  →  Nx x N_shot
        X_P_prev = [x0, X_P(:, 1:end-1)];
        X_L_prev = [x0, X_L(:, 1:end-1)];
        opti.subject_to(X_P == F_power_map( X_P_prev, U_base));
        opti.subject_to(X_L == F_linear_map(X_L_prev, U_base));

        % Expose as trajectory (same interface as mapaccum output)
        X_power_nom  = X_P;
        X_linear_nom = X_L;

        Y_cum_P_sym = [0, X_power_nom(Nx, :)];
        Y_cum_P_sym = Y_cum_P_sym(N_Sample_shot);
        Y_cum_L_sym = [0, X_linear_nom(Nx, :)];
        Y_cum_L_sym = Y_cum_L_sym(N_Sample_shot);

        Y_P_sym = Y_cum_P_sym;
        Y_L_sym = Y_cum_L_sym;
        residuals = Y_P_sym - Y_L_sym;

        %{\
        J_P_sym   = jacobian(Y_P_sym, k1);
        J_L_sym   = jacobian(Y_L_sym, k2);
        
        n_samp = numel(Y_P_sym);
        I      = MX.eye(n_samp);

        Sigma_r_P = sigma2_cases(1) * I + J_P_sym * Cov_power_cum  * J_P_sym';
        Sigma_r_L = sigma2_cases(1) * I + J_L_sym * Cov_linear_cum * J_L_sym';
        eps_reg   = 1e-10;
        Sigma_r_P = Sigma_r_P + eps_reg * I;
        Sigma_r_L = Sigma_r_L + eps_reg * I;

        j_1 = trace( Sigma_r_P * (Sigma_r_L\I) + Sigma_r_L * (Sigma_r_P\I) - 2*I );
        j_2 = residuals * ((Sigma_r_P\I) + (Sigma_r_L\I)) * residuals';
        %}
        %j   = residuals * residuals';
        %j = j * 1e3;
        j = j_1 + j_2;

        % --- Box constraints ---
        opti.subject_to(zFeedTemp       >= -1);
        opti.subject_to(zFeedTemp       <= 1);
        opti.subject_to(zFeedFlow       >= -1);
        opti.subject_to(zFeedFlow       <= 1);
        opti.subject_to(zFeedPressKnots >= -1);
        opti.subject_to(zFeedPressKnots <= 1);

        opti.set_value(k1, k1_val);
        opti.set_value(k2, k2_val);

        max_change_normalized = 10 / P_half;   % 
        opti.subject_to( diff(zFeedPressKnots, 1, 2) >= -max_change_normalized );
        opti.subject_to( diff(zFeedPressKnots, 1, 2) <=  max_change_normalized );

        % --- Smoothness penalty ---
        % T and F: penalise step-to-step changes on the full time grid
        % P: penalise changes between consecutive 30-min knots
        alpha = 1e-2;
        beta  = 1e-2;
        gamma = 1e-2;
        j_smooth = alpha * sum(diff(zFeedTemp,       1, 2).^2) + ...
                   beta  * sum(diff(zFeedFlow,       1, 2).^2) + ...
                   gamma * sum(diff(zFeedPressKnots, 1, 2).^2);

        % Soft barrier: penalise pressure entering CO2 near-critical / Widom-line
        % region (T = 303-313 K → pseudocritical locus at P ~ 80-120 bar).
        % Using a quadratic penalty below 130 bar repels IPOPT's line search
        % away from that region without hard-narrowing the [P_min, P_max] range.
        %P_barrier_norm = (130 - P_mid) / P_half;          % normalised threshold
        %barrier_viol   = fmax(0, P_barrier_norm - zFeedPressKnots);
        %j_barrier      = 1e2 * sum(barrier_viol .^ 2);    % weight ~ same order as j
        j_barrier = 0;

        opti.minimize(-(j - j_smooth));

        %% --- Initial guess ---
        z0_s              = z0_mat(:, s);
        zFeedTemp_0       = reshape(z0_s(1:N_TF),                        1, N_TF);
        zFeedFlow_0       = reshape(z0_s(N_TF+1:2*N_TF),                1, N_TF);
        zFeedPressKnots_0 = reshape(z0_s(2*N_TF+1:2*N_TF+N_P_knots),   1, N_P_knots);

        opti.set_initial(zFeedTemp,       zFeedTemp_0);
        opti.set_initial(zFeedFlow,       zFeedFlow_0);
        opti.set_initial(zFeedPressKnots, zFeedPressKnots_0);

        % Warm-start state variables by simulating forward at the initial
        % control guess.  Without this IPOPT starts with all states = 0,
        % creating O(Nx*N_shot) infeasible equality constraints.
        % Fix 3: use a single CasADi mapaccum call instead of a MATLAB for-loop.
        params_num          = cell2mat(Parameters(:));
        params_num(which_k) = [k1_val(:); k2_val(:)];

        % Build numeric control sequence at shooting-node granularity
        if use_TF_shooting
            feedTemp_ws  = T_mid + T_half * zFeedTemp_0;                          % 1 x N_TF
            feedFlow_ws  = F_mid + F_half * zFeedFlow_0;                          % 1 x N_TF
            feedPress_ws = P_mid + P_half * zFeedPressKnots_0(TF_P_knot_idx);     % 1 x N_TF
        else
            feedTemp_ws  = T_mid + T_half * zFeedTemp_0(TF_knot_index);           % 1 x N_sim
            feedFlow_ws  = F_mid + F_half * zFeedFlow_0(TF_knot_index);           % 1 x N_sim
            feedPress_ws = P_mid + P_half * zFeedPressKnots_0(P_knot_index);      % 1 x N_sim
        end

        T0_n   = feedTemp_ws(1);
        P0_n   = feedPress_ws(1);
        Z0_n   = Compressibility(T0_n, P0_n, Parameters);
        rho0_n = rhoPB_Comp(T0_n, P0_n, Z0_n, Parameters);
        h0_n   = rho0_n * SpecificEnthalpy(T0_n, P0_n, Z0_n, rho0_n, Parameters);
        x0_num = [C0fluid'; C0solid * bed_mask; (h0_n / 1e4) * ones(nstages, 1); P0_n; 0];

        U_ws = [feedTemp_ws(:)'; feedPress_ws(:)'; feedFlow_ws(:)';
                repmat(params_num(:), 1, N_shot)];   % Nu x N_shot  numeric

        F_power_accum  = F_power_nom.mapaccum('ws_P', N_shot);
        F_linear_accum = F_linear_nom.mapaccum('ws_L', N_shot);

        X_P_guess = full(F_power_accum( x0_num, U_ws));   % Nx x N_shot
        X_L_guess = full(F_linear_accum(x0_num, U_ws));   % Nx x N_shot

        opti.set_initial(X_P, X_P_guess);
        opti.set_initial(X_L, X_L_guess);

        %% --- Solve ---
        try
            sol       = opti.solve();
            valfun    = @(x) sol.value(x);
            r.success = true;
        catch
            valfun    = @(x) opti.debug.value(x);
            r.success = false;
        end

        stats = opti.stats();
        if isstruct(stats) && isfield(stats, 'iterations') && isfield(stats.iterations, 'obj')
            r.j_initial = -stats.iterations.obj(1);
        end

        r.j_1         = nan; % full(valfun(j_1));
        r.j_2         = nan; %full(valfun(j_2));
        r.j           = full(valfun(j));
        r.feedTemp    = full(valfun(feedTemp));             % 1 x N_TF
        r.feedFlow    = full(valfun(feedFlow));             % 1 x N_TF
        r.feedPress   = full(valfun(feedPressSim));         % 1 x N_sim expanded profile
        r.feedPressKnots = full(valfun(feedPressKnots));    % 1 x N_P_knots

    catch %#ok<CTCH>
    end

    msg           = struct;
    msg.seed      = s;
    msg.success   = r.success;
    msg.j_initial = r.j_initial;
    msg.j         = r.j;
    msg.t_seed    = toc(t_seed_start);
    msg.result    = r;

    send(dq, msg);
    results{s} = r;
end
fprintf('\nParallel solve complete: %.1f s total\n\n', toc(t_par));

%% Identify best seed and save
j_vals             = cellfun(@(res) res.j, results);
[j_best, idx_best] = max(j_vals);
best               = results{idx_best};
fprintf('Best seed: %d | j = %.6e\n\n', best.seed, j_best);

%save('results.mat', 'results', 'best');
fprintf('Results saved to results.mat\n');
datetime

%%
%{
results = cell(1, N_seeds);
for s = 1:N_seeds
    seed_file = fullfile(save_dir, sprintf('seed_%d.mat', s));
    if exist(seed_file, 'file') == 2
        tmp = load(seed_file, 'result');
        results{s} = tmp.result;
    else
        results{s} = [];
    end
end
save('results_rebuilt.mat', 'results');
%}

%% -----------------------------------------------------------------------
function local_save_progress(msg, N_total, save_dir)
%LOCAL_SAVE_PROGRESS  DataQueue callback — incremental save + progress line.
% Reset persistent state with:   clear local_save_progress

persistent n t_start results_so_far partial_file

if isempty(n)
    n = 0;
    t_start = posixtime(datetime('now'));

    template = struct( ...
        'seed', [], 'success', false, 'j_initial', NaN, 'j', NaN, ...
        'j_1', NaN, 'j_2', NaN, ...
        'feedTemp', [], 'feedFlow', [], 'feedPress', [], 'feedPressKnots', [], ...
        'feedTemp0', [], 'feedFlow0', [], 'feedPress0', []);
    results_so_far = repmat(template, 1, N_total);

    partial_file = fullfile(save_dir, 'partial.mat');
    save(partial_file, 'results_so_far', '-v7.3');
end

n = n + 1;
results_so_far(msg.seed) = msg.result;

result    = msg.result; %#ok<NASGU>
seed_file = fullfile(save_dir, sprintf('seed_%d.mat', msg.seed));
save(seed_file, 'result', '-v7.3');
save(partial_file, 'results_so_far', '-v7.3');

elapsed = posixtime(datetime('now')) - t_start;
eta     = (elapsed / n) * (N_total - n);
eta_str = '';
if eta >= 60
    eta_str = sprintf('%.1f min', eta / 60);
else
    eta_str = sprintf('%.0f s', eta);
end

fprintf('[%2d/%2d] seed=%3d | ok=%d | j0=%.4e → j=%.4e (j1=%.3e j2=%.3e) | %.1f min | ETA %s\n', ...
    n, N_total, msg.seed, msg.success, msg.j_initial, msg.j, ...
    msg.result.j_1, msg.result.j_2, msg.t_seed / 60, eta_str);
end