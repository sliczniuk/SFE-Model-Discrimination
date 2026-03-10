%% Initialization
startup;
%initParPool

fprintf('=============================================================================\n');
fprintf('   Optimal trajectory for model discrimination (parfor multistart)\n');
fprintf('=============================================================================\n\n');

%% Run configuration
N_seeds      = 12;
N_workers    = 6;
n_init_knots = 10;

timeStep  =  10;       % minutes per simulation step
finalTime = 600;       % total experiment duration [min]

% Pressure is a piecewise-constant signal that may change every 30 min.
% The time axis has steps of 10 min, so one pressure interval = 3 steps.
% With finalTime = 600 min this gives exactly 20 pressure intervals
P_switch_interval = 60;   % [min]  — change this to adjust how often P can switch

%casadi_path  = '/scratch/work/sliczno1/SFE-Model-Discrimination/SFE_Model_Discirimination_part_2_article/SFE_Opt_Traj/casadi_folder';
casadi_path  = 'C:\Dev\casadi-3.6.3-windows64-matlab2018b';

% Directory for incremental per-seed saves
%save_dir = 'seed_results';
%if ~exist(save_dir, 'dir')
%    mkdir(save_dir);
%end

%% Optimizer Settings
nlp_opts = struct;
nlp_opts.ipopt.max_iter              = 20;
nlp_opts.ipopt.tol                   = 1e-7;
nlp_opts.ipopt.acceptable_tol        = 1e-5;
nlp_opts.ipopt.acceptable_iter       = 10;
nlp_opts.ipopt.hessian_approximation      = 'limited-memory';
nlp_opts.ipopt.nlp_scaling_max_gradient   = 10;   % scale NLP if initial gradient exceeds this
nlp_opts.ipopt.limited_memory_init_val    = 0.01; % small initial H^{-1} → conservative first step
nlp_opts.ipopt.print_level                = 5;
nlp_opts.print_time                       = 0;
nlp_opts.ipopt.mu_strategy = 'adaptive';
nlp_opts.ipopt.obj_scaling_factor = 0.01;
nlp_opts.ipopt.bound_push = 0.1;

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
Time_in_sec     = (timeStep:timeStep:finalTime) * 60;
Time            = [0, Time_in_sec/60];
N_Time          = length(Time_in_sec);
timeStep_in_sec = timeStep * 60;

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
steps_per_interval = P_switch_interval / timeStep;   % must be integer
if mod(steps_per_interval, 1) ~= 0
    error('P_switch_interval (%d min) must be a multiple of timeStep (%d min).', ...
        P_switch_interval, timeStep);
end
steps_per_interval = round(steps_per_interval);
N_P_knots = finalTime / P_switch_interval;           % number of distinct pressure values
if mod(N_P_knots, 1) ~= 0
    error('finalTime (%d min) must be divisible by P_switch_interval (%d min).', ...
        finalTime, P_switch_interval);
end
N_P_knots = round(N_P_knots);

% Map each simulation step to its pressure knot index (1-based)
% Step i (1..N_Time) belongs to interval ceil(i / steps_per_interval)
P_knot_index = ceil((1:N_Time) / steps_per_interval);   % 1 x N_Time

fprintf('Pressure switching: every %d min | %d distinct pressure values\n\n', ...
    P_switch_interval, N_P_knots);

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
    P_min, P_max, N_P_knots, P_switch_interval);

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

z0_mat         = zeros(2 * N_Time + N_P_knots, N_seeds);
feedTemp0_all  = zeros(N_Time,    N_seeds);
feedFlow0_all  = zeros(N_Time,    N_seeds);
feedPress0_all = zeros(N_P_knots, N_seeds);   % one value per 30-min interval

for s = 1:N_seeds
    rng(s);   % governs knot *positions* for T and F only
    interior_pool = 2:(N_Time - 1);
    n_interior    = n_init_knots - 2;
    pick          = randperm(numel(interior_pool), n_interior);
    interior_idx  = sort(interior_pool(pick));
    init_knot_idx = [1, interior_idx, N_Time];

    % Scale LHS columns to physical ranges
    temp_knots  = T_min + (T_max - T_min) * lhs(s,                  1 :   n_init_knots);
    flow_knots  = F_min + (F_max - F_min) * lhs(s,   n_init_knots + 1 : 2*n_init_knots);
    %press_vals  = P_min + (P_max - P_min) * lhs(s, 2*n_init_knots + 1 : 2*n_init_knots + N_P_knots);
    press_vals   = P_min + (P_max - P_min) * rand();

    % Interpolate T and F knots onto the full time grid
    feedTemp_0  = interp1(init_knot_idx, temp_knots, 1:N_Time, 'linear');
    feedFlow_0  = interp1(init_knot_idx, flow_knots, 1:N_Time, 'linear');

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

parfor s = 1:N_seeds
    t_seed_start = tic;

    % Default failure result
    r = struct('seed', s, 'success', false, 'j_initial', NaN, 'j', NaN, ...
        'j_1', NaN, 'j_2', NaN, ...
        'feedTemp',    NaN(1, N_Time),    ...
        'feedFlow',    NaN(1, N_Time),    ...
        'feedPress',   NaN(1, N_Time),    ...   % full time-expanded profile
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

        % --- Temperature and flow: free at every timestep ---
        zFeedTemp = opti.variable(1, N_Time);
        zFeedFlow = opti.variable(1, N_Time);

        feedTemp = T_mid + T_half * zFeedTemp;
        feedFlow = F_mid + F_half * zFeedFlow;

        % --- Pressure: one normalised decision variable per 30-min interval ---
        % zFeedPressKnots is 1 x N_P_knots; each element is held constant
        % for steps_per_interval consecutive simulation steps (zero-order hold).
        zFeedPressKnots = opti.variable(1, N_P_knots);

        % Expand to full time grid using the pre-computed index map
        % feedPress(t) = P_mid + P_half * zFeedPressKnots(P_knot_index(t))
        feedPressKnots = P_mid + P_half * zFeedPressKnots;   % 1 x N_P_knots physical
        feedPress      = feedPressKnots(P_knot_index);        % 1 x N_Time  via index broadcast

        % --- Initial condition uses pressure at t=0 (first knot) ---
        T  = feedTemp(1);
        P  = feedPress(1);
        F  = feedFlow(1);
        uu = [feedTemp', feedPress', feedFlow'];

        Z            = Compressibility(T, P, Parameters);
        rho          = rhoPB_Comp(T, P, Z, Parameters);
        enthalpy_rho = rho * SpecificEnthalpy(T, P, Z, rho, Parameters);

        % enthalpy_rho stored scaled by 1/1e4 to match modelSFE ENTHALPY_SCALE=1e4
        x0 = [C0fluid'; C0solid * bed_mask; (enthalpy_rho / 1e4) * ones(nstages, 1); P; 0];

        Parameters_sym          = MX(cell2mat(Parameters));
        Parameters_sym(which_k) = k(1:numel(which_k));
        U_base                  = [uu'; repmat(Parameters_sym, 1, N_Time)];

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

        Y_cum_P_sym = [0, X_power_nom(Nx, :)];
        Y_cum_P_sym = Y_cum_P_sym(N_Sample);
        Y_cum_L_sym = [0, X_linear_nom(Nx, :)];
        Y_cum_L_sym = Y_cum_L_sym(N_Sample);

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
        j   = residuals * residuals';
        j = j * 1e3;
        %j = j_1 + j_2;

        % --- Box constraints ---
        opti.subject_to(zFeedTemp       >= -1);
        opti.subject_to(zFeedTemp       <= 1);
        opti.subject_to(zFeedFlow       >= -1);
        opti.subject_to(zFeedFlow       <= 1);
        opti.subject_to(zFeedPressKnots >= -1);
        opti.subject_to(zFeedPressKnots <= 1);

        opti.set_value(k1, k1_val);
        opti.set_value(k2, k2_val);

        max_change_normalized = 20 / P_half;   % 
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

        opti.minimize(-(j - j_smooth) + j_barrier);

        %% --- Initial guess ---
        z0_s              = z0_mat(:, s);
        zFeedTemp_0       = reshape(z0_s(1:N_Time),                          1, N_Time);
        zFeedFlow_0       = reshape(z0_s(N_Time+1:2*N_Time),                 1, N_Time);
        zFeedPressKnots_0 = reshape(z0_s(2*N_Time+1:2*N_Time+N_P_knots),    1, N_P_knots);

        opti.set_initial(zFeedTemp,       zFeedTemp_0);
        opti.set_initial(zFeedFlow,       zFeedFlow_0);
        opti.set_initial(zFeedPressKnots, zFeedPressKnots_0);

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
        r.feedTemp    = full(valfun(feedTemp));
        r.feedFlow    = full(valfun(feedFlow));
        r.feedPress   = full(valfun(feedPress));        % 1 x N_Time expanded profile
        r.feedPressKnots = full(valfun(feedPressKnots)); % 1 x N_P_knots

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