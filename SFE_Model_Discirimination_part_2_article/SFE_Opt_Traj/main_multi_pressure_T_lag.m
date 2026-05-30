%% Initialization
run_on       = 'locally';  % Choose 'trition' to use trition seeting, otherwise local settings

% Load casadi
if run_on == 'trition'
    initParPool
    datetime
    casadi_path = '/scratch/work/sliczno1/SFE-Model-Discrimination/SFE_Model_Discirimination_part_2_article/SFE_Opt_Traj/casadi_folder';
else
    startup;
    casadi_path = 'C:\Dev\casadi-3.6.3-windows64-matlab2018b';
end

%% =========================================================================
%  YIELD TYPE CONFIGURATION
%  Set YIELD_TYPE to either 'cumulative' or 'differential'.
% =========================================================================
YIELD_TYPE   = 'cumulative';
save_results = 'no save';     % Use 'save' to save the results

assert(ismember(YIELD_TYPE, {'cumulative','differential'}), ...
    'YIELD_TYPE must be ''cumulative'' or ''differential''.');

%% Run configuration
N_workers    = 1;
N_seeds      = 1;
n_init_knots = 2;
N_iter       = 2;

% --- Simulation time grid ---
timeStep  = 10;    % model integration step [min]
finalTime = 600;   % total extraction time [min]

% --- Pressure control grid ---
% Pressure is piecewise constant on a coarser 30-min grid.
% Must be a multiple of timeStep.
pressureStep = 60;   % [min] — pressure switching interval

%% Time grids
Time_in_sec     = (timeStep:timeStep:finalTime) * 60;
Time            = [0, Time_in_sec/60];
N_Time          = length(Time_in_sec);       % number of model steps
timeStep_in_sec = timeStep * 60;

%% Optimizer settings
nlp_opts = struct;
nlp_opts.ipopt.max_iter                   = N_iter;
%nlp_opts.ipopt.max_cpu_time               = ipopt_max_time;
nlp_opts.ipopt.tol                        = 1e-7;
nlp_opts.ipopt.acceptable_tol             = 1e-5;
nlp_opts.ipopt.acceptable_iter            = 10;
nlp_opts.ipopt.hessian_approximation      = 'limited-memory';
nlp_opts.ipopt.nlp_scaling_max_gradient   = 10;
nlp_opts.ipopt.limited_memory_init_val    = 0.01;
nlp_opts.ipopt.print_level                = 5;
nlp_opts.print_time                       = 0;
nlp_opts.ipopt.mu_strategy                = 'adaptive';
nlp_opts.ipopt.obj_scaling_factor         = 0.01;
nlp_opts.ipopt.bound_push                 = 0.1;
nlp_opts.ipopt.limited_memory_max_history = min(2 * N_Time, 50);


fprintf('=============================================================================\n');
fprintf('   Optimal trajectory for model discrimination — varying pressure\n');
fprintf('=============================================================================\n\n');

fprintf('Yield type : %s\n\n', YIELD_TYPE);

assert(mod(pressureStep, timeStep) == 0, ...
    'pressureStep must be a multiple of timeStep.');

checkpoint_dir = sprintf('varying_pressure_%s_seeds_results', YIELD_TYPE);

% Pressure control knots: one value per pressureStep interval
N_P_knots = finalTime / pressureStep;        % e.g. 600/30 = 20 knots
% Index of model step at which each pressure knot takes effect
% (knot k applies from step (k-1)*ratio+1 to k*ratio)
P_ratio   = pressureStep / timeStep;         % steps per pressure interval

% Expand pressure knots to full model step grid (each knot repeated P_ratio times)
% This mapping is applied inside the worker to build the full uu matrix.

fprintf('=== Run Configuration ===\n');
fprintf('Yield type        : %s\n',   YIELD_TYPE);
fprintf('Model step        : %d min\n', timeStep);
fprintf('Pressure step     : %d min\n', pressureStep);
fprintf('Pressure knots    : %d\n\n',   N_P_knots);

%% Load parameters and data
addpath(casadi_path);

Parameters_table = readtable('Parameters.csv');
Parameters       = num2cell(Parameters_table{:,3});
LabResults       = readmatrix('dataset_2.xlsx');

which_k = (0:9) + 44;
k1_val  = cell2mat(Parameters((0:3) + 44));
k2_val  = cell2mat(Parameters((4:9) + 44));

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

fprintf('Time grid  : %d steps of %d min | %d sample times\n\n', ...
    N_Time, timeStep, numel(SAMPLE));

%% Covariance matrices — cumulative yield
Cov_power_cum = [
    1.0035e-02,  1.1795e-02,  1.8268e-03,  2.5611e-02;
    1.1795e-02,  5.6469e-02,  3.1182e-03,  2.8266e-02;
    1.8268e-03,  3.1182e-03,  5.7241e-03,  6.4459e-03;
    2.5611e-02,  2.8266e-02,  6.4459e-03,  7.1744e-02
    ];

Cov_linear_cum = [
    2.7801e-02,  3.5096e-02, -6.9596e-03,  7.1573e-02,  1.0992e-02, -1.2661e-02;
    3.5096e-02,  6.8482e-01, -5.0531e-02, -4.8187e-02,  3.9209e-01, -2.6206e-02;
   -6.9596e-03, -5.0531e-02,  4.5693e-03, -7.7054e-03, -1.5915e-02,  3.3012e-03;
    7.1573e-02, -4.8187e-02, -7.7054e-03,  2.9254e-01,  6.5758e-02, -4.6300e-02;
    1.0992e-02,  3.9209e-01, -1.5915e-02,  6.5758e-02,  2.9506e+00, -1.3133e-01;
   -1.2661e-02, -2.6206e-02,  3.3012e-03, -4.6300e-02, -1.3133e-01,  1.2975e-02
    ];

%% Covariance matrices — differential yield
Cov_power_diff = [
    3.2963e-03,  1.2094e-03, -2.5042e-03,  6.8414e-03;
    1.2094e-03,  1.0981e-01, -5.7125e-04,  2.3381e-03;
   -2.5042e-03, -5.7125e-04,  1.3915e-02, -2.7301e-04;
    6.8414e-03,  2.3381e-03, -2.7301e-04,  3.8686e-02
    ];

Cov_linear_diff = [
    2.2178e-02,  1.0828e-02, -4.3832e-03,  4.3992e-02,  4.4695e-03, -7.4634e-03;
    1.0828e-02,  4.3513e-01, -2.4832e-02,  3.0423e-03,  6.7633e-01, -3.3289e-02;
   -4.3832e-03, -2.4832e-02,  2.1282e-03, -7.3766e-03, -3.2298e-02,  2.8884e-03;
    4.3992e-02,  3.0423e-03, -7.3766e-03,  3.1429e-01, -6.2258e-02, -4.7085e-02;
    4.4695e-03,  6.7633e-01, -3.2298e-02, -6.2258e-02,  6.1032e+00, -2.4975e-01;
   -7.4634e-03, -3.3289e-02,  2.8884e-03, -4.7085e-02, -2.4975e-01,  1.8474e-02
    ];

%% Select active covariance and noise variance
sigma2_cases = [2.45e-2, 1.386e-3];

switch YIELD_TYPE
    case 'cumulative'
        Cov_power_active  = Cov_power_cum;
        Cov_linear_active = Cov_linear_cum;
        sigma2_active     = sigma2_cases(1);
    case 'differential'
        Cov_power_active  = Cov_power_diff;
        Cov_linear_active = Cov_linear_diff;
        sigma2_active     = sigma2_cases(2);
end

fprintf('Active noise variance : %.3e\n\n', sigma2_active);

%% Simulation geometry
m_total = 3.0;
before  = 0.04;
bed     = 0.92;

nstages = Parameters{1};
r_geom  = Parameters{3};
epsi    = Parameters{4};
L       = Parameters{6};

nstagesbefore = 1:floor(before * nstages);
nstagesbed    = nstagesbefore(end)+1 : nstagesbefore(end) + floor(bed * nstages);
nstagesafter  = nstagesbed(end)+1 : nstages;

bed_mask = nan(nstages, 1);
bed_mask(nstagesbefore) = 0;
bed_mask(nstagesbed)    = 1;
bed_mask(nstagesafter)  = 0;

V_slice = (L/nstages) * pi * r_geom^2;
V_bed   = V_slice * numel(nstagesbed);

V_before_fluid = repmat(V_slice * numel(nstagesbefore) / numel(nstagesbefore), numel(nstagesbefore), 1);
V_bed_fluid    = repmat(V_bed * (1-epsi) / numel(nstagesbed),                  numel(nstagesbed),    1);
V_after_fluid  = repmat(V_slice * numel(nstagesafter) / numel(nstagesafter),   numel(nstagesafter),  1);
V_fluid        = [V_before_fluid; V_bed_fluid; V_after_fluid];

C0solid       = m_total * 1e-3 / (V_bed * epsi);
Parameters{2} = C0solid;

m_fluid = zeros(1, nstages);
C0fluid = m_fluid * 1e-3 ./ V_fluid';

epsi_mask           = epsi .* bed_mask;
one_minus_epsi_mask = 1 - epsi_mask;

Nx = 3 * nstages + 2;
Nu = 3 + numel(Parameters);

%% Bounds and normalisation
% Temperature and flow — same as fixed-pressure version
T_min  = 303;   T_max  = 313;
T_mid  = 0.5 * (T_min + T_max);
T_half = 0.5 * (T_max - T_min);

F_min  = 3.3e-5;  F_max  = 6.7e-5;
F_mid  = 0.5 * (F_min + F_max);
F_half = 0.5 * (F_max - F_min);

% Pressure — bounded within validated model range
P_min  = 100;   P_max  = 200;   % [bar]
P_mid  = 0.5 * (P_min + P_max);
P_half = 0.5 * (P_max - P_min);

fprintf('Decision variable bounds:\n');
fprintf('  Temperature : [%.1f, %.1f] K\n',    T_min, T_max);
fprintf('  Flow rate   : [%.2e, %.2e] kg/s\n', F_min, F_max);
fprintf('  Pressure    : [%.1f, %.1f] bar  (%d knots, every %d min)\n\n', ...
    P_min, P_max, N_P_knots, pressureStep);

%% Pre-generate initial guesses via Latin Hypercube Sampling
% Decision variables: T (N_Time), F (N_Time), P (N_P_knots)
n_lhs_cols = 2 * n_init_knots + n_init_knots;   % T knots + F knots + P knots

rng(0);
lhs = lhsdesign(N_seeds, n_lhs_cols, 'criterion', 'maximin', 'iterations', 20);

% Storage: normalised [T(N_Time); F(N_Time); P(N_P_knots)]
z0_mat        = zeros(2*N_Time + N_P_knots, N_seeds);
feedTemp0_all = zeros(N_Time,    N_seeds);
feedFlow0_all = zeros(N_Time,    N_seeds);
feedPres0_all = zeros(N_P_knots, N_seeds);

for s = 1:N_seeds
    rng(s);

    % T and F: interpolated from random knots (same as before)
    interior_pool = 2:(N_Time - 1);
    n_interior    = n_init_knots - 2;
    pick          = randperm(numel(interior_pool), n_interior);
    interior_idx  = sort(interior_pool(pick));
    init_knot_idx = [1, interior_idx, N_Time];

    temp_knots = T_min + (T_max - T_min) * lhs(s,                1 :   n_init_knots);
    flow_knots = F_min + (F_max - F_min) * lhs(s, n_init_knots + 1 : 2*n_init_knots);
    pres_knots = P_min + (P_max - P_min) * lhs(s, 2*n_init_knots+1 : 3*n_init_knots);

    feedTemp_0 = interp1(init_knot_idx, temp_knots, 1:N_Time,    'linear');
    feedFlow_0 = interp1(init_knot_idx, flow_knots, 1:N_Time,    'linear');
    % Pressure knots live directly on the N_P_knots grid — interpolate to that
    pres_knot_idx  = round(linspace(1, N_P_knots, n_init_knots));
    %feedPres_0     = interp1(pres_knot_idx, pres_knots, 1:N_P_knots, 'linear', 'extrap');
    %feedPres_0     = max(P_min, min(P_max, feedPres_0));   % clamp to bounds
    feedPres_0      = 150 * ones(1,N_P_knots);

    z0_mat(:, s) = [(feedTemp_0 - T_mid) / T_half, ...
                    (feedFlow_0 - F_mid) / F_half,  ...
                    (feedPres_0 - P_mid) / P_half]';

    feedTemp0_all(:, s) = feedTemp_0(:);
    feedFlow0_all(:, s) = feedFlow_0(:);
    feedPres0_all(:, s) = feedPres_0(:);
end

fprintf('Initial guess summary across %d seeds (LHS):\n', N_seeds);
fprintf('  Temp  mean %.2f K,  std %.2f K\n',  mean(feedTemp0_all(:)), std(feedTemp0_all(:)));
fprintf('  Flow  mean %.2e,    std %.2e\n',     mean(feedFlow0_all(:)), std(feedFlow0_all(:)));
fprintf('  Press mean %.1f bar, std %.1f bar\n\n', mean(feedPres0_all(:)), std(feedPres0_all(:)));

%% Parallel pool
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', N_workers);
end
parfevalOnAll(@() addpath(casadi_path), 0);

%% Create checkpoint folder
if ~exist(checkpoint_dir, 'dir')
    mkdir(checkpoint_dir);
end

%% Bundle fixed data for workers
fixed = struct( ...
    'YIELD_TYPE',          YIELD_TYPE,          ...
    'N_Time',              N_Time,              ...
    'N_P_knots',           N_P_knots,           ...
    'P_ratio',             P_ratio,             ...
    'nlp_opts',            nlp_opts,            ...
    'k1_val',              k1_val,              ...
    'k2_val',              k2_val,              ...
    'which_k',             which_k,             ...
    'Parameters',          {Parameters},        ...
    'bed_mask',            bed_mask,            ...
    'epsi_mask',           epsi_mask,           ...
    'one_minus_epsi_mask', one_minus_epsi_mask, ...
    'C0fluid',             C0fluid,             ...
    'C0solid',             C0solid,             ...
    'N_Sample',            N_Sample,            ...
    'sigma2_active',       sigma2_active,       ...
    'Cov_power_active',    Cov_power_active,    ...
    'Cov_linear_active',   Cov_linear_active,   ...
    'T_mid',               T_mid,               ...
    'T_half',              T_half,              ...
    'F_mid',               F_mid,               ...
    'F_half',              F_half,              ...
    'P_mid',               P_mid,               ...
    'P_half',              P_half,              ...
    'P_min',               P_min,               ...
    'P_max',               P_max,               ...
    'Nx',                  Nx,                  ...
    'Nu',                  Nu,                  ...
    'nstages',             nstages,             ...
    'timeStep_in_sec',     timeStep_in_sec,     ...
    'casadi_path',         casadi_path          ...
);

%% Launch seeds via parfeval
t_par = tic;
fprintf('Solving %d seeds on %d workers...\n', N_seeds, N_workers);
results = cell(1, N_seeds);

futures(1:N_seeds) = parallel.FevalFuture;
for s = 1:N_seeds
    futures(s) = parfeval(pool, @solve_one_seed, 1, ...
        s, z0_mat(:,s), feedTemp0_all(:,s), feedFlow0_all(:,s), ...
        feedPres0_all(:,s), fixed);
end

%% Collect results
for i = 1:N_seeds
    [~, res] = fetchNext(futures);
    results{res.seed} = res;

    fprintf('[%2d/%2d] seed=%3d | ok=%d | j0=%.4e -> j=%.4e | %.1f min\n', ...
        i, N_seeds, res.seed, res.success, res.j_initial, res.j, res.elapsed_min);

    if ~res.success && ~isempty(res.error_msg)
        fprintf('         error: %s\n', res.error_msg);
    end

    % Per-seed checkpoint
    if isequal(save_results,'save')
        checkpoint_file = fullfile(checkpoint_dir, ...
            sprintf('varyP_%s_seed_%03d.mat', YIELD_TYPE, res.seed));
        save(checkpoint_file, 'res', '-v7.3');
    end
end

fprintf('\nSolve complete: %.1f s total\n\n', toc(t_par));

% Aggregate save
if isequal(save_results,'save')
    aggregate_file = fullfile(checkpoint_dir, ...
        sprintf('varyP_%s_all_seeds.mat', YIELD_TYPE));
    save(aggregate_file, 'results', 'fixed', 'YIELD_TYPE', '-v7.3');
    fprintf('Aggregate results saved to: %s\n', aggregate_file);
end

fprintf('Done.\n');
datetime


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Worker function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function res = solve_one_seed(s, z0_s, feedTemp0_s, feedFlow0_s, feedPres0_s, fixed)

    addpath(fixed.casadi_path);

    t0        = tic;
    N_Time    = fixed.N_Time;
    N_P_knots = fixed.N_P_knots;
    P_ratio   = fixed.P_ratio;
    n_samp    = numel(fixed.N_Sample);
    n_diff    = n_samp - 1;
    use_cumul = strcmp(fixed.YIELD_TYPE, 'cumulative');
    n_out     = n_samp * use_cumul + n_diff * (~use_cumul);

    res = struct( ...
        'seed',        s,                   ...
        'success',     false,               ...
        'j_initial',   NaN,                 ...
        'j',           NaN,                 ...
        'j_1',         NaN,                 ...
        'j_2',         NaN,                 ...
        'feedTemp',    NaN(1, N_Time),      ...
        'feedFlow',    NaN(1, N_Time),      ...
        'feedPres',    NaN(1, N_P_knots),   ...
        'feedPres_full', NaN(1, N_Time),    ...   % expanded to model grid
        'feedTemp0',   feedTemp0_s',        ...
        'feedFlow0',   feedFlow0_s',        ...
        'feedPres0',   feedPres0_s',        ...
        'Y_P',         NaN(1, n_out),       ...
        'Y_L',         NaN(1, n_out),       ...
        'Y_cum_P',     NaN(1, n_samp),      ...
        'Y_cum_L',     NaN(1, n_samp),      ...
        'yield_type',  fixed.YIELD_TYPE,    ...
        'elapsed_min', NaN,                 ...
        'error_msg',   '');

    try
        import casadi.*

        opti = casadi.Opti();
        opti.solver('ipopt', fixed.nlp_opts);

        %% Decision variables (all normalised to [-1, 1])
        k1 = opti.parameter(4);
        k2 = opti.parameter(6);

        zFeedTemp = opti.variable(1, N_Time);      % temperature profile
        zFeedFlow = opti.variable(1, N_Time);      % flow rate profile
        zFeedPres = opti.variable(1, N_P_knots);   % pressure knots (coarse grid)

        feedTemp = fixed.T_mid + fixed.T_half * zFeedTemp;   % [K]
        feedFlow = fixed.F_mid + fixed.F_half * zFeedFlow;   % [kg/s]
        feedPres = fixed.P_mid + fixed.P_half * zFeedPres;   % [bar]

        %% Expand pressure knots to the full model step grid
        % Each pressure knot k applies for P_ratio consecutive model steps.
        % kron([1,1,...,1], feedPres) with P_ratio ones repeats each knot.
        % Result: 1 x N_Time vector where each block of P_ratio steps
        % holds the same pressure value.
        feedPres_full = reshape( ...
            repmat(feedPres, P_ratio, 1), ...   % P_ratio x N_P_knots
            1, N_Time);                          % 1 x N_Time

        %% Build control matrix: [T; P; F] x N_Time
        uu = [feedTemp', feedPres_full', feedFlow'];   % N_Time x 3

        %% Initial condition
        % The initial enthalpy and pressure state depend on the first
        % pressure knot, which is now an optimisation variable.
        T0 = feedTemp(1);
        P0 = feedPres(1);   % first pressure knot = initial pressure

        Z0            = Compressibility(T0, P0, fixed.Parameters);
        rho0          = rhoPB_Comp(T0, P0, Z0, fixed.Parameters);
        enthalpy_rho0 = rho0 * SpecificEnthalpy(T0, P0, Z0, rho0, fixed.Parameters);

        x0 = [fixed.C0fluid';
               fixed.C0solid * fixed.bed_mask;
               (enthalpy_rho0 / 1e4) * ones(fixed.nstages, 1);
               P0;
               0];

        %% Parameter vector
        Parameters_sym                = MX(cell2mat(fixed.Parameters));
        Parameters_sym(fixed.which_k) = [k1; k2];
        U_base = [uu'; repmat(Parameters_sym, 1, N_Time)];

        %% Integrators
        f_power  = @(x, u) modelSFE_thermal_lag(x, u, fixed.bed_mask, ...
            fixed.timeStep_in_sec, 'Power_model',  ...
            fixed.epsi_mask, fixed.one_minus_epsi_mask);
        f_linear = @(x, u) modelSFE_thermal_lag(x, u, fixed.bed_mask, ...
            fixed.timeStep_in_sec, 'Linear_model', ...
            fixed.epsi_mask, fixed.one_minus_epsi_mask);

        F_accum_power  = buildIntegrator(f_power,  [fixed.Nx, fixed.Nu], ...
            fixed.timeStep_in_sec, 'cvodes') ...
            .mapaccum('F_accum_power',  N_Time);
        F_accum_linear = buildIntegrator(f_linear, [fixed.Nx, fixed.Nu], ...
            fixed.timeStep_in_sec, 'cvodes') ...
            .mapaccum('F_accum_linear', N_Time);

        X_power_nom  = F_accum_power( x0, U_base);
        X_linear_nom = F_accum_linear(x0, U_base);

        %% Yield extraction
        Y_cum_P_all = [0, X_power_nom( fixed.Nx, :)];
        Y_cum_L_all = [0, X_linear_nom(fixed.Nx, :)];

        Y_cum_P = Y_cum_P_all(fixed.N_Sample);
        Y_cum_L = Y_cum_L_all(fixed.N_Sample);

        D_mat = diff(eye(n_samp));
        D     = MX(D_mat);

        dY_P = D * Y_cum_P';
        dY_L = D * Y_cum_L';

        %% Active yield and covariance
        if use_cumul
            Y_P_active = Y_cum_P';
            Y_L_active = Y_cum_L';
            n_active   = n_samp;

            J_P = jacobian(Y_P_active, k1);
            J_L = jacobian(Y_L_active, k2);

            I_active  = MX.eye(n_samp);
            Sigma_r_P = fixed.sigma2_active * I_active + ...
                        J_P * fixed.Cov_power_active  * J_P';
            Sigma_r_L = fixed.sigma2_active * I_active + ...
                        J_L * fixed.Cov_linear_active * J_L';
        else
            Y_P_active = dY_P;
            Y_L_active = dY_L;
            n_active   = n_diff;

            J_P = jacobian(dY_P, k1);
            J_L = jacobian(dY_L, k2);

            I_active  = MX.eye(n_diff);
            DDt       = MX(D_mat * D_mat');
            Sigma_r_P = fixed.sigma2_active * DDt + ...
                        J_P * fixed.Cov_power_active  * J_P';
            Sigma_r_L = fixed.sigma2_active * DDt + ...
                        J_L * fixed.Cov_linear_active * J_L';
        end

        Sigma_r_P = Sigma_r_P + 1e-10 * I_active;
        Sigma_r_L = Sigma_r_L + 1e-10 * I_active;

        %% Objective: T-optimality
        residuals = (Y_P_active - Y_L_active)';

        j_1 = trace(Sigma_r_P * (Sigma_r_L \ I_active) + ...
                    Sigma_r_L * (Sigma_r_P \ I_active) - 2*I_active);
        j_2 = residuals * ((Sigma_r_P \ I_active) + (Sigma_r_L \ I_active)) * residuals';
        j   = j_1 + j_2;

        %% Box constraints
        opti.subject_to(zFeedTemp >= -1);
        opti.subject_to(zFeedTemp <=  1);
        opti.subject_to(zFeedFlow >= -1);
        opti.subject_to(zFeedFlow <=  1);
        opti.subject_to(zFeedPres >= -1);
        opti.subject_to(zFeedPres <=  1);

        %% Smoothness penalty
        % Applied to all three control profiles.
        % Pressure is on a coarser grid so its penalty weight is scaled
        % by P_ratio to make it dimensionally comparable to T and F.
        alpha    = 1e-3;
        beta     = 1e-3;
        gamma_P  = 1e-3 * P_ratio;
        j_smooth = alpha   * sum(diff(zFeedTemp, 1, 2).^2) + ...
                   beta    * sum(diff(zFeedFlow, 1, 2).^2) + ...
                   gamma_P * sum(diff(zFeedPres, 1, 2).^2);

        opti.minimize(-(j - j_smooth));

        %% Fix parameter values
        opti.set_value(k1, fixed.k1_val);
        opti.set_value(k2, fixed.k2_val);

        %% Initial guesses
        zFeedTemp_0 = reshape(z0_s(1:N_Time),                         1, N_Time);
        zFeedFlow_0 = reshape(z0_s(N_Time+1:2*N_Time),                1, N_Time);
        zFeedPres_0 = reshape(z0_s(2*N_Time+1:2*N_Time+N_P_knots),    1, N_P_knots);

        opti.set_initial(zFeedTemp, zFeedTemp_0);
        opti.set_initial(zFeedFlow, zFeedFlow_0);
        opti.set_initial(zFeedPres, zFeedPres_0);

        %% Solve
        try
            sol         = opti.solve();
            valfun      = @(x) sol.value(x);
            res.success = true;
        catch solve_err
            valfun        = @(x) opti.debug.value(x);
            res.success   = false;
            res.error_msg = solve_err.message;
        end

        %% Extract initial objective from IPOPT log
        try
            stats = opti.stats();
            if isstruct(stats) && isfield(stats, 'iterations') && ...
                    isfield(stats.iterations, 'obj')
                res.j_initial = -stats.iterations.obj(1);
            end
        catch
        end

        %% Extract results
        res.j_1          = full(valfun(j_1));
        res.j_2          = full(valfun(j_2));
        res.j            = full(valfun(j));
        res.feedTemp     = full(valfun(feedTemp));
        res.feedFlow     = full(valfun(feedFlow));
        res.feedPres     = full(valfun(feedPres));        % 1 x N_P_knots
        res.feedPres_full = full(valfun(feedPres_full));  % 1 x N_Time

        res.Y_P     = full(valfun(Y_P_active'));
        res.Y_L     = full(valfun(Y_L_active'));
        res.Y_cum_P = full(valfun(Y_cum_P));
        res.Y_cum_L = full(valfun(Y_cum_L));

    catch outer_err
        res.success   = false;
        res.error_msg = outer_err.message;
    end

    res.elapsed_min = toc(t0) / 60;
end