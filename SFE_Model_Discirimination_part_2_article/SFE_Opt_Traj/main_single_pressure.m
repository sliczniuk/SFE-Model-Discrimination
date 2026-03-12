%% Initialization
startup;
%initParPool

fprintf('=============================================================================\n');
fprintf('   Optimal trajectory for model discrimination — fixed pressure\n');
fprintf('=============================================================================\n\n');

%% Run configuration
N_seeds      = 2;
N_workers    = 2;
n_init_knots = 30;

timeStep  =  10;       % minutes per simulation step
finalTime = 600;       % total experiment duration [min]

P_fixed = 130;         % [bar]  fixed pressure — not a decision variable

casadi_path  = 'C:\Dev\casadi-3.6.3-windows64-matlab2018b';
%casadi_path  = '/scratch/work/sliczno1/SFE-Model-Discrimination/SFE_Model_Discirimination_part_2_article/SFE_Opt_Traj/casadi_folder';

checkpoint_dir = sprintf('single_pressure_%g_seeds_results', P_fixed);

%% Time grid and sample matching
Time_in_sec     = (timeStep:timeStep:finalTime) * 60;
Time            = [0, Time_in_sec/60];
N_Time          = length(Time_in_sec);
timeStep_in_sec = timeStep * 60;

%% Optimizer Settings
nlp_opts = struct;
nlp_opts.ipopt.max_iter                   = 2;
nlp_opts.ipopt.hessian_approximation      = 'limited-memory';
nlp_opts.ipopt.limited_memory_max_history = min(2 * N_Time, 50);
nlp_opts.ipopt.mu_strategy                = 'adaptive';
nlp_opts.ipopt.print_level                = 0;
nlp_opts.print_time                       = 0;
nlp_opts.ipopt.bound_push                 = 0.1;

fprintf('=== Run Configuration ===\n');
fprintf('Seeds: %d | Workers: %d | Knots: %d | Max iter: %d\n', ...
    N_seeds, N_workers, n_init_knots, nlp_opts.ipopt.max_iter);
fprintf('Fixed pressure: %.1f bar\n\n', P_fixed);

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

fprintf('Time grid: %d steps of %d min | %d sample times\n\n', ...
    N_Time, timeStep, numel(SAMPLE));

%% Covariance matrices
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

sigma2_cases = [2.45e-2, 1.386e-3, 1.007e-2];

%% Simulation geometry
m_total = 3.0;
before  = 0.04;
bed     = 0.92;

nstages = Parameters{1};
r_geom  = Parameters{3};   % renamed from 'r' — avoid clash with any result struct
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

%% Bounds and normalization (temperature and flow only — pressure is fixed)
T_min  = 303;     T_max  = 313;
T_mid  = 0.5 * (T_min + T_max);
T_half = 0.5 * (T_max - T_min);

F_min  = 3.3e-5;  F_max  = 6.7e-5;
F_mid  = 0.5 * (F_min + F_max);
F_half = 0.5 * (F_max - F_min);

fprintf('Decision variable bounds:\n');
fprintf('  Temperature : [%.1f, %.1f] K\n',    T_min, T_max);
fprintf('  Flow rate   : [%.2e, %.2e] m3/s\n', F_min, F_max);
fprintf('  Pressure    : %.1f bar (fixed)\n\n', P_fixed);

%% Pre-generate all initial guesses via Latin Hypercube Sampling
n_lhs_cols = 2 * n_init_knots;

rng(0);
lhs = lhsdesign(N_seeds, n_lhs_cols, 'criterion', 'maximin', 'iterations', 20);

z0_mat        = zeros(2 * N_Time, N_seeds);
feedTemp0_all = zeros(N_Time, N_seeds);
feedFlow0_all = zeros(N_Time, N_seeds);

for s = 1:N_seeds
    rng(s);
    interior_pool = 2:(N_Time - 1);
    n_interior    = n_init_knots - 2;
    pick          = randperm(numel(interior_pool), n_interior);
    interior_idx  = sort(interior_pool(pick));
    init_knot_idx = [1, interior_idx, N_Time];

    temp_knots = T_min + (T_max - T_min) * lhs(s,                1 :   n_init_knots);
    flow_knots = F_min + (F_max - F_min) * lhs(s, n_init_knots + 1 : 2*n_init_knots);

    feedTemp_0 = interp1(init_knot_idx, temp_knots, 1:N_Time, 'linear');
    feedFlow_0 = interp1(init_knot_idx, flow_knots, 1:N_Time, 'linear');

    z0_mat(:, s) = [(feedTemp_0 - T_mid) / T_half, ...
                    (feedFlow_0 - F_mid) / F_half]';

    feedTemp0_all(:, s) = feedTemp_0(:);
    feedFlow0_all(:, s) = feedFlow_0(:);
end

fprintf('Initial guess summary across %d seeds (LHS):\n', N_seeds);
fprintf('  Temp  mean %.2f K,  std %.2f K\n',  mean(feedTemp0_all(:)), std(feedTemp0_all(:)));
fprintf('  Flow  mean %.2e,    std %.2e\n\n',   mean(feedFlow0_all(:)), std(feedFlow0_all(:)));

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
    'N_Time',             N_Time,             ...
    'P_fixed',            P_fixed,            ...
    'nlp_opts',           nlp_opts,           ...
    'k1_val',             k1_val,             ...
    'k2_val',             k2_val,             ...
    'which_k',            which_k,            ...
    'Parameters',         {Parameters},       ...
    'bed_mask',           bed_mask,           ...
    'epsi_mask',          epsi_mask,          ...
    'one_minus_epsi_mask',one_minus_epsi_mask,...
    'C0fluid',            C0fluid,            ...
    'C0solid',            C0solid,            ...
    'N_Sample',           N_Sample,           ...
    'sigma2_cases',       sigma2_cases,       ...
    'Cov_power_cum',      Cov_power_cum,      ...
    'Cov_linear_cum',     Cov_linear_cum,     ...
    'T_mid',              T_mid,              ...
    'T_half',             T_half,             ...
    'F_mid',              F_mid,              ...
    'F_half',             F_half,             ...
    'Nx',                 Nx,                 ...
    'Nu',                 Nu,                 ...
    'nstages',            nstages,            ...
    'timeStep_in_sec',    timeStep_in_sec,    ...
    'casadi_path',        casadi_path         ...
);

%% Launch all seeds asynchronously via parfeval
% FIX: replaces parfor — parfeval lets the client collect results one by
%      one as workers finish, so fprintf and save run on the client (no
%      transparency violation) and checkpoints are written immediately.
t_par = tic;
fprintf('Solving %d seeds on %d workers...\n', N_seeds, N_workers);
results = cell(1, N_seeds);

futures(1:N_seeds) = parallel.FevalFuture;
for s = 1:N_seeds
    futures(s) = parfeval(pool, @solve_one_seed, 1, ...
        s, z0_mat(:,s), feedTemp0_all(:,s), feedFlow0_all(:,s), fixed);
end

%% Collect results — runs on client so fprintf/save are fine
for i = 1:N_seeds
    [~, res] = fetchNext(futures);   % blocks until any worker finishes

    results{res.seed} = res;

    % Checkpoint: save on client, not inside worker
    save(fullfile(checkpoint_dir, sprintf('seed_%03d.mat', res.seed)), 'res');

    fprintf('[%2d/%2d] seed=%3d | ok=%d | j0=%.4e → j=%.4e | %.1f min\n', ...
        i, N_seeds, res.seed, res.success, res.j_initial, res.j, res.elapsed_min);

    if ~res.success && ~isempty(res.error_msg)
        fprintf('         error: %s\n', res.error_msg);
    end
end

fprintf('\nSolve complete: %.1f s total\n\n', toc(t_par));

%% Identify best seed and save
j_vals             = cellfun(@(x) x.j, results);
[j_best, idx_best] = max(j_vals);
best               = results{idx_best};
fprintf('Best seed: %d | j = %.6e\n\n', best.seed, j_best);

% FIX: was num2str(P_var) — P_var never defined; use P_fixed
save([num2str(P_fixed), '_results_single_pressure.mat'], 'results', 'best', 'P_fixed');
fprintf('Done.\n');
datetime

%% Reload from checkpoints (uncomment to rebuild results from disk)
%{
for s = 1:N_seeds
    fname = fullfile(checkpoint_dir, sprintf('seed_%03d.mat', s));
    if isfile(fname)
        tmp = load(fname);
        if ~isempty(tmp.res.feedTemp)
            results{s} = tmp.res;   % FIX: was tmp.r — field is 'res'
        end
    end
end
%}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Worker function — no filesystem or console I/O allowed here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function res = solve_one_seed(s, z0_s, feedTemp0_s, feedFlow0_s, fixed)

    addpath(fixed.casadi_path);

    t0     = tic;
    N_Time = fixed.N_Time;
    n_samp = numel(fixed.N_Sample);

    res = struct( ...
        'seed',        s,              ...
        'success',     false,          ...
        'j_initial',   NaN,            ...
        'j',           NaN,            ...
        'j_1',         NaN,            ...
        'j_2',         NaN,            ...
        'feedTemp',    NaN(1, N_Time), ...
        'feedFlow',    NaN(1, N_Time), ...
        'feedPress',   fixed.P_fixed,  ...
        'feedTemp0',   feedTemp0_s',   ...
        'feedFlow0',   feedFlow0_s',   ...
        'Y_cum_P',     NaN(1, n_samp), ...   % sized to sample times
        'Y_cum_L',     NaN(1, n_samp), ...
        'elapsed_min', NaN,            ...
        'error_msg',   '');

    try
        import casadi.*

        opti = casadi.Opti();
        opti.solver('ipopt', fixed.nlp_opts);

        k1 = opti.parameter(4);
        k2 = opti.parameter(6);
        k  = [k1; k2];

        zFeedTemp = opti.variable(1, N_Time);
        zFeedFlow = opti.variable(1, N_Time);

        feedTemp = fixed.T_mid + fixed.T_half * zFeedTemp;
        feedFlow = fixed.F_mid + fixed.F_half * zFeedFlow;

        uu = [feedTemp', fixed.P_fixed * ones(N_Time, 1), feedFlow'];

        T  = feedTemp(1);
        P  = MX(fixed.P_fixed);

        Z            = Compressibility(T, P, fixed.Parameters);
        rho          = rhoPB_Comp(T, P, Z, fixed.Parameters);
        enthalpy_rho = rho * SpecificEnthalpy(T, P, Z, rho, fixed.Parameters);

        x0 = [fixed.C0fluid'; fixed.C0solid * fixed.bed_mask; ...
              (enthalpy_rho / 1e4) * ones(fixed.nstages, 1); P; 0];

        Parameters_sym               = MX(cell2mat(fixed.Parameters));
        Parameters_sym(fixed.which_k) = k(1:numel(fixed.which_k));
        U_base                        = [uu'; repmat(Parameters_sym, 1, N_Time)];

        f_power  = @(x, u) modelSFE(x, u, fixed.bed_mask, fixed.timeStep_in_sec, ...
            'Power_model',  fixed.epsi_mask, fixed.one_minus_epsi_mask);
        f_linear = @(x, u) modelSFE(x, u, fixed.bed_mask, fixed.timeStep_in_sec, ...
            'Linear_model', fixed.epsi_mask, fixed.one_minus_epsi_mask);

        F_accum_power  = buildIntegrator(f_power,  [fixed.Nx, fixed.Nu], fixed.timeStep_in_sec, 'cvodes') ...
                             .mapaccum('F_accum_power',  N_Time);
        F_accum_linear = buildIntegrator(f_linear, [fixed.Nx, fixed.Nu], fixed.timeStep_in_sec, 'cvodes') ...
                             .mapaccum('F_accum_linear', N_Time);

        X_power_nom  = F_accum_power(x0,  U_base);
        X_linear_nom = F_accum_linear(x0, U_base);

        % Prepend t=0 then subset to sample times
        Y_cum_P_sym = [0, X_power_nom(fixed.Nx,  :)];
        Y_cum_P_sym = Y_cum_P_sym(fixed.N_Sample);
        Y_cum_L_sym = [0, X_linear_nom(fixed.Nx, :)];
        Y_cum_L_sym = Y_cum_L_sym(fixed.N_Sample);

        residuals = Y_cum_P_sym - Y_cum_L_sym;

        J_P_sym = jacobian(Y_cum_P_sym, k1);
        J_L_sym = jacobian(Y_cum_L_sym, k2);

        I = MX.eye(n_samp);

        Sigma_r_P = fixed.sigma2_cases(1) * I + J_P_sym * fixed.Cov_power_cum  * J_P_sym';
        Sigma_r_L = fixed.sigma2_cases(1) * I + J_L_sym * fixed.Cov_linear_cum * J_L_sym';
        Sigma_r_P = Sigma_r_P + 1e-10 * I;
        Sigma_r_L = Sigma_r_L + 1e-10 * I;

        j_1 = trace( Sigma_r_P * (Sigma_r_L\I) + Sigma_r_L * (Sigma_r_P\I) - 2*I );
        j_2 = residuals * ((Sigma_r_P\I) + (Sigma_r_L\I)) * residuals';
        j   = j_1 + j_2;

        opti.subject_to(zFeedTemp >= -1);
        opti.subject_to(zFeedTemp <=  1);
        opti.subject_to(zFeedFlow >= -1);
        opti.subject_to(zFeedFlow <=  1);

        opti.set_value(k1, fixed.k1_val);
        opti.set_value(k2, fixed.k2_val);

        alpha    = 1e-2;
        beta     = 1e-2;
        j_smooth = alpha * sum(diff(zFeedTemp, 1, 2).^2) + ...
                   beta  * sum(diff(zFeedFlow, 1, 2).^2);

        opti.minimize(-(j - j_smooth));

        zFeedTemp_0 = reshape(z0_s(1:N_Time),          1, N_Time);
        zFeedFlow_0 = reshape(z0_s(N_Time+1:2*N_Time), 1, N_Time);

        opti.set_initial(zFeedTemp, zFeedTemp_0);
        opti.set_initial(zFeedFlow, zFeedFlow_0);

        %% Solve
        try
            sol         = opti.solve();
            valfun      = @(x) sol.value(x);
            res.success = true;
        catch solve_err
            valfun      = @(x) opti.debug.value(x);
            res.success  = false;
            res.error_msg = solve_err.message;
        end

        try
            stats = opti.stats();
            if isstruct(stats) && isfield(stats, 'iterations') && isfield(stats.iterations, 'obj')
                res.j_initial = -stats.iterations.obj(1);
            end
        catch
            % stats unavailable — j_initial stays NaN
        end

        res.j_1      = full(valfun(j_1));
        res.j_2      = full(valfun(j_2));
        res.j        = full(valfun(j));        % FIX: consistent with CasADi eval
        res.feedTemp = full(valfun(feedTemp));
        res.feedFlow = full(valfun(feedFlow));
        res.Y_cum_P  = full(valfun(Y_cum_P_sym));   % n_samp values — matches pre-alloc
        res.Y_cum_L  = full(valfun(Y_cum_L_sym));

    catch outer_err
        res.success   = false;
        res.error_msg = outer_err.message;
    end

    res.elapsed_min = toc(t0) / 60;
end