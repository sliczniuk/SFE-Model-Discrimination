%% Initialization
startup;

fprintf('=============================================================================\n');
fprintf('   Optimal trajectory for model discrimination (parfeval batching / nlpsol)\n');
fprintf('=============================================================================\n\n');

%% Run configuration
N_seeds      = 36;
N_workers    = 6;
n_init_knots = 40;
if ~exist('P_var', 'var'); P_var = 150; end   % default; override: -r "P_var=250; test_7; exit"
casadi_path  = 'C:\Dev\casadi-3.6.3-windows64-matlab2018b';
setenv('MW_MINGW64_LOC', 'C:\ProgramData\MATLAB\SupportPackages\R2023a\3P.instrset\mingw_w64.instrset');

%% Optimizer Settings
% Plain struct — serialisable to parfeval workers; same options as test_5 + error_on_fail
nlp_opts = struct;
nlp_opts.ipopt.max_iter              = 50;
nlp_opts.ipopt.tol                   = 1e-7;
nlp_opts.ipopt.acceptable_tol        = 1e-5;
nlp_opts.ipopt.acceptable_iter       = 10;
nlp_opts.ipopt.hessian_approximation = 'limited-memory';
nlp_opts.ipopt.limited_memory_max_history = 20;   % default: 5
nlp_opts.ipopt.print_level           = 0;   % suppress per-worker IPOPT output
nlp_opts.print_time                  = 0;   % suppress CasADi solver timing table
nlp_opts.error_on_fail               = false; % return last iterate on IPOPT failure, never throw

fprintf('=== Run Configuration ===\n');
fprintf('Seeds: %d | Workers: %d | Knots: %d | Max iter: %d\n\n', ...
    N_seeds, N_workers, n_init_knots, nlp_opts.ipopt.max_iter);

%% Load parameters and data  (numeric only — no CasADi)
addpath(casadi_path);

Parameters_table = readtable('Parameters.csv');
Parameters       = num2cell(Parameters_table{:,3});
LabResults       = xlsread('dataset_2.xlsx');

which_k = (0:9) + 44;
k1_val  = cell2mat(Parameters((0:3) + 44));
k2_val  = cell2mat(Parameters((4:9) + 44));

%% Time grid and sample matching
timeStep  =  15;
finalTime = 600;

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

%% Covariance matrices  (numeric)
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

%% Simulation geometry  (numeric)
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
Parameters{2} = C0solid;              % update cell array before bundling into cfg

m_fluid = zeros(1, nstages);
C0fluid = m_fluid * 1e-3 ./ V_fluid';

epsi_mask           = epsi .* bed_mask;
one_minus_epsi_mask = 1 - epsi_mask;

Nx = 3 * nstages + 2;
Nu = 3 + numel(Parameters);

%% Bounds and normalization  (numeric scalars)
feedPress = P_var * ones(1, N_Time);

T_min  = 303;    T_max  = 313;
F_min  = 3.3e-5; F_max  = 6.7e-5;

T_mid  = 0.5 * (T_min + T_max);
T_half = 0.5 * (T_max - T_min);
F_mid  = 0.5 * (F_min + F_max);
F_half = 0.5 * (F_max - F_min);

%% Pre-generate all initial guesses  (plain numeric loop — no parfor)
z0_mat        = zeros(2 * N_Time, N_seeds);
feedTemp0_all = zeros(N_Time, N_seeds);
feedFlow0_all = zeros(N_Time, N_seeds);

for s = 1:N_seeds
    rng(s);
    interior_pool = 2:(N_Time-1);
    n_interior    = n_init_knots - 2;
    pick          = randperm(numel(interior_pool), n_interior);
    interior_idx  = sort(interior_pool(pick));
    init_knot_idx = [1, interior_idx, N_Time];
    temp_knots    = T_min + (T_max - T_min) * rand(1, n_init_knots);
    flow_knots    = F_min + (F_max - F_min) * rand(1, n_init_knots);
    feedTemp_0    = interp1(init_knot_idx, temp_knots, 1:N_Time, 'linear');
    feedFlow_0    = interp1(init_knot_idx, flow_knots, 1:N_Time, 'linear');

    z0_mat(:, s)         = [(feedTemp_0 - T_mid) / T_half, ...
                             (feedFlow_0 - F_mid) / F_half]';
    feedTemp0_all(:, s)  = feedTemp_0(:);
    feedFlow0_all(:, s)  = feedFlow_0(:);
end

%% Bundle all problem data into cfg struct
% Plain serialisable struct — safe to transmit across parfeval worker boundary.
cfg = struct;
cfg.casadi_path         = casadi_path;
cfg.feedPress           = feedPress;           % 1×N_Time numeric
cfg.nlp_opts            = nlp_opts;
cfg.k1_val              = k1_val;
cfg.k2_val              = k2_val;
cfg.N_Time              = N_Time;
cfg.Nx                  = Nx;
cfg.Nu                  = Nu;
cfg.timeStep_in_sec     = timeStep_in_sec;
cfg.T_mid               = T_mid;
cfg.T_half              = T_half;
cfg.F_mid               = F_mid;
cfg.F_half              = F_half;
cfg.bed_mask            = bed_mask;
cfg.epsi_mask           = epsi_mask;
cfg.one_minus_epsi_mask = one_minus_epsi_mask;
cfg.C0fluid             = C0fluid;
cfg.C0solid             = C0solid;
cfg.nstages             = nstages;
cfg.which_k             = which_k;
cfg.Parameters          = Parameters;          % cell array, already updated with C0solid
cfg.N_Sample            = N_Sample;
cfg.sigma2_cases        = sigma2_cases;
cfg.Cov_power_cum       = Cov_power_cum;
cfg.Cov_linear_cum      = Cov_linear_cum;
cfg.alpha               = 1e-2;               % smoothness weight — feedTemp
cfg.beta                = 1e-2;               % smoothness weight — feedFlow

%% Parallel pool
pool = gcp('nocreate');
if isempty(pool)
    parpool('local', N_workers);
end
% Ensure CasADi is on all worker paths before dispatching parfeval tasks
parfevalOnAll(@() addpath(casadi_path), 0);

%% Parallel multistart — parfeval batching
% Each worker builds the nlpsol Function once, then loops over its assigned seeds.
% DataQueue delivers per-seed progress to the client in real time.
clear local_show_progress;
dq = parallel.pool.DataQueue;
afterEach(dq, @(data) local_show_progress(data, N_seeds));

t_par = tic;
fprintf('Solving %d seeds on %d workers (parfeval batching / nlpsol)...\n', N_seeds, N_workers);
results = cell(1, N_seeds);

seeds_per_worker = ceil(N_seeds / N_workers);
n_fut = 0;
for w = 1:N_workers
    batch = ((w-1)*seeds_per_worker + 1) : min(w*seeds_per_worker, N_seeds);
    if isempty(batch); continue; end
    n_fut = n_fut + 1;
    futures(n_fut) = parfeval(@solve_seed_batch, 1, ...   %#ok<AGROW>
        batch, z0_mat(:, batch), feedTemp0_all(:, batch), ...
        feedFlow0_all(:, batch), cfg, dq);
end

% Collect results in completion order (DataQueue already handles real-time display)
for idx = 1:n_fut
    [~, batch_results] = fetchNext(futures);
    for b = 1:numel(batch_results)
        s = batch_results{b}.seed;
        results{s} = batch_results{b};
    end
end

fprintf('\nParallel solve complete: %.1f s total\n\n', toc(t_par));

%% Save results  (before plot — guards against AWT/display errors)
save([num2str(P_var),'_bar_nlpsol_results.mat'], 'results');
fprintf('Results saved to %d_bar_nlpsol_results.mat\n', P_var);

%% Select best result
%{
results_arr = [results{:}];

fprintf('\n=== Per-seed summary ===\n');
for s = 1:N_seeds
    fprintf('seed=%3d | success=%d | j0=%.6e | j=%.6e\n', ...
        results_arr(s).seed, results_arr(s).success, results_arr(s).j_initial, results_arr(s).j);
end

j_all  = [results_arr.j];
valid  = isfinite(j_all);
if ~any(valid)
    error('All seeds returned non-finite objectives.');
end
j_candidates         = j_all;
j_candidates(~valid) = -Inf;
[~, best_idx]        = max(j_candidates);

fprintf('\nBest seed: %d | j=%.6e\n', best_idx, j_all(best_idx));

best           = results_arr(best_idx);
feedTemp0_best = best.feedTemp0;
feedFlow0_best = best.feedFlow0;
K_best         = [best.feedTemp; best.feedFlow];

%% Plot best result
figure;
subplot(2,1,1)
hold on
stairs(Time, [feedTemp0_best, feedTemp0_best(end)] - 273, LineWidth=2)
stairs(Time, [K_best(1,:),    K_best(1,end)]       - 273, LineWidth=2)
hold off
legend('Initial (best seed)', 'Optimised')
xlabel('Time [min]')
ylabel('T [°C]')
title(sprintf('Best seed: %d | j = %.4e', best_idx, j_all(best_idx)))

subplot(2,1,2)
hold on
stairs(Time, [feedFlow0_best, feedFlow0_best(end)], LineWidth=2)
stairs(Time, [K_best(2,:),    K_best(2,end)],       LineWidth=2)
hold off
legend('Initial (best seed)', 'Optimised')
xlabel('Time [min]')
ylabel('F [kg/s]')
%}

%% -----------------------------------------------------------------------
function local_show_progress(data, N_total)
%LOCAL_SHOW_PROGRESS  DataQueue callback — real-time per-seed progress with ETA.
% Persistent state is reset by  clear local_show_progress  before the loop.
persistent n t_start;
if isempty(n)
    n       = 0;
    t_start = posixtime(datetime('now'));
end
n = n + 1;

s         = data(1);
success   = logical(data(2));
j_initial = data(3);
j_val     = data(4);
t_seed    = data(5);                                   % per-seed wall time [s]

elapsed = posixtime(datetime('now')) - t_start;        % wall clock since first send [s]
eta     = (elapsed / n) * (N_total - n);               % remaining wall time [s]

if eta >= 60
    eta_str = sprintf('%.1f min', eta / 60);
else
    eta_str = sprintf('%.0f s', eta);
end

fprintf('[%2d/%2d] seed=%3d | ok=%d | j0=%.4e → j=%.4e | %.1f min | ETA %s\n', ...
    n, N_total, s, success, j_initial, j_val, t_seed / 60, eta_str);
end

%% -----------------------------------------------------------------------
function batch_results = solve_seed_batch( ...
        batch, z0_batch, feedTemp0_batch, feedFlow0_batch, cfg, dq)
%SOLVE_SEED_BATCH  Build CasADi NLP once with nlpsol, then solve each seed in batch.
%
%   batch            — 1×B vector of seed indices
%   z0_batch         — (2*N_Time)×B normalised initial guesses [column per seed]
%   feedTemp0_batch  — N_Time×B physical feedTemp initial guesses [K]
%   feedFlow0_batch  — N_Time×B physical feedFlow initial guesses [m³/s]
%   cfg              — struct with all problem data (see main script)
%   dq               — parallel.pool.DataQueue for real-time progress reporting

% Ensure CasADi is on the worker path (defensive; parfevalOnAll also does this)
addpath(cfg.casadi_path);
import casadi.*

%% Unpack cfg
feedPress           = cfg.feedPress;
nlp_opts            = cfg.nlp_opts;
k1_val              = cfg.k1_val;
k2_val              = cfg.k2_val;
N_Time              = cfg.N_Time;
Nx                  = cfg.Nx;
Nu                  = cfg.Nu;           %#ok<NASGU>  used via buildIntegrator [Nx,Nu]
timeStep_in_sec     = cfg.timeStep_in_sec;
T_mid               = cfg.T_mid;
T_half              = cfg.T_half;
F_mid               = cfg.F_mid;
F_half              = cfg.F_half;
bed_mask            = cfg.bed_mask;
epsi_mask           = cfg.epsi_mask;
one_minus_epsi_mask = cfg.one_minus_epsi_mask;
C0fluid             = cfg.C0fluid;
C0solid             = cfg.C0solid;
nstages             = cfg.nstages;
which_k             = cfg.which_k;
Parameters          = cfg.Parameters;
N_Sample            = cfg.N_Sample;
sigma2_cases        = cfg.sigma2_cases;
Cov_power_cum       = cfg.Cov_power_cum;
Cov_linear_cum      = cfg.Cov_linear_cum;
alpha               = cfg.alpha;
beta                = cfg.beta;

%% Build symbolic NLP (once per worker — reused for all seeds in this batch)

% Decision variable: w = [zFeedTemp (N_Time×1); zFeedFlow (N_Time×1)]  (stacked column)
w = MX.sym('w', 2*N_Time, 1);
zFeedTemp = w(1:N_Time)';            % 1×N_Time row
zFeedFlow = w(N_Time+1:2*N_Time)';   % 1×N_Time row

feedTemp = T_mid + T_half * zFeedTemp;   % 1×N_Time  [K]
feedFlow = F_mid + F_half * zFeedFlow;   % 1×N_Time  [m³/s]

% NLP parameters — declared as separate pure symbolic variables so that
% jacobian(expr, k1) and jacobian(expr, k2) satisfy CasADi's requirement
% that the differentiation variable is a pure MX::Sym leaf node.
% (slices of a larger MX.sym are NOT pure symbolic and cause an error)
k1 = MX.sym('k1', 4, 1);   % power model parameters
k2 = MX.sym('k2', 6, 1);   % linear model parameters
k  = [k1; k2];              % 10×1 concatenation used in Parameters_sym substitution

% Symbolic initial state (enthalpy depends on feedTemp(1) — same as test_5.m)
T  = feedTemp(1);
P  = feedPress(1);
Z            = Compressibility(T, P, Parameters);
rho          = rhoPB_Comp(T, P, Z, Parameters);
enthalpy_rho = rho * SpecificEnthalpy(T, P, Z, rho, Parameters);
x0 = [C0fluid'; C0solid * bed_mask; enthalpy_rho * ones(nstages, 1); P; 0];

% Symbolic control input matrix  (Nu × N_Time)
uu             = [feedTemp', feedPress', feedFlow'];   % N_Time×3  (feedPress is numeric)
Parameters_sym          = MX(cell2mat(Parameters));
Parameters_sym(which_k) = k(1:numel(which_k));
U_base = [uu'; repmat(Parameters_sym, 1, N_Time)];    % Nu×N_Time

% Build integrators and mapaccum chains (identical to test_5.m parfor body)
f_power_nom       = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, ...
    'Power_model', epsi_mask, one_minus_epsi_mask);
F_power_nom       = buildIntegrator(f_power_nom, [Nx, Nu], timeStep_in_sec, 'cvodes');
F_accum_power_nom = F_power_nom.mapaccum('F_accum_power', N_Time);

f_linear_nom       = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, ...
    'Linear_model', epsi_mask, one_minus_epsi_mask);
F_linear_nom       = buildIntegrator(f_linear_nom, [Nx, Nu], timeStep_in_sec, 'cvodes');
F_accum_linear_nom = F_linear_nom.mapaccum('F_accum_linear', N_Time);

X_power_nom  = F_accum_power_nom(x0, U_base);    % Nx × N_Time
X_linear_nom = F_accum_linear_nom(x0, U_base);   % Nx × N_Time

% Cumulative yield at sample times
Y_cum_P_sym = [0, X_power_nom(Nx,  :)];
Y_cum_P_sym = Y_cum_P_sym(N_Sample);
Y_cum_L_sym = [0, X_linear_nom(Nx, :)];
Y_cum_L_sym = Y_cum_L_sym(N_Sample);

% Sensitivity matrices (Jacobians w.r.t. model parameters)
J_P_sym   = jacobian(Y_cum_P_sym, k1);        % n_samp × 4
J_L_sym   = jacobian(Y_cum_L_sym, k2);        % n_samp × 6
residuals = Y_cum_P_sym - Y_cum_L_sym;        % 1 × n_samp

% T-optimality criterion (expected prediction divergence)
n_samp    = numel(Y_cum_P_sym);
I         = MX.eye(n_samp);

Sigma_r_P = sigma2_cases(1) * I + J_P_sym * Cov_power_cum  * J_P_sym';
Sigma_r_L = sigma2_cases(1) * I + J_L_sym * Cov_linear_cum * J_L_sym';
eps_reg   = 1e-10;
Sigma_r_P = Sigma_r_P + eps_reg * I;
Sigma_r_L = Sigma_r_L + eps_reg * I;

j_2 = residuals * ((Sigma_r_P\I) + (Sigma_r_L\I)) * residuals';
j   = j_2;   % scalar discriminability criterion

% Smoothness regularization (penalises rapid control changes)
j_smooth = alpha * sum(diff(zFeedTemp, 1, 2).^2) + ...
           beta  * sum(diff(zFeedFlow, 1, 2).^2);

% NLP objective: minimise -(j - j_smooth)  ⟺  maximise (j - j_smooth)
f_obj = -(j - j_smooth);

% Box constraints: zFeedTemp, zFeedFlow ∈ [−1, 1]
lbw = -ones(2*N_Time, 1);
ubw =  ones(2*N_Time, 1);

% Compile low-level nlpsol (built once, called once per seed below)
% p field = vertcat(k1, k2) — CasADi sees a 10-element parameter vector
nlp_struct = struct('x', w, 'f', f_obj, 'p', [k1; k2]);
solver     = nlpsol('solver', 'ipopt', nlp_struct, nlp_opts);

% Auxiliary: evaluate pure discriminability j (no smoothness offset) at any (w, k1, k2)
% Used to report a clean j value comparable to test_5.m output
j_fn = Function('j_eval', {w, k1, k2}, {j});

% Nominal parameter vector — identical for all seeds in this batch
p_val = [k1_val(:); k2_val(:)];   % 10×1

%% Per-seed loop — cold-start solve for each seed in the batch
n_batch      = numel(batch);
batch_results = cell(1, n_batch);

for b = 1:n_batch
    t_seed_start = tic;
    s = batch(b);

    % Default failure result — kept if outer try/catch fires
    r = struct('seed', s, 'success', false, 'j_initial', NaN, 'j', NaN, ...
        'feedTemp',  NaN(1, N_Time), 'feedFlow',  NaN(1, N_Time), ...
        'feedTemp0', feedTemp0_batch(:, b)', ...
        'feedFlow0', feedFlow0_batch(:, b)');

    try
        w0 = z0_batch(:, b);   % (2*N_Time)×1 initial guess

        % Solve — cold-start; no state from previous seed (nlpsol has no internal state)
        sol   = solver('x0', w0, 'p', p_val, 'lbx', lbw, 'ubx', ubw);
        stats = solver.stats();

        r.success = stats.success;

        % j at iteration 0 (negate: f_obj = -(j - j_smooth), so obj(1) < 0 when j > 0)
        if isstruct(stats) && isfield(stats, 'iterations') && ...
                isfield(stats.iterations, 'obj') && ~isempty(stats.iterations.obj)
            r.j_initial = -stats.iterations.obj(1);
        end

        % Extract optimal controls and pure discriminability
        w_opt      = full(sol.x);                                     % (2*N_Time)×1
        r.j        = full(j_fn(w_opt, k1_val(:), k2_val(:)));        % scalar, no smoothness offset
        r.feedTemp = T_mid + T_half * w_opt(1:N_Time)';      % 1×N_Time  [K]
        r.feedFlow = F_mid + F_half * w_opt(N_Time+1:2*N_Time)'; % 1×N_Time  [m³/s]

    catch  %#ok<CTCH>  outer catch — worker error, r keeps NaN defaults
    end

    send(dq, [double(s), double(r.success), r.j_initial, r.j, toc(t_seed_start)]);
    batch_results{b} = r;
end
end
