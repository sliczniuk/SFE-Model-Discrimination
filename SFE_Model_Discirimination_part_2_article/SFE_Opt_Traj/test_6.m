%% Initialization
startup;

fprintf('=============================================================================\n');
fprintf('   Optimal trajectory for model discrimination (multiple shooting, parfor)\n');
fprintf('=============================================================================\n\n');

%% Run configuration
N_seeds      = 6;
N_workers    = 6;
n_init_knots = 40;
if ~exist('P_var', 'var'); P_var = 200; end   % default; override from CLI: -r "P_var=250; test_6; exit"
casadi_path  = 'C:\Dev\casadi-3.6.3-windows64-matlab2018b';

%% Optimizer Settings
% Plain struct — no CasADi objects; serializable to parfor workers
nlp_opts = struct;
nlp_opts.ipopt.max_iter              = 10;
nlp_opts.ipopt.tol                   = 1e-7;
nlp_opts.ipopt.acceptable_tol        = 1e-5;
nlp_opts.ipopt.acceptable_iter       = 10;
nlp_opts.ipopt.hessian_approximation = 'limited-memory';
nlp_opts.ipopt.print_level           = 0;   % suppress per-worker IPOPT output
nlp_opts.print_time                  = 0;   % suppress CasADi solver timing table

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
Parameters{2} = C0solid;              % update cell array before broadcasting

m_fluid = zeros(1, nstages);
C0fluid = m_fluid * 1e-3 ./ V_fluid';

epsi_mask           = epsi .* bed_mask;
one_minus_epsi_mask = 1 - epsi_mask;

Nx = 3 * nstages + 2;
Nu = 3 + numel(Parameters);

%% Bounds and normalization  (numeric scalars)
feedPress = P_var * ones(1, N_Time);

T_min  = 303;   T_max  = 313;
F_min  = 3.3e-5; F_max = 6.7e-5;

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

%% Parallel pool
pool = gcp('nocreate');
if isempty(pool)
    parpool('local', N_workers);
end
% Add CasADi to all workers once before the loop (headless-safe)
parfevalOnAll(@() addpath(casadi_path), 0);

%% Parallel multistart
% DataQueue delivers per-seed results to the client in real time.
% clear() resets the persistent progress counter before each run.
clear local_show_progress;
dq = parallel.pool.DataQueue;
afterEach(dq, @(data) local_show_progress(data, N_seeds));

t_par = tic;
fprintf('Solving %d seeds on %d workers...\n', N_seeds, N_workers);
results = cell(1, N_seeds);

parfor s = 1:N_seeds
    t_seed_start = tic;

    % Default failure result — written if the outer try/catch fires
    r = struct('seed', s, 'success', false, 'j_initial', NaN, 'j', NaN, ...
        'feedTemp',  NaN(1, N_Time), 'feedFlow',  NaN(1, N_Time), ...
        'feedTemp0', feedTemp0_all(:, s)', 'feedFlow0', feedFlow0_all(:, s)');

    try
        %% --- Step 0: Extract initial guess and compute numeric x0 ---
        % Extract control guess in physical units (needed before CasADi build
        % to forward-simulate the state trajectory for the initial guess).
        z0_s        = z0_mat(:, s);
        zFeedTemp_0 = reshape(z0_s(1:N_Time),     1, N_Time);
        zFeedFlow_0 = reshape(z0_s(N_Time+1:end), 1, N_Time);
        feedTemp_0  = T_mid + T_half * zFeedTemp_0;   % [K]
        feedFlow_0  = F_mid + F_half * zFeedFlow_0;   % [m^3/s]

        import casadi.*

        % Numeric x0: evaluate thermodynamics at the initial control guess T(1)
        T_init   = feedTemp_0(1);
        P_init   = feedPress(1);
        Z_init   = full(Compressibility(T_init, P_init, Parameters));
        rho_init = full(rhoPB_Comp(T_init, P_init, Z_init, Parameters));
        h_init   = rho_init * full(SpecificEnthalpy(T_init, P_init, Z_init, rho_init, Parameters));
        x0_num   = [C0fluid'; C0solid * bed_mask; h_init * ones(nstages, 1); P_init; 0];

        %% --- Build CasADi problem ---
        opti = casadi.Opti();
        opti.solver('ipopt', nlp_opts);

        k1 = opti.parameter(4);
        k2 = opti.parameter(6);
        k  = [k1; k2];

        zFeedTemp = opti.variable(1, N_Time);
        zFeedFlow = opti.variable(1, N_Time);

        feedTemp = T_mid + T_half * zFeedTemp;
        feedFlow = F_mid + F_half * zFeedFlow;

        T  = feedTemp(1);
        P  = feedPress(1);
        F  = feedFlow(1);
        uu = [feedTemp', feedPress', feedFlow'];

        Z            = Compressibility(T, P, Parameters);
        rho          = rhoPB_Comp(T, P, Z, Parameters);
        enthalpy_rho = rho * SpecificEnthalpy(T, P, Z, rho, Parameters);

        % Symbolic x0: first state node — enthalpy depends on symbolic feedTemp(1)
        x0 = [C0fluid'; C0solid * bed_mask; enthalpy_rho * ones(nstages, 1); P; 0];

        Parameters_sym          = MX(cell2mat(Parameters));
        Parameters_sym(which_k) = k(1:numel(which_k));
        U_base                  = [uu'; repmat(Parameters_sym, 1, N_Time)];

        %% --- Step 1: Build integrators with map (not mapaccum) ---
        % F.map(N) evaluates y_k = F(x_k, u_k) for all k independently (no chain).
        % F.mapaccum(N) chains x_{k+1} = F(x_k, u_k) — used in single shooting.
        f_power_nom     = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, ...
            'Power_model', epsi_mask, one_minus_epsi_mask);
        F_power_nom     = buildIntegrator(f_power_nom, [Nx, Nu], timeStep_in_sec, 'cvodes');
        F_mapped_power  = F_power_nom.map(N_Time);

        f_linear_nom    = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, ...
            'Linear_model', epsi_mask, one_minus_epsi_mask);
        F_linear_nom    = buildIntegrator(f_linear_nom, [Nx, Nu], timeStep_in_sec, 'cvodes');
        F_mapped_linear = F_linear_nom.map(N_Time);

        %% --- Step 2: State decision variables (Nx × N_Time each) ---
        % Each column X_nodes(:,k) = state at the START of interval k.
        X_power_nodes  = opti.variable(Nx, N_Time);
        X_linear_nodes = opti.variable(Nx, N_Time);

        %% --- Step 3: Evaluate integrator at each node independently ---
        % X_ends(:,k) = F( X_nodes(:,k), U_base(:,k) ) — end state of interval k.
        % Sensitivity spans only one 5-min interval (not the full 120-step chain).
        X_power_ends  = F_mapped_power( X_power_nodes,  U_base);   % Nx × N_Time
        X_linear_ends = F_mapped_linear(X_linear_nodes, U_base);   % Nx × N_Time

        %% --- Step 4: Initial condition + continuity constraints ---
        % Pin first node to symbolic initial state.
        % (x0 is symbolic because enthalpy depends on the decision variable feedTemp(1))
        opti.subject_to(X_power_nodes(:, 1)      == x0);
        opti.subject_to(X_linear_nodes(:, 1)     == x0);

        % Continuity: end of interval k must equal start of interval k+1
        opti.subject_to(X_power_nodes(:, 2:end)  == X_power_ends(:, 1:end-1));
        opti.subject_to(X_linear_nodes(:, 2:end) == X_linear_ends(:, 1:end-1));

        %% --- Step 5: Output extraction (unchanged from single shooting) ---
        % X_ends(Nx,:) = cumulative yield at end of each interval; prepend 0 for t=0
        Y_cum_P_sym = [0, X_power_ends(Nx, :)];
        Y_cum_P_sym = Y_cum_P_sym(N_Sample);

        Y_cum_L_sym = [0, X_linear_ends(Nx, :)];
        Y_cum_L_sym = Y_cum_L_sym(N_Sample);

        Y_diff_P_sym = Y_cum_P_sym(2:end) - Y_cum_P_sym(1:end-1);
        Y_diff_L_sym = Y_cum_L_sym(2:end) - Y_cum_L_sym(1:end-1);

        Y_P_sym = Y_cum_P_sym;
        Y_L_sym = Y_cum_L_sym;

        J_P_sym   = jacobian(Y_P_sym, k1);
        J_L_sym   = jacobian(Y_L_sym, k2);
        residuals = Y_P_sym - Y_L_sym;

        n_samp    = numel(Y_P_sym);
        I         = MX.eye(n_samp);

        Sigma_r_P = sigma2_cases(1) * I + J_P_sym * Cov_power_cum  * J_P_sym';
        Sigma_r_L = sigma2_cases(1) * I + J_L_sym * Cov_linear_cum * J_L_sym';
        eps_reg   = 1e-10;
        Sigma_r_P = Sigma_r_P + eps_reg * I;
        Sigma_r_L = Sigma_r_L + eps_reg * I;

        j_1 = trace( Sigma_r_P * (Sigma_r_L\I) + Sigma_r_L * (Sigma_r_P\I) - 2*I );
        j_2 = residuals * ((Sigma_r_P\I) + (Sigma_r_L\I)) * residuals';
        %j  = j_1 + j_2;
        j  = residuals(end).^2;
        j  = j * 1e3;

        opti.subject_to(zFeedTemp >= -1);
        opti.subject_to(zFeedTemp <= 1);
        opti.subject_to(zFeedFlow >= -1);
        opti.subject_to(zFeedFlow <= 1);

        opti.set_value(k1, k1_val);
        opti.set_value(k2, k2_val);

        alpha = 1e-3;
        beta  = 1e-3;
        j_smooth = alpha * sum(diff(zFeedTemp, 1, 2).^2) + ...
                   beta  * sum(diff(zFeedFlow, 1, 2).^2);
        opti.minimize(-(j - j_smooth));

        %opti.minimize(-j);

        %% --- Initial guesses ---
        % Control trajectory
        opti.set_initial(zFeedTemp, zFeedTemp_0);
        opti.set_initial(zFeedFlow, zFeedFlow_0);

        %% --- Step 6: Forward-simulate to initialise state nodes ---
        % Integrate with nominal parameters and the initial control guess.
        % params_num already contains the nominal k1_val / k2_val values.
        params_num    = cell2mat(Parameters);
        X_power_init  = zeros(Nx, N_Time);
        X_linear_init = zeros(Nx, N_Time);
        x_p = x0_num;
        x_l = x0_num;
        for kk = 1:N_Time
            U_k = [feedTemp_0(kk); feedPress(kk); feedFlow_0(kk); params_num];
            x_p = full(F_power_nom( x_p, U_k));
            x_l = full(F_linear_nom(x_l, U_k));
            X_power_init(:, kk)  = x_p;
            X_linear_init(:, kk) = x_l;
        end
        opti.set_initial(X_power_nodes,  X_power_init);
        opti.set_initial(X_linear_nodes, X_linear_init);

        %% --- Solve ---
        try
            sol     = opti.solve();
            valfun  = @(x) sol.value(x);
            r.success = true;
        catch
            valfun    = @(x) opti.debug.value(x);
            r.success = false;
        end

        % Extract j_initial from iteration 0 of IPOPT stats (NLP obj = -j, so negate)
        stats = opti.stats();
        if isstruct(stats) && isfield(stats, 'iterations') && isfield(stats.iterations, 'obj')
            r.j_initial = -stats.iterations.obj(1);
        end

        K_out      = full(valfun([feedTemp; feedFlow]));
        r.j        = full(valfun(j));
        r.feedTemp = K_out(1,:);
        r.feedFlow = K_out(2,:);

    catch  %#ok<CTCH>  outer catch — worker error, r keeps NaN defaults
    end

    send(dq, [double(s), double(r.success), r.j_initial, r.j, toc(t_seed_start)]);
    results{s} = r;
end
fprintf('\nParallel solve complete: %.1f s total\n\n', toc(t_par));

%% Save results  (before plot — guards against AWT/display errors)
%save([num2str(P_var),'_bar_ms_results.mat'], 'results');
%fprintf('Results saved to %d_bar_ms_results.mat\n', P_var);

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
j_candidates        = j_all;
j_candidates(~valid) = -Inf;
[~, best_idx]       = max(j_candidates);

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
