%% Run Discrimination Analysis
% Bayesian model discrimination between Power and Linear kinetic models
% for three yield representations:
%   1. Cumulative yield
%   2. Differentiated yield
%   3. Normalised differentiated yield
%
% This version uses a Laplace-style empirical-Bayes approximation with a
% CUMULATIVE Hessian that accumulates information across experiments.
%
%   Likelihood:
%       y | theta, M ~ N( G(theta), sigma^2 I )
%
%   Empirical Gaussian prior:
%       theta | M ~ N( theta_hat, Cov_theta )
%
%   The cumulative log evidence after n experiments is:
%
%       log p(y^(1:n) | M)
%       ≈ sum_{k=1}^{n} log p(y^(k) | theta_hat, M)
%         - 0.5 * log det(Cov_theta)          [prior normalisation, fixed]
%         - 0.5 * log det(A^(n))              [cumulative Hessian penalty]
%         + d/2 * log(2*pi)                   [cancels with prior norm]
%
%   where the CUMULATIVE Hessian after n experiments is:
%
%       A^(n) = Cov_theta^{-1} + (1/sigma^2) * sum_{k=1}^{n} J^(k)' J^(k)
%
% KEY DIFFERENCE FROM PER-EXPERIMENT VERSION:
%   The Hessian A^(n) accumulates the Fisher information J'J across all
%   experiments seen so far, rather than being recomputed fresh for each
%   experiment. This is the correct treatment because each experiment
%   updates the posterior curvature, and information should accumulate
%   sequentially just as in the log-likelihood sum.
%
%   Using per-experiment Hessians treats each experiment in isolation and
%   leads to instability in the posterior probability when a single
%   experiment has an unusual Jacobian structure.
%
% IMPORTANT:
% - This is an empirical-Bayes / local Gaussian approximation.
% - Cov_power and Cov_linear are treated as empirical Gaussian priors
%   centred at the parameter estimates theta_hat.
% - The quadratic prior term vanishes at theta_hat by construction.
% - Only the prior normalisation -0.5*log|Cov_theta| and the cumulative
%   Hessian penalty -0.5*log|A^(n)| distinguish the models.

%% Initialization
startup;
addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
import casadi.*

fprintf('=============================================================================\n');
fprintf('   MODEL DISCRIMINATION (EMPIRICAL GAUSSIAN PRIOR + LAPLACE, CUMUL. HESSIAN)\n');
fprintf('=============================================================================\n\n');

%% Load parameters and data
Parameters_table = readtable('Parameters.csv');
Parameters       = num2cell(Parameters_table{:,3});
LabResults       = xlsread('dataset_2.xlsx');

which_k = (0:9) + 44;      % Indices of parameters to fit (44-53)
Nk      = numel(which_k);  % 10 parameters (4 Power + 6 Linear)
k1      = MX.sym('k1', 4);
k2      = MX.sym('k2', 6);
k       = [k1; k2];

%% Set up the simulation
timeStep  = 5;    % Time step [min]
finalTime = 600;  % Extraction time [min]
Time      = 0 : timeStep : finalTime;

%% Sample Time Matching
SAMPLE = LabResults(6:19, 1);

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

% Linear model covariance (6x6): [D_i(0), D_i(Re), D_i(F), Ups(0), Ups(Re), Ups(F)]
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

% Nominal parameter values (empirical prior means = LS estimates)
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

C0solid       = m_total * 1e-3 / (V_bed * epsi);
Parameters{2} = C0solid;

m_fluid = zeros(1, nstages);
C0fluid = m_fluid * 1e-3 ./ V_fluid';

epsi_mask           = epsi .* bed_mask;
one_minus_epsi_mask = 1 - epsi_mask;

Nx = 3 * nstages + 2;
Nu = 3 + numel(Parameters);

%% Yield case definitions
yield_cases      = {'Cumulative', 'Differentiated', 'Normalised'};
Cov_power_cases  = {Cov_power_cum,  Cov_power_diff,  Cov_power_norm};
Cov_linear_cases = {Cov_linear_cum, Cov_linear_diff, Cov_linear_norm};
sigma2_cases     = [2.45e-2, 1.386e-3, 1.007e-2];
N_cases          = numel(yield_cases);
N_exp            = 12;

dP = numel(theta_power);   % 4
dL = numel(theta_linear);  % 6

%% Storage for plots
PLinear_store  = zeros(N_cases, N_exp);
BF_P_L_store   = zeros(N_cases, N_exp);
DeltaBIC_store = zeros(N_cases, N_exp);

%% Run discrimination for each yield case
for cc = 1:N_cases
    case_name  = yield_cases{cc};
    Cov_power  = Cov_power_cases{cc};
    Cov_linear = Cov_linear_cases{cc};
    sigma2_y   = sigma2_cases(cc);

    fprintf('\n=============================================================================\n');
    fprintf('   CASE %d: %s yield\n', cc, upper(case_name));
    fprintf('=============================================================================\n\n');

    % ================================================================
    % Precision matrices (fixed across experiments)
    % ================================================================
    Prec_power  = Cov_power  \ eye(size(Cov_power));
    Prec_linear = Cov_linear \ eye(size(Cov_linear));

    % ================================================================
    % Prior normalisation terms (fixed, computed once per case)
    % log p(theta_hat | M) = -0.5*log|Cov_theta| - d/2*log(2*pi)
    % ================================================================
    R_cov_P = chol(Cov_power,  'lower');
    R_cov_L = chol(Cov_linear, 'lower');

    logdetCov_P = 2 * sum(log(diag(R_cov_P)));
    logdetCov_L = 2 * sum(log(diag(R_cov_L)));

    logPriorNorm_P = -0.5 * logdetCov_P - dP/2 * log(2*pi);
    logPriorNorm_L = -0.5 * logdetCov_L - dL/2 * log(2*pi);

    % ================================================================
    % Initialise cumulative quantities
    % ================================================================
    % Cumulative Fisher information: starts at zero, accumulates J'J
    % The full cumulative Hessian is: A^(n) = Prec + (1/s2)*sum J'J
    FisherSum_P = zeros(dP, dP);   % sum_{k=1}^{n} J_P^(k)' J_P^(k)
    FisherSum_L = zeros(dL, dL);   % sum_{k=1}^{n} J_L^(k)' J_L^(k)

    % Cumulative log-likelihood
    logLik_P_cum = 0;
    logLik_L_cum = 0;

    % Per-experiment storage
    logLik_P    = zeros(1, N_exp);
    logLik_L    = zeros(1, N_exp);
    hessPen_P   = zeros(1, N_exp);   % -0.5*log|A^(n)| after experiment n
    hessPen_L   = zeros(1, N_exp);
    logEv_P     = zeros(1, N_exp);   % cumulative log evidence after n exps
    logEv_L     = zeros(1, N_exp);
    MSE_P       = zeros(1, N_exp);
    MSE_L       = zeros(1, N_exp);
    BF          = zeros(1, N_exp);
    cond_A_P    = zeros(1, N_exp);
    cond_A_L    = zeros(1, N_exp);
    rank_JP     = zeros(1, N_exp);
    rank_JL     = zeros(1, N_exp);
    n_obs_store = zeros(1, N_exp);
    BIC_P_store = zeros(1, N_exp);
    BIC_L_store = zeros(1, N_exp);

    for jj = 1:N_exp
        data_org       = LabResults(6:19, jj+1)';
        data_diff      = diff(data_org);
        max_data_diff  = max(data_diff);
        data_diff_norm = data_diff ./ max_data_diff;

        % Operating conditions
        T = LabResults(2, jj+1);
        P = LabResults(3, jj+1) * 10;
        F = LabResults(4, jj+1) * 1e-5;

        % Fluid properties
        Z            = Compressibility(T, P, Parameters);
        rho          = rhoPB_Comp(T, P, Z, Parameters);
        enthalpy_rho = rho * SpecificEnthalpy(T, P, Z, rho, Parameters);

        % Initial state
        x0 = [C0fluid'; C0solid * bed_mask; enthalpy_rho * ones(nstages, 1); P; 0];

        % Input vectors
        feedTemp  = T * ones(1, N_Time);
        feedPress = P * ones(1, N_Time);
        feedFlow  = F * ones(1, N_Time);
        uu        = [feedTemp', feedPress', feedFlow'];

        Parameters_sym          = MX(cell2mat(Parameters));
        Parameters_sym(which_k) = k(1:numel(which_k));
        U_base                  = [uu'; repmat(Parameters_sym, 1, N_Time)];

        % Build integrators
        f_power_nom = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, ...
            'Power_model', epsi_mask, one_minus_epsi_mask);
        F_power_nom = buildIntegrator(f_power_nom, [Nx, Nu], timeStep_in_sec, 'cvodes');
        F_accum_power_nom = F_power_nom.mapaccum('F_accum_power', N_Time);

        f_linear_nom = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, ...
            'Linear_model', epsi_mask, one_minus_epsi_mask);
        F_linear_nom = buildIntegrator(f_linear_nom, [Nx, Nu], timeStep_in_sec, 'cvodes');
        F_accum_linear_nom = F_linear_nom.mapaccum('F_accum_linear', N_Time);

        X_power_nom  = F_accum_power_nom(x0, U_base);
        X_linear_nom = F_accum_linear_nom(x0, U_base);

        % Extract cumulative yield at sample times
        Y_cum_P_sym = [0, X_power_nom(Nx, :)];
        Y_cum_P_sym = Y_cum_P_sym(N_Sample);

        Y_cum_L_sym = [0, X_linear_nom(Nx, :)];
        Y_cum_L_sym = Y_cum_L_sym(N_Sample);

        % Differentiated yield
        Y_diff_P_sym = Y_cum_P_sym(2:end) - Y_cum_P_sym(1:end-1);
        Y_diff_L_sym = Y_cum_L_sym(2:end) - Y_cum_L_sym(1:end-1);

        % Select symbolic output and data
        if cc == 1
            Y_P_sym  = Y_cum_P_sym;
            Y_L_sym  = Y_cum_L_sym;
            data_ref = data_org;
        elseif cc == 2
            Y_P_sym  = Y_diff_P_sym;
            Y_L_sym  = Y_diff_L_sym;
            data_ref = data_diff;
        else
            Y_P_sym  = Y_diff_P_sym ./ max_data_diff;
            Y_L_sym  = Y_diff_L_sym ./ max_data_diff;
            data_ref = data_diff_norm;
        end

        % Jacobians
        J_P_sym = jacobian(Y_P_sym, k1);
        J_L_sym = jacobian(Y_L_sym, k2);

        % CasADi evaluation at empirical prior means
        G = Function('G', {k}, {J_P_sym, J_L_sym, Y_P_sym, Y_L_sym});
        [JJ_P, JJ_L, Y_P, Y_L] = G([theta_power; theta_linear]);

        JJ_P     = full(JJ_P);
        JJ_L     = full(JJ_L);
        Y_P      = full(Y_P(:));
        Y_L      = full(Y_L(:));
        data_ref = data_ref(:);

        % Residuals
        residuals_P = Y_P - data_ref;
        residuals_L = Y_L - data_ref;

        n_obs = numel(data_ref);

        % ================================================================
        % Plain log-likelihood: y | theta, M ~ N(G(theta), sigma^2 I)
        % ================================================================
        logLik_P_j = -0.5/sigma2_y * sum(residuals_P.^2) ...
                     - n_obs/2 * log(2*pi*sigma2_y);

        logLik_L_j = -0.5/sigma2_y * sum(residuals_L.^2) ...
                     - n_obs/2 * log(2*pi*sigma2_y);

        % Accumulate log-likelihoods
        logLik_P_cum = logLik_P_cum + logLik_P_j;
        logLik_L_cum = logLik_L_cum + logLik_L_j;

        % ================================================================
        % UPDATE CUMULATIVE FISHER INFORMATION SUMS
        %
        % FisherSum^(n) = FisherSum^(n-1) + (1/sigma^2) * J^(n)' * J^(n)
        %
        % This is the key fix: information accumulates across experiments.
        % ================================================================
        FisherSum_P = FisherSum_P + (1/sigma2_y) * (JJ_P' * JJ_P);
        FisherSum_L = FisherSum_L + (1/sigma2_y) * (JJ_L' * JJ_L);

        % ================================================================
        % CUMULATIVE HESSIAN of negative log posterior
        %
        % A^(n) = Prec_theta + sum_{k=1}^{n} (1/sigma^2) J^(k)' J^(k)
        %       = Prec_theta + FisherSum^(n)
        %
        % As n grows, A^(n) becomes increasingly dominated by the data
        % term and less sensitive to any single experiment's Jacobian.
        % ================================================================
        ridge = 1e-12 * eye(dP);
        A_P_n = Prec_power  + FisherSum_P + ridge;

        ridge = 1e-12 * eye(dL);
        A_L_n = Prec_linear + FisherSum_L + ridge;

        % Cholesky of cumulative Hessian
        R_AP = chol(A_P_n, 'lower');
        R_AL = chol(A_L_n, 'lower');

        % Cumulative Hessian penalty: -0.5 * log|A^(n)|
        hessPen_P_n = -sum(log(diag(R_AP)));   % = -0.5 * log|A^(n)|
        hessPen_L_n = -sum(log(diag(R_AL)));

        % ================================================================
        % CUMULATIVE LOG EVIDENCE after n experiments
        %
        % log Ev^(n) = sum_{k=1}^{n} logLik^(k)   [cumulative likelihood]
        %            + logPriorNorm                  [fixed normalisation]
        %            + d/2 * log(2*pi)               [cancels with prior]
        %            - 0.5 * log|A^(n)|              [cumulative Hessian]
        %
        % Note: logPriorNorm already contains -d/2*log(2*pi), so
        % adding d/2*log(2*pi) gives net contribution of -0.5*log|Cov|
        % ================================================================
        logEv_P_n = logLik_P_cum + logPriorNorm_P + dP/2*log(2*pi) + hessPen_P_n;
        logEv_L_n = logLik_L_cum + logPriorNorm_L + dL/2*log(2*pi) + hessPen_L_n;

        % ================================================================
        % BIC (auxiliary, based on plain likelihood only)
        % Note: BIC uses cumulative n_obs for the log(n) penalty
        % ================================================================
        n_total_running  = sum(n_obs_store(1:jj-1)) + n_obs;
        BIC_P_n = -2 * logLik_P_cum + dP * log(n_total_running);
        BIC_L_n = -2 * logLik_L_cum + dL * log(n_total_running);

        % Store per-experiment diagnostics
        logLik_P(jj)    = logLik_P_j;
        logLik_L(jj)    = logLik_L_j;
        hessPen_P(jj)   = hessPen_P_n;   % cumulative Hessian penalty
        hessPen_L(jj)   = hessPen_L_n;
        logEv_P(jj)     = logEv_P_n;     % cumulative log evidence
        logEv_L(jj)     = logEv_L_n;
        MSE_P(jj)       = mean(residuals_P.^2);
        MSE_L(jj)       = mean(residuals_L.^2);
        % Cumulative Bayes factor: ratio of cumulative Laplace evidences
        % BF > 1 favours Power, BF < 1 favours Linear
        BF(jj)          = exp(logEv_P_n - logEv_L_n);
        cond_A_P(jj)    = cond(A_P_n);
        cond_A_L(jj)    = cond(A_L_n);
        rank_JP(jj)     = rank(JJ_P);
        rank_JL(jj)     = rank(JJ_L);
        n_obs_store(jj) = n_obs;
        BIC_P_store(jj) = BIC_P_n;
        BIC_L_store(jj) = BIC_L_n;

        % Running posterior probability of Linear model
        % Both quantities use cumulative evidence after experiments 1:jj
        PLinear_store(cc, jj) = 1.0 / (1.0 + exp(logEv_P_n - logEv_L_n));
        BF_P_L_store(cc, jj)  = exp(logEv_P_n - logEv_L_n);  % cumulative BF
        DeltaBIC_store(cc, jj) = BIC_L_n - BIC_P_n;
    end

    %% Aggregate results
    n_total = sum(n_obs_store);

    logEv_P_total = logEv_P(N_exp);   % already cumulative
    logEv_L_total = logEv_L(N_exp);

    BIC_P_total      = BIC_P_store(N_exp);
    BIC_L_total      = BIC_L_store(N_exp);
    Delta_BIC_total  = BIC_L_total - BIC_P_total;

    PL = 1.0 / (1.0 + exp(logEv_P_total - logEv_L_total));
    PP = 1.0 - PL;

    fprintf('--- Per-experiment results (cumulative quantities after each experiment) ---\n');
    fprintf('%4s | %10s %10s | %10s %10s | %10s %10s | %10s %10s | %10s | %6s %6s | %6s %6s\n', ...
        'Exp', 'logLik_P', 'logLik_L', 'hess_P(n)', 'hess_L(n)', ...
        'logEv_P(n)', 'logEv_L(n)', 'BIC_P(n)', 'BIC_L(n)', 'BF(P/L)', ...
        'rkJP', 'rkJL', 'kAP', 'kAL');
    fprintf('%s\n', repmat('-', 1, 175));

    for jj = 1:N_exp
        fprintf('%4d | %10.4f %10.4f | %10.4f %10.4f | %10.4f %10.4f | %10.4f %10.4f | %10.4e | %6d %6d | %6.1f %6.1f\n', ...
            jj, ...
            logLik_P(jj), logLik_L(jj), ...
            hessPen_P(jj), hessPen_L(jj), ...
            logEv_P(jj),  logEv_L(jj), ...
            BIC_P_store(jj), BIC_L_store(jj), ...
            BF(jj), ...
            rank_JP(jj), rank_JL(jj), ...
            cond_A_P(jj), cond_A_L(jj));
    end

    fprintf('\n--- Final aggregate results (%s) ---\n', case_name);
    fprintf('  Sum logLik_P              = %.4f\n', sum(logLik_P));
    fprintf('  Sum logLik_L              = %.4f\n', sum(logLik_L));
    fprintf('  Prior norm P              = %.4f\n', logPriorNorm_P);
    fprintf('  Prior norm L              = %.4f\n', logPriorNorm_L);
    fprintf('  Cumul. Hessian pen. P     = %.4f\n', hessPen_P(N_exp));
    fprintf('  Cumul. Hessian pen. L     = %.4f\n', hessPen_L(N_exp));
    fprintf('  logEv_P (total)           = %.4f\n', logEv_P_total);
    fprintf('  logEv_L (total)           = %.4f\n', logEv_L_total);
    fprintf('  P(Power)  = PP            = %.6f\n', PP);
    fprintf('  P(Linear) = PL            = %.6f\n', PL);
    fprintf('  BIC_P (total)             = %.4f\n', BIC_P_total);
    fprintf('  BIC_L (total)             = %.4f\n', BIC_L_total);
    fprintf('  Delta BIC (L - P)         = %.4f\n', Delta_BIC_total);
    fprintf('  Mean MSE (Power)          = %.4e\n', mean(MSE_P));
    fprintf('  Mean MSE (Linear)         = %.4e\n', mean(MSE_L));

    if PP > PL
        winner_laplace = 'Power';
    else
        winner_laplace = 'Linear';
    end
    if Delta_BIC_total > 0
        winner_bic = 'Power';
    else
        winner_bic = 'Linear';
    end

    fprintf('\n  Laplace-evidence winner: %s\n', winner_laplace);
    fprintf('  BIC winner:              %s\n', winner_bic);
    fprintf('=============================================================================\n');
end

%% Plots
figure;
plot(PLinear_store', 'LineWidth', 2)
legend(yield_cases, Location="southeast", Box="off")
xlabel('Number of experiment')
ylabel('$P_L$ (Laplace evidence, cumulative Hessian)', 'Interpreter', 'latex')
xlim([1, N_exp])
set(gca, 'fontsize', 16)
print('Cum_Prob_L_cumHessian.png', '-dpng', '-r500'); close all

figure;
plot(BF_P_L_store', 'LineWidth', 2)
legend(yield_cases, Location="southeast", Box="off")
xlabel('Number of experiment')
ylabel('$BF(P/L)$ (Laplace evidence, cumulative Hessian)', 'Interpreter', 'latex')
xlim([1, N_exp])
set(gca, 'YScale', 'log', 'fontsize', 16)
print('BF_P_L_log_cumHessian.png', '-dpng', '-r500'); close all

figure;
plot(DeltaBIC_store', 'LineWidth', 2)
legend(yield_cases, Location="southeast", Box="off")
xlabel('Number of experiment')
ylabel('$\Delta \mathrm{BIC} = \mathrm{BIC}_L - \mathrm{BIC}_P$ (cumulative)', 'Interpreter', 'latex')
xlim([1, N_exp])
set(gca, 'fontsize', 16)
print('Delta_BIC_cumHessian.png', '-dpng', '-r500'); close all