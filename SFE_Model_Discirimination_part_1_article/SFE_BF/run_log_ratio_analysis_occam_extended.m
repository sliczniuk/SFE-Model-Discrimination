%% Run Discrimination Analysis
% Bayesian model discrimination between Power and Linear kinetic models
% for three yield representations:
%   1. Cumulative yield
%   2. Differentiated yield
%   3. Normalised differentiated yield
%
% This version uses a Laplace-style empirical-Bayes approximation:
%
%   Likelihood:
%       y | theta, M ~ N( G(theta), sigma^2 I )
%
%   Empirical Gaussian prior:
%       theta | M ~ N( theta_hat, Cov_theta )
%
%   Evaluated at theta_hat, the approximate log evidence is:
%
%       log p(y|M)
%       ≈ log p(y | theta_hat, M)
%         + log p(theta_hat | M)
%         + (d/2) log(2*pi)
%         - 0.5 log det(H)
%
%   where
%
%       H = (1/sigma^2) J'J + Cov_theta^{-1}
%
%   and because the prior is centered at theta_hat,
%
%       log p(theta_hat | M)
%       = -0.5 log det(Cov_theta) - (d/2) log(2*pi)
%
%   so the (d/2) log(2*pi) terms cancel, giving:
%
%       log evidence
%       ≈ log likelihood at theta_hat
%         - 0.5 log det(Cov_theta)
%         - 0.5 log det(H)
%
% IMPORTANT:
% - This is an empirical-Bayes / local Gaussian approximation.
% - Cov_power and Cov_linear are treated as empirical Gaussian priors.
% - The likelihood below uses sigma^2 I only (not sigma^2 I + J Cov J').
% - This is more internally consistent than mixing predictive covariance
%   inflation with a separate Hessian penalty.

%% Initialization
startup;
addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
import casadi.*

fprintf('=============================================================================\n');
fprintf('   MODEL DISCRIMINATION ANALYSIS (EMPIRICAL GAUSSIAN PRIOR + LAPLACE)\n');
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

% Nominal parameter values (prior means)
theta_power  = [1.222524; 4.308414; 0.972739; 3.428618];
theta_linear = [0.19; -8.188; 0.62; 3.158; 11.922; -0.6868];

%% Setup simulation infrastructure
m_total = 3.0;
before = 0.04;
bed    = 0.92;

Time_in_sec = (timeStep:timeStep:finalTime) * 60;
Time = [0, Time_in_sec/60];
N_Time = length(Time_in_sec);
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

C0solid = m_total * 1e-3 / (V_bed * epsi);
Parameters{2} = C0solid;

m_fluid = zeros(1, nstages);
C0fluid = m_fluid * 1e-3 ./ V_fluid';

epsi_mask            = epsi .* bed_mask;
one_minus_epsi_mask  = 1 - epsi_mask;

Nx = 3 * nstages + 2;
Nu = 3 + numel(Parameters);

%% Yield case definitions
yield_cases       = {'Cumulative', 'Differentiated', 'Normalised'};
Cov_power_cases   = {Cov_power_cum,  Cov_power_diff,  Cov_power_norm};
Cov_linear_cases  = {Cov_linear_cum, Cov_linear_diff, Cov_linear_norm};
sigma2_cases      = [2.45e-2, 1.386e-3, 1.007e-2];  % empirical observation variances
N_cases           = numel(yield_cases);
N_exp             = 12;

dP = numel(theta_power);    % 4
dL = numel(theta_linear);   % 6

%% Storage for plots
PLinear_store   = zeros(N_cases, N_exp);   % running posterior-like probability of Linear
BF_P_L_store    = zeros(N_cases, N_exp);   % running BF(P/L)
DeltaBIC_store  = zeros(N_cases, N_exp);   % running Delta BIC = BIC_L - BIC_P

%% Run discrimination for each yield case
for cc = 1:N_cases
    case_name  = yield_cases{cc};
    Cov_power  = Cov_power_cases{cc};
    Cov_linear = Cov_linear_cases{cc};
    sigma2_y   = sigma2_cases(cc);

    fprintf('\n=============================================================================\n');
    fprintf('   CASE %d: %s yield\n', cc, upper(case_name));
    fprintf('=============================================================================\n\n');

    % Log-evidence approximations
    logEv_P = zeros(1, N_exp);
    logEv_L = zeros(1, N_exp);

    % Plain log-likelihoods
    logLik_P = zeros(1, N_exp);
    logLik_L = zeros(1, N_exp);

    % Prior normalization contributions
    logPriorNorm_P = zeros(1, N_exp);
    logPriorNorm_L = zeros(1, N_exp);

    % Hessian penalties
    hessPen_P = zeros(1, N_exp);
    hessPen_L = zeros(1, N_exp);

    % BIC
    BIC_P = zeros(1, N_exp);
    BIC_L = zeros(1, N_exp);
    Delta_BIC = zeros(1, N_exp);

    MSE_P = zeros(1, N_exp);
    MSE_L = zeros(1, N_exp);
    BF    = zeros(1, N_exp);

    cond_H_P = zeros(1, N_exp);
    cond_H_L = zeros(1, N_exp);
    rank_JP  = zeros(1, N_exp);
    rank_JL  = zeros(1, N_exp);
    n_obs_store = zeros(1, N_exp);

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
        Z = Compressibility(T, P, Parameters);
        rho = rhoPB_Comp(T, P, Z, Parameters);
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

        % CasADi function and evaluation at empirical prior means
        G = Function('G', {k}, {J_P_sym, J_L_sym, Y_P_sym, Y_L_sym});
        [JJ_P, JJ_L, Y_P, Y_L] = G([theta_power; theta_linear]);

        JJ_P = full(JJ_P);
        JJ_L = full(JJ_L);
        Y_P  = full(Y_P(:));
        Y_L  = full(Y_L(:));

        data_ref = data_ref(:);

        % Residuals
        residuals_P = Y_P - data_ref;
        residuals_L = Y_L - data_ref;

        n_obs = numel(data_ref);

        % ================================================================
        % Likelihood: y | theta, M ~ N(G(theta), sigma^2 I)
        % ================================================================
        Sigma_y_P = sigma2_y * eye(n_obs);
        Sigma_y_L = sigma2_y * eye(n_obs);

        % Use lower-triangular Cholesky
        L_y_P = chol(Sigma_y_P, 'lower');
        L_y_L = chol(Sigma_y_L, 'lower');

        solve_P = L_y_P \ residuals_P;
        solve_L = L_y_L \ residuals_L;

        logLik_P_j = -0.5 * sum(solve_P.^2) ...
                     - sum(log(diag(L_y_P))) ...
                     - n_obs/2 * log(2*pi);

        logLik_L_j = -0.5 * sum(solve_L.^2) ...
                     - sum(log(diag(L_y_L))) ...
                     - n_obs/2 * log(2*pi);

        % ================================================================
        % Empirical Gaussian prior normalization at theta_hat
        % theta ~ N(theta_hat, Cov_theta)
        %
        % log p(theta_hat|M) = -0.5 log det(Cov_theta) - d/2 log(2*pi)
        % ================================================================
        R_cov_P = chol(Cov_power, 'lower');
        R_cov_L = chol(Cov_linear, 'lower');

        logdetCov_P = 2 * sum(log(diag(R_cov_P)));
        logdetCov_L = 2 * sum(log(diag(R_cov_L)));

        logPriorNorm_P_j = -0.5 * logdetCov_P - dP/2 * log(2*pi);
        logPriorNorm_L_j = -0.5 * logdetCov_L - dL/2 * log(2*pi);

        % ================================================================
        % Hessian of negative log posterior (Gauss-Newton + empirical prior)
        % H = (1/sigma^2) J'J + Cov_theta^{-1}
        % ================================================================
        Prec_power  = Cov_power  \ eye(size(Cov_power));
        Prec_linear = Cov_linear \ eye(size(Cov_linear));

        H_P = (1/sigma2_y) * (JJ_P' * JJ_P) + Prec_power;
        H_L = (1/sigma2_y) * (JJ_L' * JJ_L) + Prec_linear;

        % Small ridge for numerical robustness
        ridge_P = 1e-12 * eye(size(H_P));
        ridge_L = 1e-12 * eye(size(H_L));

        R_HP = chol(H_P + ridge_P, 'lower');
        R_HL = chol(H_L + ridge_L, 'lower');

        % Hessian penalty: -0.5 log det(H)
        hessPen_P_j = -sum(log(diag(R_HP)));
        hessPen_L_j = -sum(log(diag(R_HL)));

        % ================================================================
        % Laplace evidence approximation
        %
        % log evidence = logLik + logPriorNorm + d/2 log(2*pi) - 0.5 logdet(H)
        %
        % Since logPriorNorm already contains -d/2 log(2*pi),
        % the d/2 log(2*pi) terms cancel:
        %
        % log evidence = logLik - 0.5 logdet(Cov_theta) - 0.5 logdet(H)
        % ================================================================
        logEv_P_j = logLik_P_j + logPriorNorm_P_j + dP/2 * log(2*pi) + hessPen_P_j;
        logEv_L_j = logLik_L_j + logPriorNorm_L_j + dL/2 * log(2*pi) + hessPen_L_j;

        % ================================================================
        % BIC as auxiliary complexity check (lower is better)
        % ================================================================
        BIC_P_j = -2 * logLik_P_j + dP * log(n_obs);
        BIC_L_j = -2 * logLik_L_j + dL * log(n_obs);
        Delta_BIC_j = BIC_L_j - BIC_P_j;   % > 0 favors Power

        % Store
        logLik_P(jj) = logLik_P_j;
        logLik_L(jj) = logLik_L_j;

        logPriorNorm_P(jj) = logPriorNorm_P_j;
        logPriorNorm_L(jj) = logPriorNorm_L_j;

        hessPen_P(jj) = hessPen_P_j;
        hessPen_L(jj) = hessPen_L_j;

        logEv_P(jj) = logEv_P_j;
        logEv_L(jj) = logEv_L_j;

        BIC_P(jj) = BIC_P_j;
        BIC_L(jj) = BIC_L_j;
        Delta_BIC(jj) = Delta_BIC_j;

        MSE_P(jj) = mean(residuals_P.^2);
        MSE_L(jj) = mean(residuals_L.^2);

        BF(jj) = exp(logEv_P_j - logEv_L_j);

        rank_JP(jj) = rank(JJ_P);
        rank_JL(jj) = rank(JJ_L);
        cond_H_P(jj) = cond(H_P);
        cond_H_L(jj) = cond(H_L);

        n_obs_store(jj) = n_obs;

        % Running posterior-like probability for Linear model from Laplace evidence
        PLinear_store(cc, jj) = 1.0 / (1.0 + exp(sum(logEv_P(1:jj)) - sum(logEv_L(1:jj))));
        BF_P_L_store(cc, jj)  = BF(jj);

        % Running aggregate Delta BIC
        n_total_running = sum(n_obs_store(1:jj));
        BIC_P_running = -2 * sum(logLik_P(1:jj)) + dP * log(n_total_running);
        BIC_L_running = -2 * sum(logLik_L(1:jj)) + dL * log(n_total_running);
        DeltaBIC_store(cc, jj) = BIC_L_running - BIC_P_running;
    end

    %% Aggregate results
    n_total = sum(n_obs_store);

    logEv_P_total = sum(logEv_P);
    logEv_L_total = sum(logEv_L);

    BIC_P_total = -2 * sum(logLik_P) + dP * log(n_total);
    BIC_L_total = -2 * sum(logLik_L) + dL * log(n_total);
    Delta_BIC_total = BIC_L_total - BIC_P_total;

    PL = 1.0 / (1.0 + exp(logEv_P_total - logEv_L_total));
    PP = 1.0 - PL;

    fprintf('--- Per-experiment results ---\n');
    fprintf('%4s | %10s %10s | %10s %10s | %10s %10s | %10s %10s | %10s | %6s %6s | %6s %6s\n', ...
        'Exp', 'logLik_P', 'logLik_L', 'prior_P', 'prior_L', ...
        'hess_P', 'hess_L', 'logEv_P', 'logEv_L', 'BF(P/L)', ...
        'rkJP', 'rkJL', 'kHP', 'kHL');
    fprintf('%s\n', repmat('-', 1, 160));

    for jj = 1:N_exp
        fprintf('%4d | %10.4f %10.4f | %10.4f %10.4f | %10.4f %10.4f | %10.4f %10.4f | %10.4e | %6d %6d | %6.1f %6.1f\n', ...
            jj, ...
            logLik_P(jj), logLik_L(jj), ...
            logPriorNorm_P(jj), logPriorNorm_L(jj), ...
            hessPen_P(jj), hessPen_L(jj), ...
            logEv_P(jj), logEv_L(jj), ...
            BF(jj), ...
            rank_JP(jj), rank_JL(jj), ...
            cond_H_P(jj), cond_H_L(jj));
    end

    fprintf('\n--- Aggregate results (%s) ---\n', case_name);
    fprintf('  Sum logLik_P           = %.4f\n', sum(logLik_P));
    fprintf('  Sum logLik_L           = %.4f\n', sum(logLik_L));
    fprintf('  Sum prior_P            = %.4f\n', sum(logPriorNorm_P));
    fprintf('  Sum prior_L            = %.4f\n', sum(logPriorNorm_L));
    fprintf('  Sum hess_P             = %.4f\n', sum(hessPen_P));
    fprintf('  Sum hess_L             = %.4f\n', sum(hessPen_L));
    fprintf('  Sum logEv_P            = %.4f\n', logEv_P_total);
    fprintf('  Sum logEv_L            = %.4f\n', logEv_L_total);
    fprintf('  Mean MSE (Power)       = %.4e\n', mean(MSE_P));
    fprintf('  Mean MSE (Linear)      = %.4e\n', mean(MSE_L));
    fprintf('  P(Power)  = PP         = %.6f\n', PP);
    fprintf('  P(Linear) = PL         = %.6f\n', PL);
    fprintf('  BIC_P (total)          = %.4f\n', BIC_P_total);
    fprintf('  BIC_L (total)          = %.4f\n', BIC_L_total);
    fprintf('  Delta BIC (L - P)      = %.4f\n', Delta_BIC_total);

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
    fprintf('  BIC winner: %s\n', winner_bic);
    fprintf('=============================================================================\n');
end

%% Plots
plot(PLinear_store', 'LineWidth', 2)
legend(yield_cases, Location="southeast", Box="off")
xlabel('Number of experiment')
ylabel('$P_L$ (Laplace evidence)')
xlim([1, N_exp])
set(gca, 'fontsize', 16)
print('Cum_Prob_L_empirical_prior.png', '-dpng', '-r500'); close all

plot(BF_P_L_store', 'LineWidth', 2)
legend(yield_cases, Location="southeast", Box="off")
xlabel('Number of experiment')
ylabel('$BF(P/L)$ (Laplace evidence)')
xlim([1, N_exp])
set(gca, 'YScale', 'log', 'fontsize', 16)
print('BF_P_L_log_empirical_prior.png', '-dpng', '-r500'); close all

plot(DeltaBIC_store', 'LineWidth', 2)
legend(yield_cases, Location="southeast", Box="off")
xlabel('Number of experiment')
ylabel('$\Delta \mathrm{BIC} = \mathrm{BIC}_L - \mathrm{BIC}_P$')
xlim([1, N_exp])
set(gca, 'fontsize', 16)
print('Delta_BIC_empirical_prior.png', '-dpng', '-r500'); close all