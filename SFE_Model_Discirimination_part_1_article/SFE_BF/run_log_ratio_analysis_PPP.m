%% Run Discrimination Analysis with Posterior Predictive P-value (PPP)
% Bayesian model discrimination between Power and Linear kinetic models
% for three yield representations:
%   1. Cumulative yield
%   2. Differentiated yield
%   3. Normalised differentiated yield
%
% Each case uses its own parameter covariance matrix estimated from the
% corresponding yield type.
%
% POSTERIOR PREDICTIVE P-VALUE (PPP):
%   After discrimination, the selected model (Linear) is assessed for
%   adequacy using the PPP based on the chi-squared discrepancy statistic:
%
%       T^(j) = (y^obs - y_hat)' * Sigma^{-1} * (y^obs - y_hat)
%
%   where Sigma = sigma^2*I + J*Cov_theta*J' is the predictive covariance
%   already computed in the discrimination step. Under the null hypothesis
%   that the model is correct, T^(j) follows approximately:
%
%       T^(j) ~ chi^2(nu),   nu = n_Y - d_i
%
%   where n_Y is the number of observations per experiment and d_i is the
%   number of model parameters. The PPP for experiment j is:
%
%       PPP^(j) = 1 - F_{chi^2(nu)}(T^(j))
%
%   A global PPP is computed from the pooled statistic across all
%   experiments:
%
%       T_global = sum_j T^(j) ~ chi^2(N_exp * nu)
%
%   PPP values near 0.5 indicate good fit. Values below 0.05 or above
%   0.95 indicate model inadequacy.

%% Initialization
startup;
addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
import casadi.*

fprintf('=============================================================================\n');
fprintf('   MODEL DISCRIMINATION + POSTERIOR PREDICTIVE P-VALUE ANALYSIS\n');
fprintf('=============================================================================\n\n');

%% Load parameters and data
Parameters_table = readtable('Parameters.csv');
Parameters       = num2cell(Parameters_table{:,3});
LabResults       = xlsread('dataset_2.xlsx');

which_k = (0:9) + 44;     % Indices of parameters to fit (44-53)
Nk      = numel(which_k);  % 10 parameters (4 Power + 6 Linear)
k1      = MX.sym('k1', 4);
k2      = MX.sym('k2', 6);
k       = [k1; k2];

%% Set up the simulation
timeStep  = 5;   % Time step [min]
finalTime = 600; % Extraction time [min]
Time      = 0 : timeStep: finalTime;

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

% Linear model covariance (6x6)
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

%% Storage
log_L_store    = zeros(N_cases, N_exp);
BF_P_L_store   = zeros(N_cases, N_exp);

% PPP storage: per experiment, per case, for both models
PPP_P_store    = zeros(N_cases, N_exp);   % per-experiment PPP, Power
PPP_L_store    = zeros(N_cases, N_exp);   % per-experiment PPP, Linear
T_obs_P_store  = zeros(N_cases, N_exp);   % chi-sq statistic, Power
T_obs_L_store  = zeros(N_cases, N_exp);   % chi-sq statistic, Linear

%% Run discrimination for each yield case
for cc = 1:N_cases
    case_name  = yield_cases{cc};
    Cov_power  = Cov_power_cases{cc};
    Cov_linear = Cov_linear_cases{cc};
    sigma2_y   = sigma2_cases(cc);

    fprintf('\n=============================================================================\n');
    fprintf('   CASE %d: %s yield\n', cc, upper(case_name));
    fprintf('=============================================================================\n\n');

    log_P   = zeros(1, N_exp);
    log_L   = zeros(1, N_exp);
    MSE_P   = zeros(1, N_exp);
    MSE_L   = zeros(1, N_exp);
    BF      = zeros(1, N_exp);
    cond_P  = zeros(1, N_exp);
    cond_L  = zeros(1, N_exp);
    rank_JP = zeros(1, N_exp);
    rank_JL = zeros(1, N_exp);

    % PPP per experiment
    T_obs_P = zeros(1, N_exp);
    T_obs_L = zeros(1, N_exp);
    PPP_P   = zeros(1, N_exp);
    PPP_L   = zeros(1, N_exp);
    n_Y_store = zeros(1, N_exp);   % store n_Y per experiment

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

        % Differentiated yield
        Y_diff_P_sym = Y_cum_P_sym(2:end) - Y_cum_P_sym(1:end-1);
        Y_diff_L_sym = Y_cum_L_sym(2:end) - Y_cum_L_sym(1:end-1);

        % Select symbolic output, data, and scaling based on case
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

        % CasADi function and evaluation
        G = Function('G', {k}, {J_P_sym, J_L_sym, Y_P_sym, Y_L_sym});
        [JJ_P, JJ_L, Y_P, Y_L] = G([theta_power; theta_linear]);
        JJ_P     = full(JJ_P);
        JJ_L     = full(JJ_L);
        Y_P      = full(Y_P);
        Y_L      = full(Y_L);
        data_ref = data_ref(:);

        n_Y = numel(data_ref);
        n_Y_store(jj) = n_Y;

        % Residuals
        residuals_P = Y_P(:) - data_ref;
        residuals_L = Y_L(:) - data_ref;

        % ================================================================
        % DISCRIMINATION: predictive covariance and log-likelihood
        % Sigma = sigma^2*I + J*Cov_theta*J'
        % ================================================================
        Sigma_r_P = sigma2_y * eye(n_Y) + JJ_P * Cov_power  * JJ_P';
        Sigma_r_L = sigma2_y * eye(n_Y) + JJ_L * Cov_linear * JJ_L';

        L_P = chol(Sigma_r_P, 'lower');
        L_L = chol(Sigma_r_L, 'lower');

        solve_P = L_P \ residuals_P;
        solve_L = L_L \ residuals_L;

        log_P_r_j = -0.5 * sum(solve_P.^2) - sum(log(diag(L_P))) - n_Y/2 * log(2*pi);
        log_L_r_j = -0.5 * sum(solve_L.^2) - sum(log(diag(L_L))) - n_Y/2 * log(2*pi);

        % Store discrimination results
        log_P(jj)   = log_P_r_j;
        log_L(jj)   = log_L_r_j;
        MSE_P(jj)   = mean(residuals_P.^2);
        MSE_L(jj)   = mean(residuals_L.^2);
        BF(jj)      = exp(log_P_r_j - log_L_r_j);
        cond_P(jj)  = cond(Sigma_r_P);
        cond_L(jj)  = cond(Sigma_r_L);
        rank_JP(jj) = rank(JJ_P);
        rank_JL(jj) = rank(JJ_L);

        log_L_store(cc, jj)  = 1.0 / (1.0 + exp(sum(log_P(1:jj)) - sum(log_L(1:jj))));
        BF_P_L_store(cc, jj) = exp(log_P_r_j - log_L_r_j);

        % ================================================================
        % POSTERIOR PREDICTIVE P-VALUE (PPP)
        %
        % The chi-squared discrepancy statistic for experiment j:
        %
        %   T^(j) = r' * Sigma^{-1} * r
        %
        % where r = y_obs - y_hat and Sigma is the predictive covariance.
        % Sigma already integrates out parameter uncertainty, so T^(j)
        % follows chi^2(nu) with effective degrees of freedom:
        %
        %   nu = n_Y - d_i
        %
        % where d_i accounts for the parameters absorbed in fitting.
        %
        % PPP^(j) = 1 - F_{chi^2(nu)}(T^(j))
        %
        % Note: we use the SAME Sigma already computed for discrimination,
        % ensuring internal consistency between the two analyses.
        % ================================================================

        % Degrees of freedom
        % n_Y observations minus d_i fitted parameters
        nu_P = max(n_Y - dP, 1);   % guard against zero or negative df
        nu_L = max(n_Y - dL, 1);

        % Chi-squared discrepancy statistics using Cholesky solves
        % T = r' Sigma^{-1} r = ||L^{-1} r||^2
        T_obs_P_j = sum(solve_P.^2);   % = r_P' * Sigma_P^{-1} * r_P
        T_obs_L_j = sum(solve_L.^2);   % = r_L' * Sigma_L^{-1} * r_L

        % Per-experiment PPP: probability of seeing a more extreme statistic
        % under the null hypothesis that the model is correct
        PPP_P_j = 1 - chi2cdf(T_obs_P_j, nu_P);
        PPP_L_j = 1 - chi2cdf(T_obs_L_j, nu_L);

        % Store
        T_obs_P(jj)   = T_obs_P_j;
        T_obs_L(jj)   = T_obs_L_j;
        PPP_P(jj)     = PPP_P_j;
        PPP_L(jj)     = PPP_L_j;

        T_obs_P_store(cc, jj) = T_obs_P_j;
        T_obs_L_store(cc, jj) = T_obs_L_j;
        PPP_P_store(cc, jj)   = PPP_P_j;
        PPP_L_store(cc, jj)   = PPP_L_j;
    end

    % ================================================================
    % GLOBAL PPP across all experiments
    %
    % Pool chi-squared statistics:
    %   T_global = sum_j T^(j)
    %
    % Under H0, T_global ~ chi^2(sum_j nu_j) = chi^2(N_exp * nu)
    % if all experiments have the same n_Y and d_i.
    %
    % More generally use sum of individual degrees of freedom.
    % ================================================================
    nu_P_total = sum(max(n_Y_store - dP, 1));
    nu_L_total = sum(max(n_Y_store - dL, 1));

    T_global_P = sum(T_obs_P);
    T_global_L = sum(T_obs_L);

    PPP_global_P = 1 - chi2cdf(T_global_P, nu_P_total);
    PPP_global_L = 1 - chi2cdf(T_global_L, nu_L_total);

    %% Diagnostic summary
    PL = 1.0 / (1.0 + exp(sum(log_P) - sum(log_L)));
    PP_prob = 1.0 - PL;

    fprintf('--- Per-experiment discrimination and PPP results ---\n');
    fprintf('%4s | %10s %10s | %10s | %8s %8s | %8s %8s | %8s %8s\n', ...
        'Exp', 'log_P', 'log_L', 'BF(P/L)', ...
        'T_P', 'PPP_P', 'T_L', 'PPP_L', 'MSE_P', 'MSE_L');
    fprintf('%s\n', repmat('-', 1, 120));

    for jj = 1:N_exp
        % Flag experiments with poor fit (PPP < 0.05 or PPP > 0.95)
        flag_P = '';
        flag_L = '';
        if PPP_P(jj) < 0.05 || PPP_P(jj) > 0.95
            flag_P = ' (*)';
        end
        if PPP_L(jj) < 0.05 || PPP_L(jj) > 0.95
            flag_L = ' (*)';
        end

        fprintf('%4d | %10.4f %10.4f | %10.4e | %8.3f %8.4f%s | %8.3f %8.4f%s | %8.4e %8.4e\n', ...
            jj, log_P(jj), log_L(jj), BF(jj), ...
            T_obs_P(jj), PPP_P(jj), flag_P, ...
            T_obs_L(jj), PPP_L(jj), flag_L, ...
            MSE_P(jj), MSE_L(jj));
    end

    fprintf('\n--- Global PPP (%s) ---\n', case_name);
    fprintf('  Power  model: T_global = %8.3f, df = %4d, PPP = %.4f\n', ...
        T_global_P, nu_P_total, PPP_global_P);
    fprintf('  Linear model: T_global = %8.3f, df = %4d, PPP = %.4f\n', ...
        T_global_L, nu_L_total, PPP_global_L);

    if PPP_global_L > 0.05 && PPP_global_L < 0.95
        fprintf('  Linear model adequacy: PASS (PPP = %.4f, within [0.05, 0.95])\n', PPP_global_L);
    else
        fprintf('  Linear model adequacy: FAIL (PPP = %.4f, outside [0.05, 0.95])\n', PPP_global_L);
        fprintf('  --> Consider additional model candidates.\n');
    end

    fprintf('\n--- Discrimination aggregate (%s) ---\n', case_name);
    fprintf('  Sum log_P     = %.4f\n', sum(log_P));
    fprintf('  Sum log_L     = %.4f\n', sum(log_L));
    fprintf('  P(Power)      = %.6f\n', PP_prob);
    fprintf('  P(Linear)     = %.6f\n', PL);
    if PP_prob > PL
        winner = 'Power';
    else
        winner = 'Linear';
    end
    fprintf('  Preferred model: %s\n', winner);
    fprintf('=============================================================================\n');
end

%% Plots - Discrimination
figure;
plot(log_L_store', 'LineWidth', 2)
legend(yield_cases, Location='southeast', Box='off')
xlabel('Number of experiment')
ylabel('$P_L$', 'Interpreter', 'latex')
xlim([1, N_exp])
set(gca, 'fontsize', 16)
print('Cum_Prob_L.png', '-dpng', '-r500'); close all

figure;
plot(BF_P_L_store', 'LineWidth', 2)
legend(yield_cases, Location='southeast', Box='off')
xlabel('Number of experiment')
ylabel('$BF(P/L)$', 'Interpreter', 'latex')
xlim([1, N_exp])
set(gca, 'YScale', 'log', 'fontsize', 16)
print('BF_P_L_log.png', '-dpng', '-r500'); close all

%% Plots - PPP for Linear model (selected model)
figure;
% Per-experiment PPP for Linear model across all yield cases
hold on
colors = lines(N_cases);
for cc = 1:N_cases
    plot(1:N_exp, PPP_L_store(cc,:), '-o', 'LineWidth', 2, ...
        'Color', colors(cc,:), 'DisplayName', yield_cases{cc});
end
% Adequacy thresholds
yline(0.05, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Adequacy threshold (0.05)')
yline(0.95, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off')
hold off
legend(Location='best', Box='off')
xlabel('Number of experiment')
ylabel('PPP (Linear model)', 'Interpreter', 'latex')
xlim([1, N_exp])
ylim([0, 1])
set(gca, 'fontsize', 16)
print('PPP_Linear.png', '-dpng', '-r500'); close all

figure;
% Per-experiment chi-squared statistics for both models, cumulative case
cc_plot = 1;  % show cumulative yield case as representative
bar_data = [T_obs_P_store(cc_plot,:); T_obs_L_store(cc_plot,:)]';
bar(bar_data, 'grouped')
legend({'Power', 'Linear'}, Box='off')
xlabel('Experiment')
ylabel('$T^{(j)} = r^\top \Sigma^{-1} r$', 'Interpreter', 'latex')
title(sprintf('Chi-squared discrepancy (%s yield)', yield_cases{cc_plot}))
set(gca, 'fontsize', 16)
print('Tstat_cumulative.png', '-dpng', '-r500'); close all

%% Summary table: global PPP for all cases and both models
fprintf('\n=============================================================================\n');
fprintf('   GLOBAL PPP SUMMARY\n');
fprintf('=============================================================================\n');
fprintf('%15s | %10s %10s %10s | %10s %10s %10s\n', ...
    'Yield case', 'T_P', 'df_P', 'PPP_P', 'T_L', 'df_L', 'PPP_L');
fprintf('%s\n', repmat('-', 1, 80));

for cc = 1:N_cases
    % Recompute global statistics from stored values
    n_Y_all   = size(PPP_P_store, 2);  % N_exp
    % Use stored T statistics
    T_g_P = sum(T_obs_P_store(cc,:));
    T_g_L = sum(T_obs_L_store(cc,:));
    % Degrees of freedom: need n_Y per experiment
    % For simplicity use the n_Y from the last run (same for all exps)
    nu_P_g = N_exp * max(n_Y_store - dP, 1);
    nu_L_g = N_exp * max(n_Y_store - dL, 1);
    % Use scalar df (assuming constant n_Y across experiments)
    nu_P_scalar = sum(max(n_Y_store - dP, 1));
    nu_L_scalar = sum(max(n_Y_store - dL, 1));

    PPP_g_P = 1 - chi2cdf(T_g_P, nu_P_scalar);
    PPP_g_L = 1 - chi2cdf(T_g_L, nu_L_scalar);

    fprintf('%15s | %10.3f %10d %10.4f | %10.3f %10d %10.4f\n', ...
        yield_cases{cc}, T_g_P, nu_P_scalar, PPP_g_P, ...
        T_g_L, nu_L_scalar, PPP_g_L);
end
fprintf('=============================================================================\n');
fprintf('(*) flags experiments where PPP < 0.05 or PPP > 0.95\n\n');