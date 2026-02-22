%% Run Discrimination Analysis
% Bayesian model discrimination between Power and Linear kinetic models
% for three yield representations:
%   1. Cumulative yield
%   2. Differentiated yield
%   3. Normalised differentiated yield
%
% Each case uses its own parameter covariance matrix estimated from the
% corresponding yield type.

%% Initialization
startup;
addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
import casadi.*

fprintf('=============================================================================\n');
fprintf('   MODEL DISCRIMINATION ANALYSIS \n');
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
timeStep  = 5;  % Time step [min]
finalTime = 600; % Extraction time [min]
Time      = 0 : timeStep: finalTime;

%sigma2_scale = 0.01;

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
before = 0.04;
bed = 0.92;

Time_in_sec = (timeStep:timeStep:finalTime) * 60;
Time = [0, Time_in_sec/60];
N_Time = length(Time_in_sec);
timeStep_in_sec = timeStep * 60;

nstages = Parameters{1};
r = Parameters{3};
epsi = Parameters{4};
L = Parameters{6};

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
sigma2_cases = [2.45e-2, 1.386e-3, 1.007e-2]; % Mean empirical sigma as given in the report
N_cases = numel(yield_cases);
N_exp   = 12;

%% Run discrimination for each yield case
log_L_store = zeros(N_cases, N_exp);
BF_P_L_store = zeros(N_cases, N_exp);
for cc = 1:N_cases
    case_name  = yield_cases{cc};
    Cov_power  = Cov_power_cases{cc};
    Cov_linear = Cov_linear_cases{cc};

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

    for jj = 1:N_exp
        data_org      = LabResults(6:19, jj+1)';
        data_diff     = diff(data_org);
        max_data_diff = max(data_diff);
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
        if cc == 1  % Cumulative
            Y_P_sym    = Y_cum_P_sym;
            Y_L_sym    = Y_cum_L_sym;
            data_ref   = data_org;
            scale      = 1;
        elseif cc == 2  % Differentiated
            Y_P_sym    = Y_diff_P_sym;
            Y_L_sym    = Y_diff_L_sym;
            data_ref   = data_diff;
            scale      = 1;
        else  % Normalised
            Y_P_sym    = Y_diff_P_sym ./ max_data_diff;
            Y_L_sym    = Y_diff_L_sym ./ max_data_diff;
            data_ref   = data_diff_norm;
            scale      = 1;  % already normalised
        end

        % Jacobians
        J_P_sym = jacobian(Y_P_sym, k1);
        J_L_sym = jacobian(Y_L_sym, k2);

        % CasADi function and evaluation
        G = Function('G', {k}, {J_P_sym, J_L_sym, Y_P_sym, Y_L_sym});
        [JJ_P, JJ_L, Y_P, Y_L] = G([theta_power; theta_linear]);
        JJ_P = full(JJ_P);
        JJ_L = full(JJ_L);
        Y_P  = full(Y_P);
        Y_L  = full(Y_L);

        % Residuals
        residuals_P = Y_P - data_ref;
        residuals_L = Y_L - data_ref;

        % Predictive covariance and log-likelihood (Power)
        Sigma_r_P = sigma2_cases(cc) * eye(numel(Y_P)) + JJ_P * Cov_power * JJ_P';
        L_P       = chol(Sigma_r_P);
        solve_P   = linsolve(L_P', residuals_P', struct('LT',true));
        log_P_r_j = -0.5 * sum(solve_P.^2) - sum(log(diag(L_P))) - numel(Y_P)/2 * log(2*pi);

        % Predictive covariance and log-likelihood (Linear)
        Sigma_r_L = sigma2_cases(cc) * eye(numel(Y_L)) + JJ_L * Cov_linear * JJ_L';
        L_L       = chol(Sigma_r_L);
        solve_L   = linsolve(L_L', residuals_L', struct('LT',true));
        log_L_r_j = -0.5 * sum(solve_L.^2) - sum(log(diag(L_L))) - numel(Y_L)/2 * log(2*pi);

        % Store
        log_P(jj)   = log_P_r_j;
        log_L(jj)   = log_L_r_j;
        MSE_P(jj)   = mean(residuals_P.^2);
        MSE_L(jj)   = mean(residuals_L.^2);
        BF(jj)      = exp(log_P_r_j - log_L_r_j);
        cond_P(jj)  = cond(Sigma_r_P);
        cond_L(jj)  = cond(Sigma_r_L);
        rank_JP(jj) = rank(JJ_P);
        rank_JL(jj) = rank(JJ_L);

        log_L_store(cc, jj) = 1.0 / (1.0 + exp(sum(log_P) - sum(log_L))) ; 
        BF_P_L_store(cc, jj) = exp(log_P_r_j - log_L_r_j);

    end

    %% Diagnostic Summary for this case
    PL = 1.0 / (1.0 + exp(sum(log_P) - sum(log_L)));
    PP = 1.0 - PL;

    fprintf('--- Per-experiment results ---\n');
    fprintf('%4s | %12s %12s | %12s %12s | %12s | %6s %6s | %6s %6s\n', ...
        'Exp', 'log_P', 'log_L', 'MSE_P', 'MSE_L', 'BF(P/L)', ...
        'rk_JP', 'rk_JL', 'k(S_P)', 'k(S_L)');
    fprintf('%s\n', repmat('-', 1, 110));
    for jj = 1:N_exp
        fprintf('%4d | %12.4f %12.4f | %12.4e %12.4e | %12.4e | %6d %6d | %6.1f %6.1f\n', ...
            jj, log_P(jj), log_L(jj), MSE_P(jj), MSE_L(jj), BF(jj), ...
            rank_JP(jj), rank_JL(jj), cond_P(jj), cond_L(jj));
    end

    fprintf('\n--- Aggregate results (%s) ---\n', case_name);
    fprintf('  Sum log_P         = %.4f\n', sum(log_P));
    fprintf('  Sum log_L         = %.4f\n', sum(log_L));
    fprintf('  Mean MSE (Power)  = %.4e\n', mean(MSE_P));
    fprintf('  Mean MSE (Linear) = %.4e\n', mean(MSE_L));
    fprintf('  P(Power)  = PP    = %.6f\n', PP);
    fprintf('  P(Linear) = PL    = %.6f\n', PL);
    if PP > PL
        winner = 'Power';
    else
        winner = 'Linear';
    end
    fprintf('\n  Conclusion: %s model is preferred (PP=%.4f, PL=%.4f)\n', winner, PP, PL);
    fprintf('=============================================================================\n');
end

%%
plot(log_L_store', 'LineWidth',2)
legend(yield_cases, Location="southeast", Box="off")
xlabel('Number of experiment')
ylabel('$P_L$')
xlim([1, N_exp])
set(gca,'fontsize',16)
print('Cum_Prob_L.png','-dpng', '-r500'); close all


plot(BF_P_L_store', 'LineWidth',2)
legend(yield_cases, Location="southeast", Box="off")
xlabel('Number of experiment')
ylabel('$BF(P/L)$')
xlim([1, N_exp])
set(gca, 'YScale', 'log', 'fontsize',16)
print('BF_P_L_log.png','-dpng', '-r500'); close all





































