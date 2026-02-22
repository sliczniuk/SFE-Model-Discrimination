%% Estimate Analysis
% Simulate both Power and Linear models at nominal parameters across all
% 12 operating conditions and summarise the fit quality.
%
% For each model and experiment the script computes:
%   - Cumulative yield residuals
%   - Differentiated yield residuals
%   - Normalised (by max diff) yield residuals
%
% Summary statistics per experiment: MSE, STD
% Aggregate: overall MSE, STD, parameter covariance via Gauss-Newton approx.

%% Initialization
startup;
addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
import casadi.*

fprintf('=============================================================================\n');
fprintf('   ESTIMATE ANALYSIS - MODEL FIT SUMMARY\n');
fprintf('=============================================================================\n\n');

%% Load parameters and data
Parameters_table = readtable('Parameters.csv');
Parameters       = num2cell(Parameters_table{:,3});
LabResults       = xlsread('dataset_2.xlsx');

% Nominal parameter values
theta_power  = [1.222524; 4.308414; 0.972739; 3.428618];
theta_linear = [0.19; -8.188; 0.62; 3.158; 11.922; -0.6868];

which_k_P = 44:47;   % Power model parameter indices
which_k_L = 48:53;   % Linear model parameter indices

%% Set up the simulation
timeStep  = 5;          % Time step [min]
finalTime = 600;        % Extraction time [min]
Time      = 0:timeStep:finalTime;

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

N_exp     = 12;
N_samples = numel(SAMPLE);       % 14 sample points (cumulative)
N_diff    = N_samples - 1;        % 13 differentiated observations

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

C0solid        = m_total * 1e-3 / (V_bed * epsi);
Parameters{2}  = C0solid;

m_fluid = zeros(1, nstages);
C0fluid = m_fluid * 1e-3 ./ V_fluid';

epsi_mask           = epsi .* bed_mask;
one_minus_epsi_mask = 1 - epsi_mask;

Nx = 3 * nstages + 2;
Nu = 3 + numel(Parameters);

%% Symbolic parameters for Jacobian computation
k1 = MX.sym('k1', numel(which_k_P));
k2 = MX.sym('k2', numel(which_k_L));

%% Preallocate storage
% Per-experiment metrics
MSE_cum_P  = zeros(1, N_exp);   MSE_cum_L  = zeros(1, N_exp);
MSE_diff_P = zeros(1, N_exp);   MSE_diff_L = zeros(1, N_exp);
MSE_norm_P = zeros(1, N_exp);   MSE_norm_L = zeros(1, N_exp);

STD_cum_P  = zeros(1, N_exp);   STD_cum_L  = zeros(1, N_exp);
STD_diff_P = zeros(1, N_exp);   STD_diff_L = zeros(1, N_exp);
STD_norm_P = zeros(1, N_exp);   STD_norm_L = zeros(1, N_exp);

% Fisher information accumulators: sum of J'*J across experiments
% Power model: 4x4, Linear model: 6x6
n_P = numel(which_k_P);
n_L = numel(which_k_L);

FIM_cum_P  = zeros(n_P);   FIM_cum_L  = zeros(n_L);
FIM_diff_P = zeros(n_P);   FIM_diff_L = zeros(n_L);
FIM_norm_P = zeros(n_P);   FIM_norm_L = zeros(n_L);

SSE_cum_P  = 0;   SSE_cum_L  = 0;
SSE_diff_P = 0;   SSE_diff_L = 0;
SSE_norm_P = 0;   SSE_norm_L = 0;

N_total_cum  = 0;  % total observation count (for sigma^2 estimation)
N_total_diff = 0;
N_total_norm = 0;

%% Main loop
fprintf('Simulating %d experiments...\n\n', N_exp);

for jj = 1:N_exp
    fprintf('Experiment %d experiments...\n', jj);
    % Experimental data
    data_cum      = LabResults(6:19, jj+1)';        % cumulative yield at sample times
    data_diff     = diff(data_cum);                  % differentiated yield
    max_data_diff = max(data_diff);
    data_norm     = data_diff ./ max_data_diff;      % normalised differentiated yield

    % Operating conditions
    T = LabResults(2, jj+1);
    P = LabResults(3, jj+1) * 10;
    F = LabResults(4, jj+1) * 1e-5;

    % Fluid properties
    Z           = Compressibility(T, P, Parameters);
    rho         = rhoPB_Comp(T, P, Z, Parameters);
    enthalpy_rho = rho * SpecificEnthalpy(T, P, Z, rho, Parameters);

    % Initial state
    x0 = [C0fluid'; C0solid * bed_mask; enthalpy_rho * ones(nstages, 1); P; 0];

    % Input sequence
    feedTemp  = T * ones(1, N_Time);
    feedPress = P * ones(1, N_Time);
    feedFlow  = F * ones(1, N_Time);
    uu        = [feedTemp', feedPress', feedFlow'];

    % --- Build symbolic input vectors ---
    Parameters_sym_P              = MX(cell2mat(Parameters));
    Parameters_sym_P(which_k_P)   = k1;
    U_sym_P = [uu'; repmat(Parameters_sym_P, 1, N_Time)];

    Parameters_sym_L              = MX(cell2mat(Parameters));
    Parameters_sym_L(which_k_L)   = k2;
    U_sym_L = [uu'; repmat(Parameters_sym_L, 1, N_Time)];

    % --- Power model (symbolic) ---
    f_power = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, ...
        'Power_model', epsi_mask, one_minus_epsi_mask);
    F_power       = buildIntegrator(f_power, [Nx, Nu], timeStep_in_sec, 'cvodes');
    F_accum_power = F_power.mapaccum('F_accum_power', N_Time);

    X_power_sym   = F_accum_power(x0, U_sym_P);
    Y_cum_P_sym   = [0, X_power_sym(Nx, :)];
    Y_cum_P_sym   = Y_cum_P_sym(N_Sample);
    Y_diff_P_sym  = Y_cum_P_sym(2:end) - Y_cum_P_sym(1:end-1);
    Y_norm_P_sym  = Y_diff_P_sym ./ max_data_diff;

    J_cum_P_sym  = jacobian(Y_cum_P_sym,  k1);
    J_diff_P_sym = jacobian(Y_diff_P_sym, k1);
    J_norm_P_sym = jacobian(Y_norm_P_sym, k1);

    G_P = Function('G_P', {k1}, ...
        {Y_cum_P_sym, Y_diff_P_sym, Y_norm_P_sym, ...
         J_cum_P_sym, J_diff_P_sym, J_norm_P_sym});

    [Y_cum_P_v, Y_diff_P_v, Y_norm_P_v, ...
     J_cum_P_v, J_diff_P_v, J_norm_P_v] = G_P(theta_power);

    Y_cum_P   = full(Y_cum_P_v);
    Y_diff_P  = full(Y_diff_P_v);
    Y_norm_P  = full(Y_norm_P_v);
    JJ_cum_P  = full(J_cum_P_v);
    JJ_diff_P = full(J_diff_P_v);
    JJ_norm_P = full(J_norm_P_v);

    % --- Linear model (symbolic) ---
    f_linear = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, ...
        'Linear_model', epsi_mask, one_minus_epsi_mask);
    F_linear       = buildIntegrator(f_linear, [Nx, Nu], timeStep_in_sec, 'cvodes');
    F_accum_linear = F_linear.mapaccum('F_accum_linear', N_Time);

    X_linear_sym   = F_accum_linear(x0, U_sym_L);
    Y_cum_L_sym    = [0, X_linear_sym(Nx, :)];
    Y_cum_L_sym    = Y_cum_L_sym(N_Sample);
    Y_diff_L_sym   = Y_cum_L_sym(2:end) - Y_cum_L_sym(1:end-1);
    Y_norm_L_sym   = Y_diff_L_sym ./ max_data_diff;

    J_cum_L_sym  = jacobian(Y_cum_L_sym,  k2);
    J_diff_L_sym = jacobian(Y_diff_L_sym, k2);
    J_norm_L_sym = jacobian(Y_norm_L_sym, k2);

    G_L = Function('G_L', {k2}, ...
        {Y_cum_L_sym, Y_diff_L_sym, Y_norm_L_sym, ...
         J_cum_L_sym, J_diff_L_sym, J_norm_L_sym});

    [Y_cum_L_v, Y_diff_L_v, Y_norm_L_v, ...
     J_cum_L_v, J_diff_L_v, J_norm_L_v] = G_L(theta_linear);

    Y_cum_L   = full(Y_cum_L_v);
    Y_diff_L  = full(Y_diff_L_v);
    Y_norm_L  = full(Y_norm_L_v);
    JJ_cum_L  = full(J_cum_L_v);
    JJ_diff_L = full(J_diff_L_v);
    JJ_norm_L = full(J_norm_L_v);

    % --- Residuals ---
    res_cum_P  = Y_cum_P  - data_cum;
    res_cum_L  = Y_cum_L  - data_cum;
    res_diff_P = Y_diff_P - data_diff;
    res_diff_L = Y_diff_L - data_diff;
    res_norm_P = Y_norm_P - data_norm;
    res_norm_L = Y_norm_L - data_norm;

    % --- Per-experiment statistics ---
    MSE_cum_P(jj)  = mean(res_cum_P.^2);    MSE_cum_L(jj)  = mean(res_cum_L.^2);
    MSE_diff_P(jj) = mean(res_diff_P.^2);   MSE_diff_L(jj) = mean(res_diff_L.^2);
    MSE_norm_P(jj) = mean(res_norm_P.^2);   MSE_norm_L(jj) = mean(res_norm_L.^2);

    STD_cum_P(jj)  = std(res_cum_P);        STD_cum_L(jj)  = std(res_cum_L);
    STD_diff_P(jj) = std(res_diff_P);       STD_diff_L(jj) = std(res_diff_L);
    STD_norm_P(jj) = std(res_norm_P);       STD_norm_L(jj) = std(res_norm_L);

    % --- Accumulate Fisher information and SSE ---
    FIM_cum_P  = FIM_cum_P  + JJ_cum_P'  * JJ_cum_P;
    FIM_cum_L  = FIM_cum_L  + JJ_cum_L'  * JJ_cum_L;
    FIM_diff_P = FIM_diff_P + JJ_diff_P' * JJ_diff_P;
    FIM_diff_L = FIM_diff_L + JJ_diff_L' * JJ_diff_L;
    FIM_norm_P = FIM_norm_P + JJ_norm_P' * JJ_norm_P;
    FIM_norm_L = FIM_norm_L + JJ_norm_L' * JJ_norm_L;

    SSE_cum_P  = SSE_cum_P  + sum(res_cum_P.^2);
    SSE_cum_L  = SSE_cum_L  + sum(res_cum_L.^2);
    SSE_diff_P = SSE_diff_P + sum(res_diff_P.^2);
    SSE_diff_L = SSE_diff_L + sum(res_diff_L.^2);
    SSE_norm_P = SSE_norm_P + sum(res_norm_P.^2);
    SSE_norm_L = SSE_norm_L + sum(res_norm_L.^2);

    N_total_cum  = N_total_cum  + numel(res_cum_P);
    N_total_diff = N_total_diff + numel(res_diff_P);
    N_total_norm = N_total_norm + numel(res_norm_P);
end

%% Parameter covariance matrices: Cov_theta = sigma^2 * inv(J'J)
% sigma^2 estimated as SSE / (N_total - n_params)

sigma2_cum_P  = SSE_cum_P  / (N_total_cum  - n_P);
sigma2_cum_L  = SSE_cum_L  / (N_total_cum  - n_L);
sigma2_diff_P = SSE_diff_P / (N_total_diff - n_P);
sigma2_diff_L = SSE_diff_L / (N_total_diff - n_L);
sigma2_norm_P = SSE_norm_P / (N_total_norm - n_P);
sigma2_norm_L = SSE_norm_L / (N_total_norm - n_L);

Cov_theta_cum_P  = sigma2_cum_P  * inv(FIM_cum_P);
Cov_theta_cum_L  = sigma2_cum_L  * inv(FIM_cum_L);
Cov_theta_diff_P = sigma2_diff_P * inv(FIM_diff_P);
Cov_theta_diff_L = sigma2_diff_L * inv(FIM_diff_L);
Cov_theta_norm_P = sigma2_norm_P * inv(FIM_norm_P);
Cov_theta_norm_L = sigma2_norm_L * inv(FIM_norm_L);

%% Print Summary
fprintf('\n=============================================================================\n');
fprintf('   FIT SUMMARY\n');
fprintf('=============================================================================\n\n');

% --- Cumulative yield ---
fprintf('--- Cumulative Yield ---\n');
fprintf('%4s | %12s %12s | %12s %12s\n', ...
    'Exp', 'MSE_P', 'MSE_L', 'STD_P', 'STD_L');
fprintf('%s\n', repmat('-', 1, 60));
for jj = 1:N_exp
    fprintf('%4d | %12.4e %12.4e | %12.4e %12.4e\n', ...
        jj, MSE_cum_P(jj), MSE_cum_L(jj), ...
        STD_cum_P(jj), STD_cum_L(jj));
end
fprintf('%4s | %12.4e %12.4e | %12.4e %12.4e\n', ...
    'Avg', mean(MSE_cum_P), mean(MSE_cum_L), ...
    mean(STD_cum_P), mean(STD_cum_L));

% --- Differentiated yield ---
fprintf('\n--- Differentiated Yield ---\n');
fprintf('%4s | %12s %12s | %12s %12s\n', ...
    'Exp', 'MSE_P', 'MSE_L', 'STD_P', 'STD_L');
fprintf('%s\n', repmat('-', 1, 60));
for jj = 1:N_exp
    fprintf('%4d | %12.4e %12.4e | %12.4e %12.4e\n', ...
        jj, MSE_diff_P(jj), MSE_diff_L(jj), ...
        STD_diff_P(jj), STD_diff_L(jj));
end
fprintf('%4s | %12.4e %12.4e | %12.4e %12.4e\n', ...
    'Avg', mean(MSE_diff_P), mean(MSE_diff_L), ...
    mean(STD_diff_P), mean(STD_diff_L));

% --- Normalised yield ---
fprintf('\n--- Normalised Yield ---\n');
fprintf('%4s | %12s %12s | %12s %12s\n', ...
    'Exp', 'MSE_P', 'MSE_L', 'STD_P', 'STD_L');
fprintf('%s\n', repmat('-', 1, 56));
for jj = 1:N_exp
    fprintf('%4d | %12.4e %12.4e | %12.4e %12.4e\n', ...
        jj, MSE_norm_P(jj), MSE_norm_L(jj), ...
        STD_norm_P(jj), STD_norm_L(jj));
end
fprintf('%4s | %12.4e %12.4e | %12.4e %12.4e\n', ...
    'Avg', mean(MSE_norm_P), mean(MSE_norm_L), ...
    mean(STD_norm_P), mean(STD_norm_L));

% --- Parameter covariance matrices ---
fprintf('\n--- Parameter Covariance Matrices: Cov_theta = sigma^2 * inv(J''J) ---\n');

param_names_P = {'k_w0', 'a_w', 'b_w', 'n'};
param_names_L = {'D_i0', 'D_iRe', 'D_iF', 'Ups0', 'UpsRe', 'UpsF'};

print_cov_summary('Power',  'Cumulative',    Cov_theta_cum_P,  sigma2_cum_P,  param_names_P);
print_cov_summary('Linear', 'Cumulative',    Cov_theta_cum_L,  sigma2_cum_L,  param_names_L);
print_cov_summary('Power',  'Differentiated',Cov_theta_diff_P, sigma2_diff_P, param_names_P);
print_cov_summary('Linear', 'Differentiated',Cov_theta_diff_L, sigma2_diff_L, param_names_L);
print_cov_summary('Power',  'Normalised',    Cov_theta_norm_P, sigma2_norm_P, param_names_P);
print_cov_summary('Linear', 'Normalised',    Cov_theta_norm_L, sigma2_norm_L, param_names_L);

% --- Overall verdict ---
fprintf('\n=============================================================================\n');
fprintf('   OVERALL COMPARISON\n');
fprintf('=============================================================================\n');
fprintf('                        Power           Linear\n');
fprintf('  MSE  (cumulative)   %12.4e    %12.4e\n', mean(MSE_cum_P),  mean(MSE_cum_L));
fprintf('  MSE  (diff)         %12.4e    %12.4e\n', mean(MSE_diff_P), mean(MSE_diff_L));
fprintf('  MSE  (normalised)   %12.4e    %12.4e\n', mean(MSE_norm_P), mean(MSE_norm_L));
fprintf('  STD  (cumulative)   %12.4e    %12.4e\n', mean(STD_cum_P),  mean(STD_cum_L));
fprintf('  STD  (diff)         %12.4e    %12.4e\n', mean(STD_diff_P), mean(STD_diff_L));
fprintf('  STD  (normalised)   %12.4e    %12.4e\n', mean(STD_norm_P), mean(STD_norm_L));
fprintf('=============================================================================\n');

%% =========================================================================
function print_cov_summary(model, yield_type, Cov_theta, sigma2, param_names)
    n = size(Cov_theta, 1);
    std_theta = sqrt(diag(Cov_theta));

    fprintf('\n%s model - %s yield (sigma^2 = %.4e):\n', model, yield_type, sigma2);
    fprintf('  %-8s  %12s  %12s\n', 'Param', 'Std Dev', '95%% CI half');
    fprintf('  %s\n', repmat('-', 1, 36));
    for ii = 1:n
        fprintf('  %-8s  %12.4e  %12.4e\n', param_names{ii}, std_theta(ii), 1.96*std_theta(ii));
    end
    fprintf('  Cond(Cov) = %.2e\n', cond(Cov_theta));

    % Print full covariance matrix
    header = '          ';
    for ii = 1:n
        header = [header, sprintf(' %10s', param_names{ii})]; %#ok<AGROW>
    end
    fprintf('%s\n', header);
    for ii = 1:n
        row_str = sprintf('  %-8s', param_names{ii});
        for kk = 1:n
            row_str = [row_str, sprintf(' %10.4e', Cov_theta(ii, kk))]; %#ok<AGROW>
        end
        fprintf('%s\n', row_str);
    end
end
