function [D_stats] = compute_discrimination_with_uncertainty(T0, P0, F0, Parameters, ...
    F_accum_power, F_accum_linear, Cov_power, Cov_linear_full, ...
    theta_power, theta_Di, theta_Upsilon, ...
    x0, N_Time, Nx, Time, N_MC)
% COMPUTE_DISCRIMINATION_WITH_UNCERTAINTY
% Computes discrimination metrics between Power and Linear models with
% uncertainty propagation using first-order Taylor expansion (delta method)
% from parameter variance-covariance matrices.
%
% Inputs:
%   T0, P0, F0     - Operating conditions (Temperature [K], Pressure [bar], Flow [m³/s])
%   Parameters     - Cell array of fixed parameters
%   F_accum_power  - CasADi mapaccum integrator for Power model
%   F_accum_linear - CasADi mapaccum integrator for Linear model
%   Cov_power      - 4x4 covariance matrix for Power model [k_w0, a_w, b_w, n_k]
%   Cov_linear_full - 6x6 FULL covariance matrix for Linear model
%                     [D_i(0), D_i(Re), D_i(F), Ups(0), Ups(Re), Ups(F)]
%                     This includes cross-covariances between D_i and Upsilon parameters
%   theta_power    - 4x1 optimal Power model parameters
%   theta_Di       - 3x1 optimal Diffusion parameters [D_i(0), Re_coef, F_coef]
%   theta_Upsilon  - 3x1 optimal Decay parameters [Ups(0), Re_coef, F_coef]
%   x0             - Initial state vector
%   N_Time         - Number of time steps
%   Nx             - Number of states
%   Time           - Time vector [min]
%   N_MC           - Number of Monte Carlo samples (reserved for future use)
%
% Outputs:
%   D_stats        - Structure with discrimination statistics

import casadi.*

%% Initialize output structure
D_stats = struct();

%% Compute fluid properties at operating conditions
Z            = Compressibility(T0, P0, Parameters);
rho          = rhoPB_Comp(T0, P0, Z, Parameters);
enthalpy_rho = rho .* SpecificEnthalpy(T0, P0, Z, rho, Parameters);

%% Build input vectors
feedTemp  = T0 * ones(1, N_Time);
feedPress = P0 * ones(1, N_Time);
feedFlow  = F0 * ones(1, N_Time);

uu = [feedTemp', feedPress', feedFlow'];
U_base = [uu'; repmat(cell2mat(Parameters), 1, N_Time)];

%% Baseline simulation (nominal parameters)
try
    X_power_nom  = F_accum_power(x0, U_base);
    X_linear_nom = F_accum_linear(x0, U_base);

    Y_power_nom  = full([0, X_power_nom(Nx, :)]);
    Y_linear_nom = full([0, X_linear_nom(Nx, :)]);
catch ME
    warning('Baseline simulation failed: %s', ME.message);
    D_stats.valid = false;
    return;
end

%% Cholesky decomposition for sampling (reserved for future MC)
try
    L_power = chol(Cov_power + 1e-10*eye(4), 'lower');
    L_linear = chol(Cov_linear_full + 1e-10*eye(6), 'lower');
catch ME
    warning('Cholesky decomposition failed, using diagonal approximation');
    L_power = diag(sqrt(diag(Cov_power)));
    L_linear = diag(sqrt(diag(Cov_linear_full)));
end

%% Monte Carlo sampling for uncertainty propagation
Y_power_samples  = zeros(N_MC, length(Time));
Y_linear_samples = zeros(N_MC, length(Time));

% Note: For full uncertainty propagation, we would need to modify the
% model parameters in the CasADi function. Since the current implementation
% has hardcoded parameters, we use a linearized approximation based on
% sensitivity analysis.

% Compute sensitivities numerically (finite differences)
delta_rel = 1e-4;

%% Sensitivity for Power model yield
% The Power model extraction rate: re = k_w * beta * C_solid
% where k_w = k_w0 * (rho/800)^a_w * (F/5)^b_w * 1e-4
%       beta = 1 / (alpha + 1)^n

% Approximate sensitivity of final yield to parameters
% dY/d(theta) at nominal conditions

% For now, use analytical approximations based on model structure
% Power model: Y depends on [k_w0, a_w, b_w, n]
%   dY/d(k_w0) ~ Y / k_w0 (linear scaling)
%   dY/d(a_w)  ~ Y * ln(rho/800)
%   dY/d(b_w)  ~ Y * ln(F/5)
%   dY/d(n)    ~ complex (affects extraction dynamics)

% Compute Reynolds number for Linear model sensitivity
nstages = size(x0, 1) - 2;  % Approximate
MU = Viscosity(T0, rho);
dp = Parameters{5};
V_approx = F0 / (pi * Parameters{3}^2);  % Approximate velocity
Re_approx = dp * rho * V_approx / MU * 1.3;

%% Analytical variance propagation (first-order Taylor expansion)
% For Power model:
%   Y = f(k_w0, a_w, b_w, n)
%   Var(Y) ≈ J_power * Cov_power * J_power'
%   where J_power = [dY/dk_w0, dY/da_w, dY/db_w, dY/dn]

% Compute Jacobian elements for Power model (at final time)
Y_final_power = Y_power_nom(end);
k_w0 = theta_power(1);
a_w  = theta_power(2);
b_w  = theta_power(3);
n_k  = theta_power(4);

% Approximate sensitivities (log-linear approximations)
dY_dk_w0 = Y_final_power / k_w0;
dY_da_w  = Y_final_power * log(max(rho/800, 0.1));
dY_db_w  = Y_final_power * log(max(F0*1e5/5, 0.1));
dY_dn    = -Y_final_power * 0.3;  % Approximate (decay effect)

J_power = [dY_dk_w0, dY_da_w, dY_db_w, dY_dn];
Var_Y_power = J_power * Cov_power * J_power';
Std_Y_power = sqrt(max(Var_Y_power, 0));

%% For Linear model:
% D_i = a + b*Re + c*F*1e5
% Upsilon = a + b*Re + c*F*1e5
% re = (Sat_coe / mi / lp²) * C_solid
% Sat_coe = D_i * exp(-Upsilon * alpha)

Y_final_linear = Y_linear_nom(end);
a_Di = theta_Di(1);
b_Di = theta_Di(2);
c_Di = theta_Di(3);
a_Ups = theta_Upsilon(1);
b_Ups = theta_Upsilon(2);
c_Ups = theta_Upsilon(3);

% Compute D_i and Upsilon at operating conditions
D_i_nom = a_Di + b_Di * Re_approx + c_Di * F0 * 1e5;
Ups_nom = a_Ups + b_Ups * Re_approx + c_Ups * F0 * 1e5;

% Sensitivities for Linear model using FULL 6x6 covariance matrix
% Parameters: [D_i(0), D_i(Re), D_i(F), Ups(0), Ups(Re), Ups(F)]
%
% dY/d(D_i params): Y scales roughly with D_i (diffusion coefficient)
% dY/d(Ups params): Upsilon affects decay, negative effect on yield

% Build the FULL Jacobian vector (1x6) for Linear model
% J_linear = [dY/dD_i(0), dY/dD_i(Re), dY/dD_i(F), dY/dUps(0), dY/dUps(Re), dY/dUps(F)]

scale_factor_Di = Y_final_linear / max(D_i_nom, 1e-6);
scale_factor_Ups = -Y_final_linear * 0.2;  % Decay effect (negative)

% Partial derivatives w.r.t. D_i parameters
dY_dDi0   = scale_factor_Di * 1;
dY_dDiRe  = scale_factor_Di * Re_approx;
dY_dDiF   = scale_factor_Di * F0 * 1e5;

% Partial derivatives w.r.t. Upsilon parameters
dY_dUps0  = scale_factor_Ups * 1;
dY_dUpsRe = scale_factor_Ups * Re_approx;
dY_dUpsF  = scale_factor_Ups * F0 * 1e5;

% Full Jacobian (1x6)
J_linear_full = [dY_dDi0, dY_dDiRe, dY_dDiF, dY_dUps0, dY_dUpsRe, dY_dUpsF];

% Variance using FULL covariance matrix (includes cross-covariances)
% Var(Y) = J * Cov * J'
Var_Y_linear = J_linear_full * Cov_linear_full * J_linear_full';
Std_Y_linear = sqrt(max(Var_Y_linear, 0));

% Also compute component variances for diagnostics
J_Di = [dY_dDi0, dY_dDiRe, dY_dDiF];
J_Ups = [dY_dUps0, dY_dUpsRe, dY_dUpsF];
Cov_linear_Di = Cov_linear_full(1:3, 1:3);
Cov_linear_Ups = Cov_linear_full(4:6, 4:6);
Cov_linear_cross = Cov_linear_full(1:3, 4:6);

Var_Y_linear_Di  = J_Di * Cov_linear_Di * J_Di';
Var_Y_linear_Ups = J_Ups * Cov_linear_Ups * J_Ups';
Var_Y_linear_cross = 2 * J_Di * Cov_linear_cross * J_Ups';  % Cross-term contribution

%% Compute discrimination metrics
% 1. Absolute difference
diff_nominal = Y_power_nom - Y_linear_nom;
D_stats.diff_trajectory = diff_nominal;

% 2. Maximum absolute difference
[D_stats.max_diff, idx_max] = max(abs(diff_nominal));
D_stats.t_max_diff = Time(idx_max);

% 3. Integrated absolute difference
D_stats.integrated_diff = trapz(Time, abs(diff_nominal));

% 4. Final yield difference
D_stats.final_diff = Y_final_power - Y_final_linear;

% 5. Pooled standard error for t-test
Std_pooled = sqrt(Std_Y_power^2 + Std_Y_linear^2);
D_stats.std_pooled = Std_pooled;

% 6. T-statistic for final yield difference
if Std_pooled > 1e-10
    D_stats.t_statistic = abs(D_stats.final_diff) / Std_pooled;
else
    D_stats.t_statistic = Inf;
end

% 7. Weighted discrimination (difference / uncertainty)
% This gives higher values where models differ AND we're confident
D_stats.weighted_discrimination = D_stats.integrated_diff / max(Std_pooled, 1e-6);

% 8. Coefficient of variation for each model
D_stats.CV_power  = Std_Y_power / max(Y_final_power, 1e-6);
D_stats.CV_linear = Std_Y_linear / max(Y_final_linear, 1e-6);

% 9. Store nominal trajectories
D_stats.Y_power_nom  = Y_power_nom;
D_stats.Y_linear_nom = Y_linear_nom;

% 10. Store uncertainties
D_stats.Std_Y_power  = Std_Y_power;
D_stats.Std_Y_linear = Std_Y_linear;

% 11. Operating conditions
D_stats.T = T0;
D_stats.P = P0;
D_stats.F = F0;
D_stats.rho = rho;
D_stats.Re = Re_approx;

% 12. Variance decomposition for Linear model (diagnostics)
D_stats.Var_Y_linear_Di    = Var_Y_linear_Di;     % Contribution from D_i parameters
D_stats.Var_Y_linear_Ups   = Var_Y_linear_Ups;    % Contribution from Upsilon parameters
D_stats.Var_Y_linear_cross = Var_Y_linear_cross;  % Cross-covariance contribution
D_stats.Var_Y_linear_total = Var_Y_linear;        % Total variance

% 13. Jacobian vectors for sensitivity analysis
D_stats.J_power = J_power;
D_stats.J_linear_full = J_linear_full;

% 14. Nominal parameter values at operating conditions
D_stats.D_i_nom = D_i_nom;
D_stats.Ups_nom = Ups_nom;

D_stats.valid = true;

end
