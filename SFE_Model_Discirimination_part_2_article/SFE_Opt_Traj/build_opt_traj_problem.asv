function problem = build_opt_traj_problem(static, config, verbose)
%BUILD_OPT_TRAJ_PROBLEM Build CasADi Opti problem and symbolic outputs once.

if nargin < 3 || isempty(verbose)
    verbose = false;
end

t_build = tic;

addpath(config.casadi_path);
import casadi.*

%% Optimizer Settings
nlp_opts = struct;
nlp_opts.print_time                   = false;  % Disable CasADi timing table
nlp_opts.ipopt.max_iter               = config.ipopt.max_iter;
nlp_opts.ipopt.tol                    = config.ipopt.tol;
nlp_opts.ipopt.acceptable_tol         = config.ipopt.acceptable_tol;
nlp_opts.ipopt.acceptable_iter        = config.ipopt.acceptable_iter;
nlp_opts.ipopt.hessian_approximation  = config.ipopt.hessian_approximation;
nlp_opts.ipopt.sb                     = config.ipopt.sb;
nlp_opts.ipopt.print_level            = 5 * double(verbose);

if verbose
    fprintf('=== Optimizer Configuration ===\n');
    fprintf('Max iterations: %d\n', nlp_opts.ipopt.max_iter);
    fprintf('Convergence tolerance: %.0e\n', nlp_opts.ipopt.tol);
end

opti = casadi.Opti();
opti.solver('ipopt', nlp_opts);

%% Parameters to estimate
k1 = opti.parameter(4);
k2 = opti.parameter(6);
k  = [k1; k2];

%% Decision variables (normalized)
zFeedTemp = opti.variable(1, static.N_Time);  % normalized to [-1, 1]
zFeedFlow = opti.variable(1, static.N_Time);  % normalized to [-1, 1]

feedTemp = static.T_mid + static.T_half * zFeedTemp;
feedFlow = static.F_mid + static.F_half * zFeedFlow;
feedPress = static.feedPress;

T = feedTemp(1);
P = feedPress(1);

uu = [feedTemp', feedPress', feedFlow'];

%% Fluid properties and initial state
Z            = Compressibility(T, P, static.Parameters);
rho          = rhoPB_Comp(T, P, Z, static.Parameters);
enthalpy_rho = rho * SpecificEnthalpy(T, P, Z, rho, static.Parameters);

x0 = [static.C0fluid'; ...
      static.C0solid * static.bed_mask; ...
      enthalpy_rho * ones(static.nstages, 1); ...
      P; 0];

Parameters_sym = MX(cell2mat(static.Parameters));
Parameters_sym(static.which_k) = k(1:numel(static.which_k));
U_base = [uu'; repmat(Parameters_sym, 1, static.N_Time)];

%% Build integrators (expensive - done once per problem build)
f_power_nom = @(x, u) modelSFE(x, u, static.bed_mask, static.timeStep_in_sec, ...
    'Power_model', static.epsi_mask, static.one_minus_epsi_mask);
F_power_nom       = buildIntegrator(f_power_nom, [static.Nx, static.Nu], static.timeStep_in_sec, 'cvodes');
F_accum_power_nom = F_power_nom.mapaccum('F_accum_power', static.N_Time);

f_linear_nom = @(x, u) modelSFE(x, u, static.bed_mask, static.timeStep_in_sec, ...
    'Linear_model', static.epsi_mask, static.one_minus_epsi_mask);
F_linear_nom       = buildIntegrator(f_linear_nom, [static.Nx, static.Nu], static.timeStep_in_sec, 'cvodes');
F_accum_linear_nom = F_linear_nom.mapaccum('F_accum_linear', static.N_Time);

X_power_nom  = F_accum_power_nom(x0, U_base);
X_linear_nom = F_accum_linear_nom(x0, U_base);

%% Yield outputs at sample times
Y_cum_P_sym = [0, X_power_nom(static.Nx, :)];
Y_cum_P_sym = Y_cum_P_sym(static.N_Sample);
Y_cum_L_sym = [0, X_linear_nom(static.Nx, :)];
Y_cum_L_sym = Y_cum_L_sym(static.N_Sample);

Y_diff_P_sym = Y_cum_P_sym(2:end) - Y_cum_P_sym(1:end-1); %#ok<NASGU>
Y_diff_L_sym = Y_cum_L_sym(2:end) - Y_cum_L_sym(1:end-1); %#ok<NASGU>

% Currently optimize on cumulative yields
Y_P_sym = Y_cum_P_sym;
Y_L_sym = Y_cum_L_sym;

%% Discrimination metric
J_P_sym = jacobian(Y_P_sym, k1);
J_L_sym = jacobian(Y_L_sym, k2);

residuals = Y_P_sym - Y_L_sym;

n = numel(Y_P_sym);
I = MX.eye(n);

Sigma_r_P = static.sigma2_cases(1) * I + J_P_sym * static.Cov_power_cum  * J_P_sym';
Sigma_r_L = static.sigma2_cases(1) * I + J_L_sym * static.Cov_linear_cum * J_L_sym';

eps_reg   = 1e-10;
Sigma_r_P = Sigma_r_P + eps_reg * I;
Sigma_r_L = Sigma_r_L + eps_reg * I;

j_1 = trace(Sigma_r_P * (Sigma_r_L\I) + Sigma_r_L * (Sigma_r_P\I) - 2*I);
j_2 = residuals * ((Sigma_r_P\I) + (Sigma_r_L\I)) * residuals';

if config.use_full_discrimination_objective
    j = j_1 + j_2;
else
    j = residuals(end).^2;
end
j = j * config.objective_scale;

%% Constraints and fixed parameter values
opti.subject_to(zFeedTemp >= -1);
opti.subject_to(zFeedTemp <= 1);
opti.subject_to(zFeedFlow >= -1);
opti.subject_to(zFeedFlow <= 1);

opti.set_value(k1, static.k1_val);
opti.set_value(k2, static.k2_val);

opti.minimize(-j);

problem = struct;
problem.opti         = opti;
problem.zFeedTemp    = zFeedTemp;
problem.zFeedFlow    = zFeedFlow;
problem.feedTemp     = feedTemp;
problem.feedFlow     = feedFlow;
problem.j            = j;
problem.j_1          = j_1;
problem.j_2          = j_2;
problem.Y_P_sym      = Y_P_sym;
problem.Y_L_sym      = Y_L_sym;
problem.Sigma_r_P    = Sigma_r_P;
problem.Sigma_r_L    = Sigma_r_L;
problem.static       = static;
problem.verbose      = verbose;

problem.timings = struct;
problem.timings.t_problem_build = toc(t_build);

end
