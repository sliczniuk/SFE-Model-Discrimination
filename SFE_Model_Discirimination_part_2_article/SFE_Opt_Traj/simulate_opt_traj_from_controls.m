function out = simulate_opt_traj_from_controls(static, config, controls, seed, j_values, timings)
%SIMULATE_OPT_TRAJ_FROM_CONTROLS Evaluate model outputs/CI for fixed controls.
% This function does not call IPOPT solve; it only evaluates model expressions.

if nargin < 4 || isempty(seed)
    seed = NaN;
end
if nargin < 5 || isempty(j_values)
    j_values = struct;
end
if nargin < 6 || isempty(timings)
    timings = struct;
end

if ~isfield(controls, 'feedTemp') || ~isfield(controls, 'feedFlow')
    error('controls must contain feedTemp and feedFlow.');
end

feedTemp = reshape(full(controls.feedTemp), 1, []);
feedFlow = reshape(full(controls.feedFlow), 1, []);
if numel(feedTemp) ~= static.N_Time || numel(feedFlow) ~= static.N_Time
    error('Control vector lengths must match N_Time=%d.', static.N_Time);
end

t_eval = tic;
problem = build_opt_traj_problem(static, config, false);
opti = problem.opti;

zFeedTemp = (feedTemp - static.T_mid) / static.T_half;
zFeedFlow = (feedFlow - static.F_mid) / static.F_half;
opti.set_initial(problem.zFeedTemp, zFeedTemp);
opti.set_initial(problem.zFeedFlow, zFeedFlow);

valfun = @(x) opti.debug.value(x);
j_eval = local_try_eval(valfun, problem.j, NaN);
Y_P_num = local_try_eval(valfun, problem.Y_P_sym, []);
Y_L_num = local_try_eval(valfun, problem.Y_L_sym, []);
Sigma_r_P_n = local_try_eval(valfun, problem.Sigma_r_P, []);
Sigma_r_L_n = local_try_eval(valfun, problem.Sigma_r_L, []);

ci_P = [];
ci_L = [];
if ~isempty(Sigma_r_P_n) && ~isempty(Sigma_r_L_n)
    Sigma_r_P_n = full(Sigma_r_P_n);
    Sigma_r_L_n = full(Sigma_r_L_n);
    Sigma_r_P_n = 0.5 * (Sigma_r_P_n + Sigma_r_P_n.');
    Sigma_r_L_n = 0.5 * (Sigma_r_L_n + Sigma_r_L_n.');
    z95 = 1.96;
    ci_P = z95 * sqrt(max(diag(Sigma_r_P_n), 0));
    ci_L = z95 * sqrt(max(diag(Sigma_r_L_n), 0));
end

out = struct;
out.seed = seed;
out.success = local_get_or_default(j_values, 'success', true);
out.j_initial = local_get_or_default(j_values, 'j_initial', NaN);
out.j_final = local_get_or_default(j_values, 'j_final', NaN);
out.j = local_get_or_default(j_values, 'j', j_eval);
out.j_simulated = j_eval;
out.status = local_get_or_default(j_values, 'status', 'simulated_from_screening');
out.iter_count = local_get_or_default(j_values, 'iter_count', NaN);
out.error_message = local_get_or_default(j_values, 'error_message', '');

out.feedTemp = feedTemp;
out.feedFlow = feedFlow;
out.feedTemp0 = local_get_or_default(controls, 'feedTemp0', []);
out.feedFlow0 = local_get_or_default(controls, 'feedFlow0', []);

out.Time = static.Time;
out.yieldTime = static.Time(static.N_Sample);
out.yieldPower = local_row(Y_P_num);
out.yieldLinear = local_row(Y_L_num);
out.yieldPowerCI = local_row(ci_P);
out.yieldLinearCI = local_row(ci_L);

out.t_setup = local_get_or_default(timings, 't_setup', NaN);
out.t_file_io = local_get_or_default(timings, 't_file_io', NaN);
out.t_problem_build = local_get_or_default(timings, 't_problem_build', NaN);
out.t_solve = local_get_or_default(timings, 't_solve', NaN);
out.t_post = toc(t_eval);
out.t_total = local_get_or_default(timings, 't_total', NaN);
end

function y = local_try_eval(valfun, expr, fallback)
try
    y = full(valfun(expr));
catch
    y = fallback;
end
end

function y = local_row(x)
if isempty(x)
    y = [];
else
    y = reshape(full(x), 1, []);
end
end

function v = local_get_or_default(s, field_name, default_value)
if isstruct(s) && isfield(s, field_name)
    v = s.(field_name);
else
    v = default_value;
end
end
