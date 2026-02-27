function out = solve_opt_traj_problem_seed(problem, init, options)
%SOLVE_OPT_TRAJ_PROBLEM_SEED Solve one seed on a pre-built problem.

if nargin < 3
    options = struct;
end
if ~isfield(options, 'output_detail') || isempty(options.output_detail)
    options.output_detail = 'summary';
end
if ~isfield(options, 'include_stats') || isempty(options.include_stats)
    options.include_stats = strcmpi(options.output_detail, 'full');
end
if ~isfield(options, 'collect_timings') || isempty(options.collect_timings)
    options.collect_timings = false;
end
if ~isfield(options, 't_setup') || isempty(options.t_setup)
    options.t_setup = NaN;
end
if ~isfield(options, 't_file_io') || isempty(options.t_file_io)
    options.t_file_io = NaN;
end
if ~isfield(options, 't_problem_build') || isempty(options.t_problem_build)
    options.t_problem_build = NaN;
end

t_total = tic;
opti = problem.opti;

% Reset all decision variable initials explicitly to preserve seed independence.
opti.set_initial(problem.zFeedTemp, init.zFeedTemp0);
opti.set_initial(problem.zFeedFlow, init.zFeedFlow0);

% Objective value at the initial guess (before optimization)
debugfun = @(x) opti.debug.value(x);
j_initial = local_try_eval(debugfun, problem.j, NaN);

t_solve = tic;
error_message = '';
try
    sol = opti.solve();
    valfun = @(x) sol.value(x);
    success = true;
catch err
    valfun = @(x) opti.debug.value(x);
    success = false;
    error_message = err.message;
end
t_solve_elapsed = toc(t_solve);

t_post = tic;
obj = local_try_eval(valfun, problem.j, NaN);
j_final = obj;

stats = opti.stats();
status_text = '';
iter_count = NaN;
if isstruct(stats)
    if isfield(stats, 'return_status')
        status_text = local_to_char(stats.return_status);
    end
    if isfield(stats, 'iter_count')
        iter_count = stats.iter_count;
    end
end

out = struct;
out.seed          = init.seed;
out.success       = success;
out.j_initial     = j_initial;
out.j_final       = j_final;
out.j             = j_final;  % backward-compatible alias
out.status        = status_text;
out.iter_count    = iter_count;
out.error_message = error_message;
out.t_setup       = options.t_setup;
out.t_file_io     = options.t_file_io;
out.t_problem_build = options.t_problem_build;
out.t_solve       = t_solve_elapsed;
out.t_post        = NaN;
out.t_total       = NaN;

if strcmpi(options.output_detail, 'full')
    K_out = local_try_eval(valfun, [problem.feedTemp; problem.feedFlow], []);
    if ~isempty(K_out)
        K_out = full(K_out);
    end

    Y_P_num     = local_try_eval(valfun, problem.Y_P_sym, []);
    Y_L_num     = local_try_eval(valfun, problem.Y_L_sym, []);
    Sigma_r_P_n = local_try_eval(valfun, problem.Sigma_r_P, []);
    Sigma_r_L_n = local_try_eval(valfun, problem.Sigma_r_L, []);

    if ~isempty(Sigma_r_P_n) && ~isempty(Sigma_r_L_n)
        Sigma_r_P_n = full(Sigma_r_P_n);
        Sigma_r_L_n = full(Sigma_r_L_n);
        Sigma_r_P_n = 0.5 * (Sigma_r_P_n + Sigma_r_P_n.');
        Sigma_r_L_n = 0.5 * (Sigma_r_L_n + Sigma_r_L_n.');
        z95 = 1.96;
        ci_P = z95 * sqrt(max(diag(Sigma_r_P_n), 0));
        ci_L = z95 * sqrt(max(diag(Sigma_r_L_n), 0));
    else
        ci_P = [];
        ci_L = [];
    end

    Y_P_num = local_row(Y_P_num);
    Y_L_num = local_row(Y_L_num);
    ci_P    = local_row(ci_P);
    ci_L    = local_row(ci_L);

    out.feedTemp      = [];
    out.feedFlow      = [];
    if ~isempty(K_out)
        out.feedTemp = K_out(1, :);
        out.feedFlow = K_out(2, :);
    end
    out.feedTemp0     = init.feedTemp0;
    out.feedFlow0     = init.feedFlow0;
    out.Time          = problem.static.Time;
    out.yieldTime     = problem.static.Time(problem.static.N_Sample);
    out.yieldPower    = Y_P_num;
    out.yieldLinear   = Y_L_num;
    out.yieldPowerCI  = ci_P;
    out.yieldLinearCI = ci_L;

    if options.include_stats
        out.stats = stats;
    end
else
    if options.include_stats
        out.stats = stats;
    end
end

t_post_elapsed = toc(t_post);
out.t_post  = t_post_elapsed;
out.t_total = toc(t_total);

if options.collect_timings
    out.timings = struct( ...
        't_setup', out.t_setup, ...
        't_file_io', out.t_file_io, ...
        't_problem_build', out.t_problem_build, ...
        't_solve', out.t_solve, ...
        't_postprocess', out.t_post, ...
        't_total', out.t_total);
end

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

function c = local_to_char(x)
if isstring(x)
    c = char(x);
elseif ischar(x)
    c = x;
else
    c = char(string(x));
end
end
