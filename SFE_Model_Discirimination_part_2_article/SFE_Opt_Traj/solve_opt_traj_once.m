function out = solve_opt_traj_once(seed, verbose)
%SOLVE_OPT_TRAJ_ONCE Backward-compatible single-seed full solve wrapper.
%
%   out = solve_opt_traj_once(seed)
%   out = solve_opt_traj_once(seed, verbose)
%
% Returns a full result struct including optimized trajectories and yield CI.

if nargin < 1 || isempty(seed)
    seed = 1;
end
if nargin < 2 || isempty(verbose)
    verbose = false;
end

config = opt_traj_default_config();

if verbose
    fprintf('=============================================================================\n');
    fprintf('   Optimal trajectory for the model discrimination (seed %d)\n', seed);
    fprintf('=============================================================================\n\n');
end

t_total = tic;

static = load_opt_traj_static_data(config);
problem = build_opt_traj_problem(static, config, verbose);
init = make_seed_initial_guess(seed, problem, config);

seed_options = struct;
seed_options.output_detail   = 'full';
seed_options.include_stats   = true;
seed_options.collect_timings = true;
seed_options.t_setup         = static.timings.t_static_total + problem.timings.t_problem_build;
seed_options.t_file_io       = static.timings.t_file_io;
seed_options.t_problem_build = problem.timings.t_problem_build;

out = solve_opt_traj_problem_seed(problem, init, seed_options);

% Keep a richer timing breakdown for profiling and benchmarking.
out.timings = struct( ...
    't_file_io', static.timings.t_file_io, ...
    't_precompute_static', static.timings.t_precompute, ...
    't_static_total', static.timings.t_static_total, ...
    't_problem_build', problem.timings.t_problem_build, ...
    't_setup', out.t_setup, ...
    't_solve', out.t_solve, ...
    't_postprocess', out.t_post, ...
    't_total', toc(t_total));
out.t_total = out.timings.t_total;

end
