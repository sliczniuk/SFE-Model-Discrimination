function results = solve_opt_traj_batch(seeds_block, config, batch_options)
%SOLVE_OPT_TRAJ_BATCH Solve a block of seeds while reusing static setup/problem build.

if nargin < 3
    batch_options = struct;
end
if ~isfield(batch_options, 'output_detail') || isempty(batch_options.output_detail)
    batch_options.output_detail = 'summary';
end
if ~isfield(batch_options, 'verbose_each_run') || isempty(batch_options.verbose_each_run)
    batch_options.verbose_each_run = false;
end
if ~isfield(batch_options, 'collect_timings') || isempty(batch_options.collect_timings)
    batch_options.collect_timings = false;
end
if ~isfield(batch_options, 'dataQueue')
    batch_options.dataQueue = [];
end
if ~isfield(batch_options, 'block_id') || isempty(batch_options.block_id)
    batch_options.block_id = NaN;
end

nSeeds = numel(seeds_block);
results = repmat(local_summary_template(), 1, nSeeds);
if nSeeds == 0
    return;
end

t_setup = tic;
try
    static = load_opt_traj_static_data(config);
    problem = build_opt_traj_problem(static, config, batch_options.verbose_each_run);
    setup_error = '';
catch err
    static = [];
    problem = [];
    setup_error = err.message;
end
setup_elapsed = toc(t_setup);

if isempty(problem)
    for i = 1:nSeeds
        results(i) = local_summary_template();
        results(i).seed = seeds_block(i);
        results(i).success = false;
        results(i).j = -Inf;
        results(i).error_message = sprintf('Batch setup failed: %s', setup_error);
        if batch_options.collect_timings
            results(i).t_setup = setup_elapsed / nSeeds;
            results(i).t_total = results(i).t_setup;
        end
        local_send_progress(batch_options.dataQueue, results(i), batch_options.block_id, NaN);
    end
    return;
end

if isfield(static, 'timings')
    t_file_io_total = static.timings.t_file_io;
else
    t_file_io_total = NaN;
end
if isfield(problem, 'timings') && isfield(problem.timings, 't_problem_build')
    t_problem_build_total = problem.timings.t_problem_build;
else
    t_problem_build_total = NaN;
end

t_setup_per_seed = setup_elapsed / nSeeds;
t_file_io_per_seed = t_file_io_total / nSeeds;
t_problem_build_per_seed = t_problem_build_total / nSeeds;

for i = 1:nSeeds
    seed = seeds_block(i);
    init = make_seed_initial_guess(seed, problem, config);

    t_iter = tic;
    seed_options = struct;
    seed_options.output_detail    = batch_options.output_detail;
    seed_options.include_stats    = false;
    seed_options.collect_timings  = batch_options.collect_timings;
    seed_options.t_setup          = t_setup_per_seed;
    seed_options.t_file_io        = t_file_io_per_seed;
    seed_options.t_problem_build  = t_problem_build_per_seed;

    res = solve_opt_traj_problem_seed(problem, init, seed_options);
    results(i) = local_cast_summary(res);

    local_send_progress(batch_options.dataQueue, results(i), batch_options.block_id, toc(t_iter));
end

end

function s = local_summary_template()
s = struct( ...
    'seed', NaN, ...
    'success', false, ...
    'j', -Inf, ...
    'j_initial', NaN, ...
    'j_final', NaN, ...
    'status', '', ...
    'iter_count', NaN, ...
    'error_message', '', ...
    'feedTemp', [], ...
    'feedFlow', [], ...
    'feedTemp0', [], ...
    'feedFlow0', [], ...
    'Time', [], ...
    'yieldTime', [], ...
    'yieldPower', [], ...
    'yieldLinear', [], ...
    'yieldPowerCI', [], ...
    'yieldLinearCI', [], ...
    't_setup', NaN, ...
    't_file_io', NaN, ...
    't_problem_build', NaN, ...
    't_solve', NaN, ...
    't_post', NaN, ...
    't_total', NaN);
end

function out = local_cast_summary(in)
out = local_summary_template();
f = fieldnames(out);
for k = 1:numel(f)
    if isfield(in, f{k})
        out.(f{k}) = in.(f{k});
    end
end
end

function local_send_progress(dq, result, block_id, iter_time)
if isempty(dq)
    return;
end
msg = struct( ...
    'seed', result.seed, ...
    'success', result.success, ...
    'iter_time', iter_time, ...
    'block_id', block_id);
send(dq, msg);
end
