%% Multi-start parallel trajectory optimization (block-based parfor)
% Optimized implementation:
% - parallelize over seed blocks (not individual seeds)
% - reuse static data + CasADi problem once per block
% - configurable screening detail (summary/full)
% - optional best-seed re-solve (can be skipped to avoid extra solve cost)

% Initialize client session paths first (startup may clear variables)
startup;

config = opt_traj_default_config();
addpath(config.casadi_path);

%% User-tunable run configuration
%{
config.N = 8;
config.seeds = [];                 % [] => 1:N
config.verbose_each_run = false;
config.output_detail_multistart = 'summary';
config.output_detail_best       = 'full';
config.resolve_best_seed        = false;  % false => no second optimization for best seed
config.chunkSize = [];            % [] => auto
config.collect_timings = false;   % enable timing fields in summaries
config.show_progress   = true;    % DataQueue progress + ETA
config.maxWorkers      = [];      % [] => default local pool size
config.save_results    = true;
config.save_scope      = 'everything';
config.save_filename_policy = 'timestamped';
config.save_prefix     = 'multistart_results';
config.save_dir        = '';      % '' => pwd
config.save_mat_version = 'auto';
%}

if ~isfield(config, 'resolve_best_seed') || isempty(config.resolve_best_seed)
    config.resolve_best_seed = false;
end
if ~config.resolve_best_seed && ~strcmpi(config.output_detail_multistart, 'full')
    fprintf(['resolve_best_seed=false requires full screening outputs. ', ...
             'Switching config.output_detail_multistart: ''%s'' -> ''full''.\n'], ...
            char(config.output_detail_multistart));
    config.output_detail_multistart = 'full';
end

%% Parallel pool setup

if isempty(config.seeds)
    seeds = 1:config.N;
else
    seeds = config.seeds(:).';
    config.N = numel(seeds);
end

pool = gcp('nocreate');
if isempty(pool)
    if isempty(config.maxWorkers)
        pool = parpool('local');
    else
        pool = parpool('local', config.maxWorkers);
    end
elseif ~isempty(config.maxWorkers) && pool.NumWorkers ~= config.maxWorkers
    warning('Existing pool has %d workers (config.maxWorkers=%d). Reusing existing pool.', ...
        pool.NumWorkers, config.maxWorkers);
end

numWorkers = pool.NumWorkers;
if isempty(config.chunkSize)
    chunkSize = max(2, ceil(config.N / (2 * numWorkers)));
else
    chunkSize = max(1, config.chunkSize);
end
seedBlocks = local_make_blocks(seeds, chunkSize);
numBlocks  = numel(seedBlocks);

fprintf('Running %d seeds across %d blocks (chunkSize=%d, workers=%d)\n', ...
    config.N, numBlocks, chunkSize, numWorkers);

%% Progress tracking (client-side only)
dq = [];
if config.show_progress
    dq = parallel.pool.DataQueue;
    local_progress_monitor('reset', config.N);
    afterEach(dq, @(msg) local_progress_monitor('update', msg));
end

%% Parallel multi-start (summary-only)
batchOptionsBase = struct;
batchOptionsBase.output_detail    = config.output_detail_multistart;
batchOptionsBase.verbose_each_run = config.verbose_each_run;
batchOptionsBase.collect_timings  = config.collect_timings;
batchOptionsBase.dataQueue        = dq;
batchOptionsBase.block_id         = NaN;

t_multistart = tic;
blockResults = cell(1, numBlocks);
parfor b = 1:numBlocks
    batchOptions = batchOptionsBase;
    batchOptions.block_id = b;
    blockResults{b} = solve_opt_traj_batch(seedBlocks{b}, config, batchOptions);
end
t_multistart_elapsed = toc(t_multistart);

if config.show_progress
    local_progress_monitor('finalize', []);
end

results = [blockResults{:}];
if isempty(results)
    error('No results returned from solve_opt_traj_batch.');
end

% Ensure deterministic ordering for summary/reporting
[~, order] = sort([results.seed]);
results = results(order);

%% Select best seed from summaries
success_mask = [results.success];
if any(success_mask)
    candidates = results(success_mask);
    [~, best_idx_local] = max([candidates.j]);
    best_summary = candidates(best_idx_local);
else
    finite_mask = isfinite([results.j]);
    if ~any(finite_mask)
        error('All multi-start runs failed and no finite objective values are available.');
    end
    warning('No successful solves. Selecting best finite debug-evaluated objective among failed runs.');
    candidates = results(finite_mask);
    [~, best_idx_local] = max([candidates.j]);
    best_summary = candidates(best_idx_local);
end

%% Build best full result for plotting
if config.resolve_best_seed
    if ~strcmpi(config.output_detail_best, 'full')
        error('resolve_best_seed=true requires config.output_detail_best = ''full''.');
    end
    best = solve_opt_traj_once(best_summary.seed, false);
else
    best = local_get_result_by_seed(results, best_summary.seed);
    if isempty(best.feedTemp) || isempty(best.feedFlow) || isempty(best.yieldPower)
        error(['Best screening result does not contain full trajectories/yields. ', ...
               'Use config.output_detail_multistart = ''full'' or set resolve_best_seed=true.']);
    end
    fprintf('Using best screening result directly (no second optimization solve).\n');
end

%% Summary reporting
fprintf('\n=== Multi-start summary (screening phase) ===\n');
for i = 1:numel(results)
    fprintf('seed=%3d | success=%d | j=% .6e | iter=%g | status=%s', ...
        results(i).seed, results(i).success, results(i).j, ...
        results(i).iter_count, results(i).status);
    if config.collect_timings
        fprintf(' | t_setup=%.2fs | t_solve=%.2fs | t_post=%.2fs', ...
            results(i).t_setup, results(i).t_solve, results(i).t_post);
    end
    fprintf('\n');
end
fprintf('Screening wall time: %.2f min\n', t_multistart_elapsed / 60);
fprintf('Best seed: %d | success=%d | j=% .6e\n\n', best.seed, best.success, best.j);

%% Plot best full result
plot_error_message = '';
try
    figure;
    subplot(3,1,1)
    hold on
    stairs(best.Time, [best.feedTemp0, best.feedTemp0(end)] - 273, 'LineWidth', 2)
    stairs(best.Time, [best.feedTemp,  best.feedTemp(end)]  - 273, 'LineWidth', 2)
    hold off
    xlabel('Time min')
    ylabel('T C')
    legend('Initial guess', 'Optimized', 'Location', 'best')
    title(sprintf('Best multi-start result (seed %d)', best.seed))

    subplot(3,1,2)
    hold on
    stairs(best.Time, [best.feedFlow0, best.feedFlow0(end)], 'LineWidth', 2)
    stairs(best.Time, [best.feedFlow,  best.feedFlow(end)],  'LineWidth', 2)
    hold off
    xlabel('Time min')
    ylabel('F kg/s')
    legend('Initial guess', 'Optimized', 'Location', 'best')

    subplot(3,1,3)
    hold on
    x = best.yieldTime(:)';
    p_lo = best.yieldPower  - best.yieldPowerCI;
    p_hi = best.yieldPower  + best.yieldPowerCI;
    l_lo = best.yieldLinear - best.yieldLinearCI;
    l_hi = best.yieldLinear + best.yieldLinearCI;

    fill([x, fliplr(x)], [p_lo, fliplr(p_hi)], [0.20, 0.45, 0.85], ...
        'FaceAlpha', 0.15, 'EdgeColor', 'none');
    fill([x, fliplr(x)], [l_lo, fliplr(l_hi)], [0.90, 0.35, 0.20], ...
        'FaceAlpha', 0.15, 'EdgeColor', 'none');
    plot(x, best.yieldPower,  '-', 'Color', [0.10, 0.30, 0.75], 'LineWidth', 2);
    plot(x, best.yieldLinear, '-', 'Color', [0.75, 0.20, 0.10], 'LineWidth', 2);
    hold off
    xlabel('Time min')
    ylabel('Yield')
    legend('Power 95% CI', 'Linear 95% CI', 'Power model', 'Linear model', 'Location', 'best')
    title('Optimized yield curves with predictive 95% CI')
catch plotErr
    plot_error_message = plotErr.message;
    warning('Plot generation failed; proceeding to save results. Error: %s', plot_error_message);
end

%% Build aggregate summary + autosave (timestamped MAT)
screening_summary = local_build_screening_summary(results, best_summary, t_multistart_elapsed, config);

save_timestamp = datestr(now, 'yyyymmdd_HHMMSS');
save_meta = struct;
save_meta.timestamp = save_timestamp;
save_meta.matlab_version = version;
save_meta.hostname = local_get_hostname();
save_meta.pwd = pwd;
save_meta.script_name = 'run_multistart.m';
save_meta.save_scope = config.save_scope;
save_meta.filename = '';
save_meta.success_count = sum([results.success]);
save_meta.N = config.N;
save_meta.numWorkers = numWorkers;
save_meta.chunkSize = chunkSize;
save_meta.numBlocks = numBlocks;
save_meta.plot_error_message = plot_error_message;

if config.save_results
    t_save = tic;
    try
        save_path = local_make_save_filename(config, save_timestamp);
        save_meta.filename = save_path;

        local_save_multistart_matfile(save_path, config.save_mat_version, ...
            results, best_summary, best, config, seeds, seedBlocks, ...
            numWorkers, chunkSize, numBlocks, t_multistart_elapsed, ...
            save_meta, screening_summary);

        d = dir(save_path);
        if isempty(d)
            fprintf('Saved results: %s\n', save_path);
        else
            fprintf('Saved results: %s (%.2f MB) in %.2f s\n', ...
                save_path, d.bytes / 1024 / 1024, toc(t_save));
        end
    catch saveErr
        warning('Failed to save results MAT file: %s', saveErr.message);
    end
end

%% Local helpers (script-local functions; MATLAB R2016b+)
function blocks = local_make_blocks(seed_list, chunk_size)
    if isempty(seed_list)
        blocks = {};
        return;
    end
    n = numel(seed_list);
    nb = ceil(n / chunk_size);
    blocks = cell(1, nb);
    idx = 1;
    for k = 1:nb
        j = min(idx + chunk_size - 1, n);
        blocks{k} = seed_list(idx:j);
        idx = j + 1;
    end
end

function local_progress_monitor(mode, payload)
    persistent state
    switch lower(mode)
        case 'reset'
            N_total = payload;
            state = struct( ...
                'N', N_total, ...
                'completed', 0, ...
                'success', 0, ...
                'tStart', tic, ...
                'tLastPrint', tic);
        case 'update'
            if isempty(state)
                return;
            end
            msg = payload;
            state.completed = state.completed + 1;
            if isfield(msg, 'success') && msg.success
                state.success = state.success + 1;
            end
            elapsed = toc(state.tStart);
            rate = state.completed / max(elapsed, eps);
            eta_sec = (state.N - state.completed) / max(rate, eps);
            if toc(state.tLastPrint) >= 1 || state.completed == state.N
                last_iter = NaN;
                if isfield(msg, 'iter_time')
                    last_iter = msg.iter_time;
                end
                seed = NaN;
                if isfield(msg, 'seed')
                    seed = msg.seed;
                end
                fprintf(['Progress: %d/%d (%.1f%%) | success=%d | elapsed=%.1f min | ', ...
                         'ETA=%.1f min | last=%.1f s | seed=%g\n'], ...
                    state.completed, state.N, 100 * state.completed / state.N, ...
                    state.success, elapsed / 60, eta_sec / 60, last_iter, seed);
                state.tLastPrint = tic;
            end
        case 'finalize'
            if isempty(state)
                return;
            end
            elapsed = toc(state.tStart);
            fprintf('Progress complete: %d/%d | success=%d | elapsed=%.2f min\n', ...
                state.completed, state.N, state.success, elapsed / 60);
        otherwise
            error('Unknown progress monitor mode: %s', mode);
    end
end

function r = local_get_result_by_seed(results, seed)
idx = find([results.seed] == seed, 1, 'first');
if isempty(idx)
    error('Could not find seed %g in screening results.', seed);
end
r = results(idx);
end

function summary = local_build_screening_summary(results, best_summary, t_multistart_elapsed, config)
success_mask = [results.success];
iter_vals = [results.iter_count];
j_vals = [results.j];

summary = struct;
summary.N = numel(results);
summary.n_success = sum(success_mask);
summary.n_failed = summary.N - summary.n_success;
summary.success_rate = summary.n_success / max(summary.N, 1);
summary.best_seed = best_summary.seed;
summary.best_j_screening = best_summary.j;
summary.screening_wall_time_sec = t_multistart_elapsed;
summary.mean_iter_count = local_mean_no_nan(iter_vals);
summary.median_iter_count = local_median_no_nan(iter_vals);
summary.mean_j = local_mean_no_nan(j_vals);
summary.config_collect_timings = logical(config.collect_timings);

if config.collect_timings
    summary.mean_t_setup = local_mean_no_nan([results.t_setup]);
    summary.mean_t_file_io = local_mean_no_nan([results.t_file_io]);
    summary.mean_t_problem_build = local_mean_no_nan([results.t_problem_build]);
    summary.mean_t_solve = local_mean_no_nan([results.t_solve]);
    summary.mean_t_post = local_mean_no_nan([results.t_post]);
    summary.mean_t_total = local_mean_no_nan([results.t_total]);
end
end

function save_path = local_make_save_filename(config, ts_override)
if nargin < 2
    ts_override = '';
end
if ~isfield(config, 'save_dir') || isempty(config.save_dir)
    out_dir = pwd;
else
    out_dir = config.save_dir;
    if exist(out_dir, 'dir') ~= 7
        [ok, msg] = mkdir(out_dir);
        if ~ok
            error('Failed to create save directory "%s": %s', out_dir, msg);
        end
    end
end

policy = 'timestamped';
if isfield(config, 'save_filename_policy') && ~isempty(config.save_filename_policy)
    policy = lower(string(config.save_filename_policy));
end

prefix = 'multistart_results';
if isfield(config, 'save_prefix') && ~isempty(config.save_prefix)
    prefix = char(config.save_prefix);
end

switch char(policy)
    case 'timestamped'
        if isempty(ts_override)
            ts = datestr(now, 'yyyymmdd_HHMMSS');
        else
            ts = ts_override;
        end
        base_name = sprintf('%s_%s', prefix, ts);
    case 'fixed latest'
        base_name = prefix;
    case 'fixed'
        base_name = prefix;
    otherwise
        error('Unsupported save_filename_policy: %s', char(policy));
end

save_path = fullfile(out_dir, [base_name, '.mat']);
if exist(save_path, 'file') ~= 2
    return;
end

% Collision handling for timestamped filenames (same-second reruns)
idx = 1;
while exist(save_path, 'file') == 2
    save_path = fullfile(out_dir, sprintf('%s_%03d.mat', base_name, idx));
    idx = idx + 1;
end
end

function local_save_multistart_matfile(save_path, mat_version, ...
    results, best_summary, best, config, seeds, seedBlocks, ...
    numWorkers, chunkSize, numBlocks, t_multistart_elapsed, ...
    save_meta, screening_summary)

if nargin < 2 || isempty(mat_version)
    mat_version = 'auto';
end
mat_version = lower(string(mat_version));

switch char(mat_version)
    case 'v7'
        save(save_path, 'results', 'best_summary', 'best', 'config', 'seeds', ...
            'seedBlocks', 'numWorkers', 'chunkSize', 'numBlocks', ...
            't_multistart_elapsed', 'save_meta', 'screening_summary');
    case 'v7.3'
        save(save_path, 'results', 'best_summary', 'best', 'config', 'seeds', ...
            'seedBlocks', 'numWorkers', 'chunkSize', 'numBlocks', ...
            't_multistart_elapsed', 'save_meta', 'screening_summary', '-v7.3');
    case 'auto'
        try
            save(save_path, 'results', 'best_summary', 'best', 'config', 'seeds', ...
                'seedBlocks', 'numWorkers', 'chunkSize', 'numBlocks', ...
                't_multistart_elapsed', 'save_meta', 'screening_summary');
        catch err
            if contains(lower(err.message), '2gb') || contains(lower(err.message), 'v7.3')
                save(save_path, 'results', 'best_summary', 'best', 'config', 'seeds', ...
                    'seedBlocks', 'numWorkers', 'chunkSize', 'numBlocks', ...
                    't_multistart_elapsed', 'save_meta', 'screening_summary', '-v7.3');
            else
                rethrow(err);
            end
        end
    otherwise
        error('Unsupported save_mat_version: %s', char(mat_version));
end
end

function host = local_get_hostname()
host = getenv('COMPUTERNAME');
if isempty(host)
    host = getenv('HOSTNAME');
end
if isempty(host)
    host = 'unknown';
end
end

function m = local_mean_no_nan(x)
x = x(~isnan(x));
if isempty(x)
    m = NaN;
else
    m = mean(x);
end
end

function m = local_median_no_nan(x)
x = x(~isnan(x));
if isempty(x)
    m = NaN;
else
    m = median(x);
end
end
