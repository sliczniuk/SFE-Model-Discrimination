%% Multi-start parallel trajectory optimization (seed-level parfor)
% Simplified implementation:
% - parallelize directly over seeds (one seed per parfor iteration)
% - optimize each seed once during screening
% - no plotting; save optimization summaries for downstream analysis

% Initialize client session paths first (startup may clear variables)
startup;

config = opt_traj_default_config();
addpath(config.casadi_path);

%% User-tunable run configuration
%{
config.N = 8;
config.seeds = [];                 % [] => 1:N
config.verbose_each_run = false;
config.collect_timings = false;   % enable timing fields in summaries
config.show_progress   = true;    % DataQueue progress + ETA
config.maxWorkers      = [];      % [] => default local pool size
config.save_results    = true;
config.save_scope      = 'everything';
config.save_prefix     = 'multistart_results';
config.save_dir        = '';      % '' => pwd
config.save_mat_version = 'auto';
%}

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

% Headless-safe worker initialization:
% Prevent Java/AWT graphics initialization on workers (can trigger
% HeadlessGraphicsEnvironment/Win32GraphicsEnvironment ClassCastException).
try
    fInit = parfevalOnAll(@local_worker_init_headless, 0, config.casadi_path);
    wait(fInit);
catch initErr
    warning(['Worker headless initialization failed. Continuing anyway. ', ...
        'Error: %s'], initErr.message);
end

numWorkers = pool.NumWorkers;

fprintf('Running %d seeds with seed-level parfor (workers=%d)\n', ...
    config.N, numWorkers);

%% Progress tracking (client-side only)
dq = [];
if config.show_progress
    dq = parallel.pool.DataQueue;
    local_progress_monitor('reset', config.N);
    afterEach(dq, @(msg) local_progress_monitor('update', msg));
end

%% Parallel multi-start (seed-level)
batchOptionsBase = struct;
batchOptionsBase.output_detail    = 'summary';
batchOptionsBase.verbose_each_run = config.verbose_each_run;
batchOptionsBase.collect_timings  = config.collect_timings;
batchOptionsBase.dataQueue        = dq;
batchOptionsBase.block_id         = NaN;

t_multistart = tic;
resultsCell = cell(1, config.N);
parfor i = 1:config.N
    batchOptions = batchOptionsBase;
    batchOptions.block_id = i;
    seedResult = solve_opt_traj_batch(seeds(i), config, batchOptions);
    resultsCell{i} = seedResult(1);
end
t_multistart_elapsed = toc(t_multistart);

if config.show_progress
    local_progress_monitor('finalize', []);
end

results = [resultsCell{:}];
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
fprintf('Best seed: %d | success=%d | j=% .6e\n\n', ...
    best_summary.seed, best_summary.success, best_summary.j);

%% Autosave (pressure-based MAT filename)
if config.save_results
    t_save = tic;
    try
        save_path = local_make_save_filename(config);
        save_runs = struct( ...
            'seed', num2cell([results.seed]), ...
            'j_initial', num2cell([results.j_initial]), ...
            'j_final', num2cell([results.j_final]), ...
            'feedTemp0', {results.feedTemp0}, ...
            'feedFlow0', {results.feedFlow0}, ...
            'feedTemp', {results.feedTemp}, ...
            'feedFlow', {results.feedFlow});

        mat_version = 'auto';
        if isfield(config, 'save_mat_version') && ~isempty(config.save_mat_version)
            mat_version = lower(string(config.save_mat_version));
        end
        switch char(mat_version)
            case 'v7'
                save(save_path, 'save_runs');
            case 'v7.3'
                save(save_path, 'save_runs', '-v7.3');
            case 'auto'
                try
                    save(save_path, 'save_runs');
                catch err
                    if contains(lower(err.message), '2gb') || contains(lower(err.message), 'v7.3')
                        save(save_path, 'save_runs', '-v7.3');
                    else
                        rethrow(err);
                    end
                end
            otherwise
                error('Unsupported save_mat_version: %s', char(mat_version));
        end

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

function save_path = local_make_save_filename(config)
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

prefix = 'multistart_results';
if isfield(config, 'save_prefix') && ~isempty(config.save_prefix)
    prefix = char(config.save_prefix);
end

if ~isfield(config, 'feedPressValue') || isempty(config.feedPressValue)
    error('config.feedPressValue is required for pressure-based filenames.');
end
pressure_tag = local_pressure_tag(config.feedPressValue);
base_name = sprintf('%s_P%sbar', prefix, pressure_tag);

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

function tag = local_pressure_tag(p)
if ~isscalar(p) || ~isfinite(p)
    error('feedPressValue must be a finite scalar.');
end
tag = num2str(p, '%.12g');
tag = strrep(tag, '.', 'p');
tag = strrep(tag, '-', 'm');
end

function local_worker_init_headless(casadi_path)
if nargin >= 1 && ~isempty(casadi_path)
    addpath(casadi_path);
end
try
    java.lang.System.setProperty('java.awt.headless', 'true');
catch
    % If Java is unavailable/not needed, silently continue.
end
end
