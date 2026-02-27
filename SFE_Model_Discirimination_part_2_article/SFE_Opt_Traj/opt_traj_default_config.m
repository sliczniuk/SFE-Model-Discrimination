function config = opt_traj_default_config()
%OPT_TRAJ_DEFAULT_CONFIG Default configuration for SFE optimal trajectory runs.

config = struct;

% Environment
config.casadi_path = 'C:\Dev\casadi-3.6.3-windows64-matlab2018b';

% Simulation / optimization horizon
config.timeStep   = 5;      % [min]
config.finalTime  = 600;    % [min]
config.sample_tol = 1e-3;   % [min]

% Input operating conditions / bounds
config.feedPressValue = 200;
config.T_min = 303;
config.T_max = 313;
config.F_min = 3.3e-5;
config.F_max = 6.7e-5;

% Random initial trajectory generation
config.n_init_knots = 15;

% Objective selection / scaling
config.use_full_discrimination_objective = false;  % false => residual(end)^2
config.objective_scale = 1e0;

% IPOPT / CasADi solver options
config.ipopt = struct;
config.ipopt.max_iter              = 100;
config.ipopt.tol                   = 1e-7;
config.ipopt.acceptable_tol        = 1e-5;
config.ipopt.acceptable_iter       = 10;
config.ipopt.hessian_approximation = 'limited-memory';
config.ipopt.sb                    = 'yes';  % suppress Ipopt banner

% Run / multi-start defaults
config.N = 60;
config.seeds = [];
config.verbose_each_run = false;
config.output_detail_multistart = 'full';
config.output_detail_best       = 'full';
config.resolve_best_seed        = false;  % false => reuse best result from screening
config.chunkSize = 6; %[];
config.collect_timings = false;
config.show_progress   = true;
config.maxWorkers      = 6;

% Result saving / export
config.save_results = true;
config.save_dir = '';  % '' => current working directory
config.save_filename_policy = 'timestamped';
config.save_prefix = 'multistart_results';
config.save_scope = 'everything';
config.save_plot = false;  % MAT only by default
config.include_workspace_snapshot = false;
config.save_mat_version = 'auto';  % 'auto' | 'v7' | 'v7.3'

end
