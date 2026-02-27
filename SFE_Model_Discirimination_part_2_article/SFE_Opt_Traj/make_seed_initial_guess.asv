function init = make_seed_initial_guess(seed, problem, config)
%MAKE_SEED_INITIAL_GUESS Create reproducible random initial trajectories.

if nargin < 1 || isempty(seed)
    seed = 1;
end

N_Time = problem.static.N_Time;

rng(seed);
init_knot_idx = round(linspace(1, N_Time, config.n_init_knots));

temp_knots = problem.static.T_min + (problem.static.T_max - problem.static.T_min) * rand(1, config.n_init_knots);
flow_knots = problem.static.F_min + (problem.static.F_max - problem.static.F_min) * rand(1, config.n_init_knots);

feedTemp_0 = interp1(init_knot_idx, temp_knots, 1:N_Time, 'linear');
feedFlow_0 = interp1(init_knot_idx, flow_knots, 1:N_Time, 'linear');

zFeedTemp_0 = (feedTemp_0 - problem.static.T_mid) / problem.static.T_half;
zFeedFlow_0 = (feedFlow_0 - problem.static.F_mid) / problem.static.F_half;

init = struct;
init.seed       = seed;
init.feedTemp0  = feedTemp_0;
init.feedFlow0  = feedFlow_0;
init.zFeedTemp0 = zFeedTemp_0;
init.zFeedFlow0 = zFeedFlow_0;

end
