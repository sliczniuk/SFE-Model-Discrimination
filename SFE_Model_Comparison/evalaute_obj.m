function fun = evalaute_obj(which_k, Parameters, LabResults, bed_mask, timeStep_in_sec, ...
    epsi_mask, one_minus_epsi_mask, Nx, Nu, N_Time, N_Sample, C0fluid, C0solid, N_trial, nlp_opts)
% RUN_SINGLE_START builds and solves one Opti instance for a given start.

import casadi.*

Nk = numel(which_k);

opti = casadi.Opti();
opti.solver('ipopt', nlp_opts);

k = opti.variable(Nk);

% Integrator for forward simulation
f = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, [], epsi_mask, one_minus_epsi_mask);
F = buildIntegrator(f, [Nx, Nu], timeStep_in_sec, 'cvodes');
F_accum = F.mapaccum('F_accum', N_Time);

J = 0;
Parameters_sym          = MX(cell2mat(Parameters));
Parameters_sym(which_k) = k(1:numel(which_k));

nstages = numel(bed_mask);

for jj = N_trial
    data_org  = LabResults(6:19, jj+1)';
    data_diff = diff(data_org);

    % Operating conditions
    T0homog   = LabResults(2, jj+1);
    feedPress = LabResults(3, jj+1) * 10;
    Flow      = LabResults(4, jj+1) * 1e-5;

    Z            = Compressibility(T0homog, feedPress, Parameters);
    rho          = rhoPB_Comp(T0homog, feedPress, Z, Parameters);
    enthalpy_rho = rho .* SpecificEnthalpy(T0homog, feedPress, Z, rho, Parameters);

    feedTemp  = T0homog   * ones(1, N_Time);
    feedPress = feedPress * ones(1, N_Time);
    feedFlow  = Flow      * ones(1, N_Time);

    uu = [feedTemp', feedPress', feedFlow'];

    % Initial state
    x0 = [C0fluid';
          C0solid * bed_mask;
          enthalpy_rho * ones(nstages, 1);
          feedPress(1);
          0];

    % Build input matrix for all time steps [Nu x N_Time]
    U_all = [uu'; repmat(Parameters_sym, 1, N_Time)];

    % Symbolic simulation using mapaccum (more efficient than loop)
    X_all = F_accum(x0, U_all);
    X = [x0, X_all];

    % Compute residuals (normalized)
    Yield_estimate      = X(Nx, N_Sample);
    Yield_estimate_diff = diff(Yield_estimate);

    max_data_diff = max(data_diff);
    if max_data_diff <= 1e-9
        % Avoid division by zero for flat signals
        max_data_diff = 1;
    end
    Yield_estimate_diff_norm = Yield_estimate_diff ./ max_data_diff;
    data_diff_norm           = data_diff           ./ max_data_diff;
    residuals                = Yield_estimate_diff_norm - data_diff_norm;

    % Accumulate cost
    J = J + residuals * residuals';
end

fun = Function('fun',{k},{J});

% Non-negativity constraints
%opti.subject_to(0 <= k <= inf);

%opti.minimize(J);
%opti.set_initial(k, k_init);

%sol  = opti.solve();
%K_out = full(sol.value(k));
%obj = full(sol.value(J));
%status = opti.stats();

end
