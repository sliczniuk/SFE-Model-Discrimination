function xdot = modelSFE_thermal_lag(x, p, mask, ~, N, epsi_mask, one_minus_epsi_mask)
    % SFE Model with (F)luid, (S)olid, (T)emperature dynamics
    %
    % Inputs:
    %   x                   - State vector [C_fluid; C_solid; enthalpy_rho; pressure; yield]
    %   p                   - Parameters cell array {T_u, P_u, F_u, parameters...}
    %   mask                - Bed mask (1 = bed, 0 = empty)
    %   dt                  - Time step [s]
    %   N                   - Number of RBF nodes / model selector string
    %   epsi_mask           - Precomputed epsi .* mask (optional)
    %   one_minus_epsi_mask - Precomputed (1 - epsi) .* mask (optional)
    %
    % Energy balance modification:
    %   The enthalpy equation now correctly accounts for the thermal inertia
    %   of the solid phase under the Local Thermal Equilibrium (LTE) assumption
    %   (T_fluid = T_solid = T at every spatial node).
    %
    %   Starting from the two-phase accumulation:
    %
    %     phi * d(rho_f*h)/dt + (1-phi)*rho_s*cp_s * dT/dt = RHS
    %
    %   Since T = f(rho_f*h, P) via the polynomial reconstruction, dT/dt is
    %   eliminated via the chain rule:
    %
    %     dT/dt = dT/d(rho_f*h) * d(rho_f*h)/dt + dT/dP * dP/dt
    %
    %   Substituting and solving for d(rho_f*h)/dt:
    %
    %     d(rho_f*h)/dt = [ RHS - (1-phi)*rho_s*cp_s * dT/dP * dP/dt ]
    %                     / [ phi + (1-phi)*rho_s*cp_s * dT/d(rho_f*h) ]
    %
    %   The denominator (acc_eff) acts as an effective accumulation coefficient.
    %   No new state variable is introduced; the correction is purely algebraic.
    %
    % References:
    %   Polynomial T reconstruction: Eq. 3.13 (dissertation)
    %   Effective thermal diffusivity: Hankalin et al. (2009)

    %% Unpack inputs
    T_u        = p{1};
    P_u        = p{2};
    F_u        = p{3};
    parameters = p(4:end);

    %% Parameters
    C0solid =  parameters{2};   % Initial solid concentration [kg/m3]
    r       =  parameters{3};   % Extractor radius [m]
    epsi    =  parameters{4};   % Void bed fraction [-]
    dp      =  parameters{5};   % Particle diameter [m]
    L       =  parameters{6};   % Extractor length [m]
    rho_s   =  parameters{7};   % Solid density [kg/m3]
    mi      =  parameters{9};   % Sphericity coefficient [-]
    cp_s    =  parameters{24} * 1e3;  % Solid specific heat [kJ/kg/K -> J/kg/K]

    %% Bed geometry
    A   = pi * r^2;             % Cross-section [m2]
    rp  = dp / 2;               % Particle radius [m]
    lp2 = (rp / 3)^2;          % Characteristic diffusion length squared [m2]

    %% Precomputed bed masks
    if nargin < 6 || isempty(epsi_mask)
        epsi_mask = epsi .* mask;
    end
    if nargin < 7 || isempty(one_minus_epsi_mask)
        one_minus_epsi_mask = 1 - epsi_mask;
    end

    nstages_index = numel(mask);

    %% Unpack states
    % ENTHALPY_RHO is stored scaled by 1/ENTHALPY_SCALE to normalise ODE
    % magnitudes (~-150,000 J/m3 -> ~-15). Physical values are recovered
    % here for all property evaluations; the returned derivative is divided
    % by ENTHALPY_SCALE at the bottom of this function.
    ENTHALPY_SCALE = 1e4;

    FLUID        = x(1:nstages_index);
    SOLID        = x(nstages_index+1:2*nstages_index);
    ENTHALPY_RHO = x(2*nstages_index+1:3*nstages_index) * ENTHALPY_SCALE;  % [J/m3]
    PRESSURE     = x(3*nstages_index+1);

    %% Temperature reconstruction (polynomial approximation, Eq. 3.13)
    TEMP = reconstruct_T_polynomial_approximation(log(-ENTHALPY_RHO), PRESSURE);

    %% Fluid properties
    Z        = Compressibility(TEMP, PRESSURE, parameters);
    RHO      = rhoPB_Comp(TEMP, PRESSURE, Z, parameters);
    VELOCITY = Velocity(F_u, RHO(round(nstages_index/2)), parameters);

    MU = Viscosity(TEMP, RHO);
    RE = dp .* RHO .* VELOCITY ./ MU .* 1.3;

    %% Thermal properties
    CP        = SpecificHeatComp(TEMP, PRESSURE, Z, RHO, parameters);
    [~, K_EFF] = kRHOcp_Comp(TEMP, PRESSURE, Z, RHO, CP, epsi_mask, parameters);

    %% Extraction kinetics
    Csolid_percentage_left = (1 - SOLID ./ C0solid) .* mask;

    %% Boundary conditions
    Cf_0 = 0;
    Cf_B = FLUID(nstages_index);
    T_0  = T_u;
    T_B  = TEMP(nstages_index);

    % Inlet enthalpy density
    Z_0            = Compressibility(T_0, PRESSURE, parameters);
    rho_0          = rhoPB_Comp(T_0, P_u, Z_0, parameters);
    H_0            = SpecificEnthalpy(T_0, PRESSURE, Z_0, rho_0, parameters);
    enthalpy_rho_0 = rho_0 .* H_0;

    %% Spatial derivatives
    dz = L / nstages_index;

    d2Tdz2       = central_diff_2_order(TEMP,              T_0,              T_B, dz);
    dHdz         = upwind_2nd_order(VELOCITY .* ENTHALPY_RHO, VELOCITY .* enthalpy_rho_0, [], dz);
    d_cons_CF_dz = upwind_2nd_order(VELOCITY .* FLUID,        VELOCITY .* Cf_0,           [], dz);

    %% Pressure time derivative
    %dPdt = backward_diff_1_order(P_u, PRESSURE, [], dt);
    tau_P = 120;   % [s] pressure relaxation time constant
    dPdt  = (P_u - PRESSURE) / tau_P;

    %% Extraction rate
    if isequal(N, 'Power_model')
        re = two_kinetic_model(SOLID, Csolid_percentage_left, RHO, (F_u * 1e5), parameters);
    elseif isequal(N, 'Linear_model')
        Di      = Diffusion(RE, F_u, parameters) .* 1e-13;
        gamma   = Decay_Function_Coe(RE, F_u, parameters);
        Sat_coe = Saturation_Concentration(Csolid_percentage_left, gamma, Di);
        re      = (Sat_coe ./ mi ./ lp2) .* SOLID;
    else
        error('modelSFE: unknown model type ''%s''.', N);
    end
    
    %% Compute dT/d(rho*h) and dT/dP via CasADi AD
    % Build a scalar CasADi Function from fresh MX.sym symbols, then map
    % over all spatial stages.  This guarantees consistency with
    % reconstruct_T_polynomial_approximation and correct chain rule.
    import casadi.*
    hr_ad = MX.sym('hr_ad');            % scalar rho*h  (pure symbol)
    p_ad  = MX.sym('p_ad');             % scalar P      (pure symbol)
    T_ad  = reconstruct_T_polynomial_approximation(log(-hr_ad), p_ad);
    dT_fn = casadi.Function('dT_fn', {hr_ad, p_ad}, ...
                {jacobian(T_ad, hr_ad), jacobian(T_ad, p_ad)});
    dT_fn_map = dT_fn.map(nstages_index);
    [dTdHR_t, dTdP_t] = dT_fn_map( ...
        ENTHALPY_RHO', repmat(PRESSURE, 1, nstages_index));
    dTdHR = dTdHR_t';
    dTdP  = dTdP_t';

    % Solid volumetric heat capacity weighted by solid volume fraction  [J/m3/K]
    %rhocp_solid = rho_s .* cp_s .* (1 - epsi_mask);
    rhocp_solid = rho_s .* cp_s .* (1 - epsi) .* mask;

    % Effective accumulation coefficient:
    %   phi + (1-phi)*rho_s*cp_s * dT/d(rho_f*h)
    % In the bed:   epsi_mask = phi,  so this > phi  (solid slows accumulation)
    % Outside bed:  epsi_mask = 0,    rhocp_solid = 0, acc_eff = 0 (handled below)
    %acc_eff = epsi_mask + rhocp_solid .* dTdHR;
    phi_fluid = epsi_mask + (1 - mask);   % epsi in bed, 1 outside
    acc_eff   = phi_fluid + rhocp_solid .* dTdHR;

    % Guard against zero or negative acc_eff outside the bed.
    % Outside the bed there is no solid, so the fluid-only value (epsi_mask)
    % should be used. For grid points where mask=0, epsi_mask=0 as well;
    % protect against division by zero by flooring at a small positive value.
    %acc_eff = max(acc_eff, 1e-6);

    % Solid pressure-coupling term:
    %   (1-phi)*rho_s*cp_s * dT/dP * dPdt   [J/m3/s]
    % This accounts for energy absorbed/released by the solid during
    % pressure transients, mediated through the temperature reconstruction.
    solid_pressure_term = rhocp_solid .* dTdP .* dPdt;

    %% RHS of enthalpy equation (before accumulation correction)
    enthalpy_RHS = -dHdz ./ one_minus_epsi_mask ...
                   + phi_fluid .* dPdt ...
                   + K_EFF .* d2Tdz2;

    %% Assemble state derivatives
    xdot = [
        % --- Fluid phase concentration ---
        % Convection + kinetic source; diffusion term omitted (D_e^M = 0)
        (-d_cons_CF_dz + epsi_mask .* re) ./ one_minus_epsi_mask;

        % --- Solid phase concentration ---
        -mask .* re;

        % --- Enthalpy density (scaled) ---
        % Full two-phase energy balance under LTE, solid inertia via chain rule.
        % Dividing RHS by acc_eff introduces the solid thermal mass into the
        % effective accumulation; subtracting solid_pressure_term accounts for
        % energy exchange with the solid during pressure changes.
        (enthalpy_RHS - solid_pressure_term) ./ acc_eff / ENTHALPY_SCALE;

        % --- Pressure ---
        dPdt;

        % --- Yield [g/s] ---
        F_u ./ RHO(nstages_index) .* FLUID(nstages_index) * 1e3;
    ];

end