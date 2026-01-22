function xdot = modelSFE(x, p, mask, dt, N, epsi_mask, one_minus_epsi_mask)
    % SFE Model with (F)luid, (S)olid, (T)emperature dynamics
    % Inputs:
    %   x                   - State vector [C_fluid; C_solid; enthalpy_rho; pressure; yield]
    %   p                   - Parameters cell array {T_u, P_u, F_u, parameters...}
    %   mask                - Bed mask (1 = bed, 0 = empty)
    %   dt                  - Time step [s]
    %   N                   - Number of RBF nodes
    %   epsi_mask           - Precomputed epsi .* mask (optional)
    %   one_minus_epsi_mask - Precomputed 1 - epsi .* mask (optional)

    %% Unpack inputs
    T_u        = p{1};
    P_u        = p{2};
    F_u        = p{3};
    parameters = p(4:end);

    %% Parameters
    C0solid       =     parameters{2};     % Extractor initial concentration of extract
    r             =     parameters{3};     % Extractor length (m)
    epsi          =     parameters{4};     % Void bed fraction
    dp            =     parameters{5};     % Diameter of the particle (m)
    L             =     parameters{6};     % Length of the extractor (m)
    rho_s         =     parameters{7};     %
    mi            =     parameters{9};
    %% Properties of the bed
    A             =     pi*r^2 ;       % Cross-section of the extractor (m^2)
    rp            =     dp / 2;
    lp2           =     (rp / 3)^2;
    %%

    % Use precomputed constants if provided, otherwise compute
    if nargin < 6 || isempty(epsi_mask)
        epsi_mask = epsi .* mask;
    end
    if nargin < 7 || isempty(one_minus_epsi_mask)
        one_minus_epsi_mask = 1 - epsi_mask;
    end

    nstages_index = numel(mask);

    %% States
    FLUID        = x(1:nstages_index);
    SOLID        = x(nstages_index+1:2*nstages_index);
    ENTHALPY_RHO = x(2*nstages_index+1:3*nstages_index);
    PRESSURE     = x(3*nstages_index+1);

    %% Temperature reconstruction (polynomial approximation)
    %TEMP = reconstruct_T_polynomial_approximation(log(-ENTHALPY_RHO(1)), PRESSURE) * ones(nstages_index,1);
    TEMP = T_u * ones(nstages_index,1);

    %% Fluid properties
    Z        = Compressibility(TEMP, PRESSURE, parameters);
    RHO      = rhoPB_Comp(TEMP, PRESSURE, Z, parameters);
    VELOCITY = Velocity(F_u, RHO(round(nstages_index/2)), parameters);

    %%
    MU            =     Viscosity(TEMP,RHO);
    dp            =     parameters{5};     % Diameter of the particle (m)
    RE            =     dp .* RHO .* VELOCITY ./ MU .* 1.3;

    %% Thermal properties
    CP     = SpecificHeatComp(TEMP, PRESSURE, Z, RHO, parameters);
    KRHOCP = kRHOcp_Comp(TEMP, PRESSURE, Z, RHO, CP, epsi_mask, parameters);

    %% Extraction kinetics
    Csolid_percentage_left = (1 - SOLID./C0solid) .* mask;  % Zero where no bed

    %% Boundary conditions
    Cf_0 = 0;
    Cf_B = FLUID(nstages_index);
    T_0  = T_u;
    T_B  = TEMP(nstages_index);

    % Inlet properties
    Z_0   = Compressibility(T_0, PRESSURE, parameters);
    rho_0 = rhoPB_Comp(T_0, P_u, Z_0, parameters);
    H_0   = SpecificEnthalpy(T_0, PRESSURE, Z_0, rho_0, parameters);
    enthalpy_rho_0 = rho_0 .* H_0;

    %% Spatial derivatives
    dz = L / nstages_index;

    % Diffusion (second derivative) - currently disabled
    % d2Cfdz2 = central_diff_2_order(FLUID, FLUID(1), Cf_B, dz);
    d2Tdz2  = central_diff_2_order(TEMP, T_0, T_B, dz);

    % Advection (first derivative) - second-order upwind
    dHdz         = upwind_2nd_order(VELOCITY .* ENTHALPY_RHO, VELOCITY .* enthalpy_rho_0, [], dz);
    d_cons_CF_dz = upwind_2nd_order(VELOCITY .* FLUID, VELOCITY .* Cf_0, [], dz);

    %% Temporal derivative
    dPdt = backward_diff_1_order(P_u, PRESSURE, [], dt) * 1e2;

    %% Extraction rate (Mechanistic model)
    % Two-Kinetic Model - ACTIVE
    if isequal(N,'Power_model')
        re = two_kinetic_model(SOLID, Csolid_percentage_left, RHO, (F_u * 1e5), parameters);
    elseif isequal(N,'Linear_model')
        Di     = Diffusion(RE, F_u, parameters) .* 1e-13 ;
        gamma  = Decay_Function_Coe(RE, F_u, parameters);
        Sat_coe = Saturation_Concentration(Csolid_percentage_left, gamma, Di);        % Inverse logistic is used to control saturation. Close to saturation point, the Sat_coe goes to zero.
        re = (Sat_coe ./ mi ./ lp2)  .* ( SOLID );
    else
        error('Wrong model')
    end


    %% Model equations
    xdot = [
        % Fluid phase concentration (diffusion term commented out: + Dx .* d2Cfdz2)
        (-d_cons_CF_dz + epsi_mask .* re) ./ one_minus_epsi_mask;

        % Solid phase concentration
        -mask .* re;

        % Enthalpy
        -dHdz ./ one_minus_epsi_mask + dPdt + KRHOCP .* d2Tdz2;

        % Pressure
        dPdt;

        % Output: mass flow rate [g/s]
        F_u ./ RHO(nstages_index) .* FLUID(nstages_index) * 1e3;
    ];

end