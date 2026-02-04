function xdot = modelSFE_Corr(x, p, mask, dt)
    % (t, x, u, parameters)
    % Model with (F)luid, (S)olid, (T)emperature
    % Di is a function of temperature (T), the function works with numbers,
    % vectors of numbers and vectors of symbolic variables
    % Rho (Peng-Robinson) are constant numbers

    %% Load Paramters
    T_u           =     p{1};
    P_u           =     p{2};
    F_u           =     p{3} * 1e-5;

    parameters    =     p(4:end);

    %nstages       =     parameters{1};
    C0solid       =     parameters{2};     % Extractor initial concentration of extract
    r             =     parameters{3};     % Extractor length (m)
    epsi          =     parameters{4};     % Void bed fraction
    dp            =     parameters{5};     % Diameter of the particle (m)
    L             =     parameters{6};     % Length of the extractor (m)
    rho_s         =     parameters{7};     %
    mi            =     parameters{9};

    %km            =     parameters{8} ;
    %Di            =     parameters{44}* 1e-13;      
    %Dx            =     parameters{45}* 1e-12;      
    Dx            =     1e-15;      
    %gamma         =     parameters{46};

    nstages_index =     numel(mask);

    %% Properties of the bed
    A             =     pi*r^2 ;       % Cross-section of the extractor (m^2)
    rp            =     dp / 2;
    lp2           =     (rp / 3)^2;

    %% States
    FLUID         =     x(0*nstages_index+1:1*nstages_index);
    SOLID         =     x(1*nstages_index+1:2*nstages_index);
    ENTHALPY_RHO  =     x(2*nstages_index+1:3*nstages_index);
    PRESSURE      =     x(3*nstages_index+1);

    TEMP          =     Reconstruct_T_from_enthalpy(ENTHALPY_RHO, PRESSURE, parameters);
      
    %Properties of the fluid in the extractor
    Z             =     Compressibility(TEMP, PRESSURE,    parameters);

    RHO           =     rhoPB_Comp(     TEMP, PRESSURE, Z, parameters);   
    MU            =     Viscosity(TEMP,RHO);
    VELOCITY      =     Velocity(F_u, mean([RHO(round(linspace(1,nstages_index,5)))]), parameters) .* ones(nstages_index,1);
    %VELOCITY      =     Velocity(F_u, RHO(1), parameters) .* ones(nstages_index,1);

    RE            =     dp .* RHO .* VELOCITY ./ MU .* 1.3;
    
    %% Thermal Properties
    CP            =     SpecificHeatComp(TEMP, PRESSURE, Z, RHO,                 parameters);            % [kJ/kg/K]
    %CPRHOCP       =     cpRHOcp_Comp(    TEMP, PRESSURE, Z, RHO, CP, epsi.*mask, parameters);
    KRHOCP        =     kRHOcp_Comp(     TEMP, PRESSURE, Z, RHO, CP, epsi.*mask, parameters);

    %% Extraction kientic
    Di            = Diffusion(RE, F_u, parameters) .* 1e-13 ;
    gamma         = Decay_Function_Coe(RE, F_u, parameters);
    %Dx            = axial_diffusion(RHO, parameters) .* 1e-6;

    %% Saturation
    Csolid_percentage_left = 1 - (SOLID./C0solid);
    Csolid_percentage_left(find(~mask)) = 0;                                                % inserte zeros instead of NAN in pleces where there is no bed
    Sat_coe       =     Saturation_Concentration(Csolid_percentage_left, gamma, Di);        % Inverse logistic is used to control saturation. Close to saturation point, the Sat_coe goes to zero.

    %% BC
    Cf_0          =     0;
    Cf_B          =     FLUID(nstages_index);
                                                                                            % to avoid different small mismatch of T between the inlet and inside of the extractor
    T_0           =      T_u;
    
    T_B           =     TEMP(nstages_index);

    Z_0           =     Compressibility(T_0, PRESSURE,     parameters);
    
    rho_0         =     rhoPB_Comp(     T_0, P_u, Z_0, parameters);
    
    u_0           =      VELOCITY(1);
    
    H_0           =     SpecificEnthalpy(T_0, PRESSURE, Z_0, rho_0, parameters );   

    enthalpy_rho_0 = rho_0 .* H_0 ;                                                         % If the sensitivity of F is consider, then set the input h*rho as equal to the h*rho inside of the extractor
                                                                                            % to avoid different small mismatch betweenat the inlet and inside of the extractor
    %% Derivatives
    dz            = L/nstages_index;
    
    d2Cfdz2       = central_diff_2_order(FLUID, FLUID(1), Cf_B, dz);
    
    d2Tdz2        = central_diff_2_order(TEMP, T_0, T_B, dz);
        
    dHdz          = backward_diff_1_order(VELOCITY .* ENTHALPY_RHO, u_0 .* enthalpy_rho_0, [], dz);

    d_cons_CF_dz  = backward_diff_1_order(VELOCITY .* FLUID, u_0 .* Cf_0, [], dz);

    dPdt          = backward_diff_1_order(P_u, PRESSURE, [], dt)*1e2;
   
    %re            = (Di ./ mi ./ lp2)  .* ( SOLID  );
    re            = (Sat_coe ./ mi ./ lp2)  .* ( SOLID );
    %re            = RBF_Function_1D( Csolid_percentage_left, 2, parameters) * 1e-3;
    %re             = RBF_Function_2D_Single_Layer( Csolid_percentage_left, RE_vector, N, parameters) * 1e-3;
    %re            = RBF_Function_2D_Double_Layer_Double_Hidden( Csolid_percentage_left, RE(ind), N, parameters) * 1e-3;
    %re            = RBF_Function_3D_Single_Layer( Csolid_percentage_left, RE_vector, RHO./800, N, parameters) * 1e-3;
    %re            = (abs(re) + re)./2;
    
    %% model
    xdot = [
    
    %--------------------------------------------------------------------
    % Concentration of extract in fluid phase | 0
   - 1            ./  ( 1 - epsi .* mask ) .* d_cons_CF_dz   + ...
    Dx            ./  ( 1 - epsi .* mask ) .* d2Cfdz2        + ...
    (epsi.*mask)  ./  ( 1 - epsi .* mask ) .* re;
    %zeros(nstages_index,1);

    %--------------------------------------------------------------------
    % Concentration of extract in solid phase | 1
    - mask                                 .* re;
    %zeros(nstages_index,1);
    
    %--------------------------------------------------------------------
    % enthalpy | 2
    - 1      ./ ( 1 - epsi .* mask )  .* dHdz + dPdt + KRHOCP.* d2Tdz2;

    %--------------------------------------------------------------------
    % Pressure | 3
     dPdt;
    
    %--------------------------------------------------------------------
    % output equation
    %VELOCITY(nstages_index) * A * FLUID(nstages_index) * 1e3 ;   %kg/s - > g/s
    F_u ./ RHO(nstages_index) .* FLUID(nstages_index) * 1e3;

    ];

end