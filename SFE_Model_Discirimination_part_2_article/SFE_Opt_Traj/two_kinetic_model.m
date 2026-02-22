function re = two_kinetic_model(Csolid, alpha, rho, F, parameters)
    % Langmuir model with parameters depending on control variables (T, P, F)
   
    import casadi.*

    %% Extract parameters (indices 44-54, 11 parameters)
    k_w_0      = parameters{44};  % Base half-saturation [kg/m3]
    a_w        = parameters{45};  % K_m temperature coefficient [kg/m3/K]
    b_w        = parameters{46};  % K_m pressure coefficient [kg/m3/bar]
    n          = parameters{47}; % Polynomial order of
    
    %%
    k_w = k_w_0 .* (rho ./ 800).^a_w .* (F ./ 5).^b_w .* 1e-4;

    %%
    beta = 1 ./ ( (alpha + 1).^n );
    re = (k_w .* (beta) ) .* Csolid;

end
