function re = two_kinetic_model(Csolid, alpha, rho, F, parameters)
    % Langmuir model with parameters depending on control variables (T, P, F)
   
    import casadi.*

    %% Extract parameters (indices 44-54, 11 parameters)
    k_w_0      = 1.222524;  % Base half-saturation [kg/m3]
    a_w        = 4.308414;  % K_m temperature coefficient [kg/m3/K]
    b_w        = 0.972739;  % K_m pressure coefficient [kg/m3/bar]
    n          = 3.428618; % Polynomial order of
    
    %%
    k_w = k_w_0 .* (rho ./ 800).^a_w .* (F ./ 5).^b_w .* 1e-4;

    %%
    beta = 1 ./ ( (alpha + 1).^n );
    re = (k_w .* (beta) ) .* Csolid;

end
