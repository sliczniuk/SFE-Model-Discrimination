function P = Pressure_PR(T,rho,theta)

    Tc      = theta{10};   % Critical temperature [K]
    Pc      = theta{11};   % Critical pressure [bar]
    R       = theta{12};   % Universal gas constant, [m3-bar/K-mol]
    kappa   = theta{13};
    MW      = theta{14};   % Molar mass [g/mol]
    Tr      = T ./ Tc;
    
    a       = 0.4572350 .* R.^2 .* Tc.^2 ./ Pc;
    b       = 0.0777961 .* R    .* Tc    ./ Pc;
    alpha = (1 + kappa .* (1 - sqrt(Tr))).^2;

    rho_mol = rho./MW ;

    V = 1./rho_mol;

    P = (R .* T) ./ (V - b) - (alpha .* a) ./ (V.^2 + 2.*b.*V - b.^2) ;

end