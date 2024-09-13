function krhocp = kRHOcp_Comp(T, P, Z, RHO, CP, epsi, theta)

    % http://www.ffrc.fi/FlameDays_2009/3B/HankalinPaper.pdf
    
    %epsi     = theta{4};                                   % Void bed fraction    
    rhoSolid = theta{7};                                    % Density of solid
    cpSolid  = theta{24} * 1e3;                             %  [kJ/kg/K] -> [J/kg/K]

    k_solid  = 0.18;                                        % thermal conductivity of solid particles W/m/K - Some engineering and thermal properties of black cumin(Nigella sativaL.) seeds | 10.1111/j.1365-2621.2007.01561.x
    
    k_fluid = HeatConductivity_Comp(T, RHO) * 10^-3;        % mili W/m/K -> W/m/K
    cpFluid = CP;                                           % [J/kg/K]
    
    krhocp = ( (1-epsi) .* k_fluid + epsi .* k_solid ) ./ (cpFluid .* (1-epsi) .* RHO + cpSolid .* epsi .* rhoSolid);

end
