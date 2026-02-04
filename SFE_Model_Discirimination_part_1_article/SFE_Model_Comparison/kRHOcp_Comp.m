function krhocp = kRHOcp_Comp(T, P, Z, RHO, CP, epsi, theta)
% kRHOcp_Comp - Calculate effective thermal diffusivity for packed bed
%
% Computes k/(rho*cp) for the fluid-solid mixture in the extractor.
% Reference: http://www.ffrc.fi/FlameDays_2009/3B/HankalinPaper.pdf
%
% Inputs:
%   T     - Temperature [K]
%   P     - Pressure [bar] (unused, kept for interface compatibility)
%   Z     - Compressibility factor [-] (unused)
%   RHO   - Fluid density [kg/m3]
%   CP    - Fluid specific heat [J/kg/K]
%   epsi  - Void fraction (can be vector with bed mask applied)
%   theta - Parameters cell array
%
% Output:
%   krhocp - Effective thermal diffusivity [m2/s]

    %% Extract parameters
    rhoSolid = theta{7};         % Solid density [kg/m3]
    cpSolid  = theta{24} * 1e3;  % Solid specific heat [kJ/kg/K -> J/kg/K]

    %% Thermal conductivities
    k_solid = 0.18;  % Solid thermal conductivity [W/(m·K)]
    k_fluid = HeatConductivity_Comp(T, RHO) * 1e-3;  % [mW/(m·K) -> W/(m·K)]

    %% Effective thermal diffusivity
    % Weighted average conductivity / weighted average (rho*cp)
    k_eff     = (1-epsi) .* k_fluid + epsi .* k_solid;
    rhocp_eff = CP .* (1-epsi) .* RHO + cpSolid .* epsi .* rhoSolid;

    krhocp = k_eff ./ rhocp_eff;

end
