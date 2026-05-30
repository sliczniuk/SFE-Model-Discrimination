function Cp = SpecificHeatComp(T, P, Z, RHO, theta)
% SpecificHeatComp - Calculate specific heat capacity for real gas (Peng-Robinson)
%
% Uses departure function method for real gas correction.
% References:
%   https://cheguide.com/specific_heat_ratio.html
%   https://www.sciencedirect.com/science/article/abs/pii/S0017931014006577
%
% Inputs:
%   T     - Temperature [K]
%   P     - Pressure [bar]
%   Z     - Compressibility factor [-]
%   RHO   - Density [kg/m3]
%   theta - Parameters cell array
%
% Output:
%   Cp    - Specific heat capacity [J/kg/K]

    %% Extract parameters
    Tc   = theta{10};    % Critical temperature [K]
    Pc   = theta{11};    % Critical pressure [bar]
    MW   = theta{14};    % Molar mass [g/mol]
    CP_A = theta{18};    % Ideal gas Cp coefficients
    CP_B = theta{19};
    CP_C = theta{20};
    CP_D = theta{21};

    %% Constants
    R     = 83.1447;     % Universal gas constant [cm3·bar/(mol·K)]
    kappa = 0.2250;      % Peng-Robinson acentric factor parameter
    m     = 0.37464 + 1.54226*kappa - 0.26992*kappa^2;

    %% Peng-Robinson parameters
    Tr    = T ./ Tc;
    Pr    = P ./ Pc;
    alpha = (1 + m .* (1 - sqrt(Tr))).^2;

    a = 0.45724 .* R^2 .* Tc^2 .* alpha ./ Pc;
    b = 0.0777961 .* R .* Tc ./ Pc;

    A = 0.45723553 .* alpha .* Pr ./ Tr.^2;
    B = b .* P ./ (R .* T);

    % Molar volume [cm3/mol]
    v = MW ./ RHO * 1e3;

    %% Thermodynamic derivatives
    % Common term: v(v+b) + b(v-b)
    vb_term = v.*(v+b) + b.*(v-b);

    % (∂a/∂T)_V
    dadT = -m .* a ./ (sqrt(T.*Tc) .* (1 + m .* (1 - sqrt(Tr))));

    % (∂A/∂T)_P
    dAdT = (P ./ (R.*T).^2) .* (dadT - 2*a./T);

    % (∂B/∂T)_P
    dBdT = -b .* P ./ (R .* T.^2);

    % (∂P/∂T)_V
    dPdT = R ./ (v-b) - dadT ./ vb_term;

    % (∂Z/∂T)_P
    Num   = dAdT .* (B-Z) + dBdT .* (6*B.*Z + 2.*Z - 3*B.^2 - 2*B + A - Z.^2);
    Denom = 3.*Z.^2 + 2.*(B-1).*Z + (A - 2.*B - 3.*B.^2);
    dZdT  = Num ./ Denom;

    % (∂V/∂T)_P
    dVdT = (R./P) .* (T.*dZdT + Z);

    % (∂²a/∂T²)
    d2adT2 = a .* m .* (1+m) .* sqrt(Tc./T) ./ (2.*T.*Tc);

    %% Ideal gas heat capacity
    Cp_Ideal = CP_A + CP_B.*T + CP_C.*T.^2 + CP_D.*T.^3;

    %% Real gas correction (departure function)
    sqrt2 = sqrt(2);
    sqrt8 = sqrt(8);

    % Cv correction
    log_term = log((Z + B.*(1+sqrt2)) ./ (Z + B.*(1-sqrt2)));
    Cv_corr  = (T .* d2adT2 ./ (b .* sqrt8)) .* log_term ./ 10;

    % Cp correction
    Cp_corr = Cv_corr + T .* dPdT .* dVdT / 10 - R/10;

    %% Final Cp [J/kg/K]
    Cp = (Cp_Ideal + Cp_corr) ./ MW;

end
