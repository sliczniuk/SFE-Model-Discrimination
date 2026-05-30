function Z = Cardano(T, P, theta)
% Cardano - Analytical solution to Peng-Robinson cubic EOS
%
% Solves: Z^3 + UC*Z^2 + SC*Z + TC = 0
% using Cardano's formula for the case D > 0 (one real root)
%
% Inputs:
%   T     - Temperature [K]
%   P     - Pressure [bar]
%   theta - Parameters cell array (uses theta{14} for MW)
%
% Output:
%   Z     - Compressibility factor [-]

    % Convert pressure: bar -> MPa for internal calculation
    P = P ./ 10;

    % CO2 critical properties (constants)
    Tc    = 304.2;      % Critical temperature [K]
    Pc    = 7.382;      % Critical pressure [MPa]
    R     = 8.314472;   % Universal gas constant [J/(mol·K)]
    kappa = 0.228;      % Peng-Robinson parameter for CO2

    % Peng-Robinson parameters (precomputed constants)
    a_const = 0.45723555289 * (R * Tc)^2 / Pc;  % ~3.6439e5
    b_const = 0.0777961 * R * Tc / Pc;          % ~26.65

    % Temperature-dependent terms
    Tr    = T ./ Tc;
    alpha = (1 + kappa .* (1 - sqrt(Tr))).^2;

    % Dimensionless parameters
    A = a_const .* alpha .* P ./ (R^2 .* T.^2);
    B = b_const .* P ./ (R .* T);

    % Cubic equation coefficients: Z^3 + UC*Z^2 + SC*Z + TC = 0
    UC = -(1 - B);
    SC = A - 2.*B - 3.*B.^2;
    TC = -(A.*B - B.^2 - B.^3);

    % Cardano's method - depressed cubic transformation
    PC = (3.*SC - UC.^2) ./ 3;
    QC = (2.*UC.^3 - 9.*UC.*SC + 27.*TC) ./ 27;

    % Discriminant
    DC = (PC./3).^3 + (QC./2).^2;

    % Solution for D > 0 (one real root - gas phase)
    sqrt_DC = sqrt(DC);
    term1 = sqrt_DC - QC./2;
    Z = term1.^(1/3) - PC ./ (3 .* term1.^(1/3)) - UC./3;

end
