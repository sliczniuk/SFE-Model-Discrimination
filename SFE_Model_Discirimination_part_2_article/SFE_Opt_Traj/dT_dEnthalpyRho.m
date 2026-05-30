function [dTdHR, dTdP] = dT_dEnthalpyRho(ENTHALPY_RHO, PRESSURE)
% dT_dEnthalpyRho - Analytical partial derivatives of the polynomial
%                   temperature reconstruction T = f(xi, P)
%                   where xi = log(-ENTHALPY_RHO)
%
% Inputs:
%   ENTHALPY_RHO  - Enthalpy density rho*h  [J/m3]  (negative valued)
%   PRESSURE      - Reduced pressure P/Pc   [-]
%
% Outputs:
%   dTdHR  - dT/d(rho*h)   [K·m3/J]
%   dTdP   - dT/dP         [K]        (with respect to reduced pressure)

    xi  = log(-ENTHALPY_RHO);          % xi = ln(-rho*h)
    P   = PRESSURE;

    % Polynomial coefficients (from Eq. 3.13 in thesis)
    % T = a0 + a1*xi + a2*P + a3*xi^2 + a4*xi*P + a5*P^2
    %       + a6*xi^3 + a7*xi^2*P + a8*xi*P^2

    % dT/dxi  (chain rule: dT/d(rho*h) = dT/dxi * dxi/d(rho*h))
    dTdxi = 0.1185 ...
          + 2 * 0.05594  .* xi ...
          -     0.2521   .* P ...
          + 3 * (-0.04466) .* xi.^2 ...
          + 2 * 0.07846  .* xi .* P ...
          -     0.01329  .* P.^2;

    % dxi/d(rho*h) = d/d(rho*h) [ ln(-(rho*h)) ] = -1/(rho*h)
    % Note: ENTHALPY_RHO is negative, so -1/ENTHALPY_RHO is positive
    dxi_dHR = -1 ./ ENTHALPY_RHO;

    dTdHR = dTdxi .* dxi_dHR;

    % dT/dP
    dTdP  =  0.2601 ...
          -  0.2521  .* xi ...
          + 2 * 0.02087 .* P ...
          +  0.07846  .* xi.^2 ...
          - 2 * 0.01329 .* xi .* P;
end