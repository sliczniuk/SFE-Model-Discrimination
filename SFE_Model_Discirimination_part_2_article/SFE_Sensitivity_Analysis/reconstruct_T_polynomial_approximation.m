function T = reconstruct_T_polynomial_approximation(H, P)
% reconstruct_T_polynomial_approximation - Reconstruct temperature from enthalpy
%
% Polynomial approximation: T(H,P) where H = log(-enthalpy_rho)
%
% Inputs:
%   H - Log of negative enthalpy*density [nstages x 1]
%   P - Pressure (scalar)
%
% Output:
%   T - Temperature [K]

    % Polynomial coefficients (fitted)
    p = [7550.3, -2139, 14.756, 204.02, -1.8946, -0.0085, -6.3579, 0.0576, 0.00068133];

    % Precompute powers (reduces operations)
    H2 = H .* H;
    H3 = H2 .* H;
    P2 = P * P;

    % Evaluate polynomial using Horner-like grouping
    % T = p1 + p2*H + p3*P + p4*H^2 + p5*H*P + p6*P^2 + p7*H^3 + p8*H^2*P + p9*H*P^2
    T = p(1) + ...
        H  .* (p(2) + p(4).*H + p(7).*H2) + ...
        P  .* (p(3) + p(5).*H + p(8).*H2) + ...
        P2 .* (p(6) + p(9).*H);

end
