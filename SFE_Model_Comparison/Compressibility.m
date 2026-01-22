function Z = Compressibility(T, P, theta)
% Compressibility - Calculate compressibility factor using Cardano's formula
%
% Solves the Peng-Robinson cubic equation of state analytically.
%
% Inputs:
%   T     - Temperature [K]
%   P     - Pressure [bar]
%   theta - Parameters cell array
%
% Output:
%   Z     - Compressibility factor [-]

    Z = Cardano(T, P, theta);

end
