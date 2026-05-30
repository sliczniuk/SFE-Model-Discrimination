function d = upwind_2nd_order(U, U_0, U_B, dz)
% upwind_2nd_order - Second-order upwind finite difference for advection
%
% Uses 3-point stencil for interior points: (3*U_i - 4*U_{i-1} + U_{i-2}) / (2*dz)
% Falls back to first-order at boundaries where stencil is unavailable.
%
% Inputs:
%   U   - Vector of values [n x 1]
%   U_0 - Inlet boundary value (upstream)
%   U_B - Outlet boundary value (downstream, unused for upwind)
%   dz  - Grid spacing
%
% Output:
%   d   - First derivative dU/dz [n x 1]
%
% Note: Assumes positive velocity (flow from index 1 to n).
%       For better accuracy with same grid, or same accuracy with coarser grid.

    % Vectorized second-order upwind (works with both numeric and CasADi)
    % Point 1: first-order backward (only U_0 available)
    % Point 2: first-order backward (only one upstream point)
    % Points 3:n: second-order upwind stencil

    d = [
        (U(1) - U_0) / dz;                                      % 1st order at i=1
        (U(2) - U(1)) / dz;                                     % 1st order at i=2
        (3*U(3:end) - 4*U(2:end-1) + U(1:end-2)) / (2*dz)       % 2nd order upwind
    ];

end
