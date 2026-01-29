function xdot = modelSFE_power_with_params(x, p, mask, dt, epsi_mask, one_minus_epsi_mask, varargin)
% MODELSFE_POWER_WITH_PARAMS
% SFE Model using Power (two-kinetic) extraction model with EXPLICIT parameters.
% This allows uncertainty propagation by passing different parameter values.
%
% Inputs:
%   x                   - State vector [C_fluid; C_solid; enthalpy_rho; pressure; yield]
%   p                   - Input/parameters cell array {T_u, P_u, F_u, parameters...}
%   mask                - Bed mask (1 = bed, 0 = empty)
%   dt                  - Time step [s]
%   epsi_mask           - Precomputed epsi .* mask
%   one_minus_epsi_mask - Precomputed 1 - epsi .* mask
%   k_w0, a_w, b_w, n_k - Power model kinetic parameters
%   n_param             - Number of base parameters when passing packed vector p

import casadi.*

%% Unpack inputs
if numel(varargin) == 1
    n_param = varargin{1};
    if iscell(p)
        T_u = p{1};
        P_u = p{2};
        F_u = p{3};
        parameters = p(4:3+n_param);
        theta = p(4+n_param:end);
        if iscell(theta)
            theta = cell2mat(theta);
        end
    else
        T_u = p(1);
        P_u = p(2);
        F_u = p(3);
        parameters = num2cell(p(4:3+n_param));
        theta = p(4+n_param:end);
    end
    if numel(theta) ~= 4
        error('modelSFE_power_with_params:thetaSize', ...
            'Expected 4 theta parameters, got %d.', numel(theta));
    end
    k_w0 = theta(1);
    a_w = theta(2);
    b_w = theta(3);
    n_k = theta(4);
elseif numel(varargin) == 4
    k_w0 = varargin{1};
    a_w = varargin{2};
    b_w = varargin{3};
    n_k = varargin{4};
    if iscell(p)
        T_u = p{1};
        P_u = p{2};
        F_u = p{3};
        parameters = p(4:end);
    else
        T_u = p(1);
        P_u = p(2);
        F_u = p(3);
        parameters = num2cell(p(4:end));
    end
else
    error('modelSFE_power_with_params:args', ...
        'Expected n_param or explicit kinetic parameters.');
end

%% Parameters
C0solid       = parameters{2};
r             = parameters{3};
epsi          = parameters{4};
dp            = parameters{5};
L             = parameters{6};
rho_s         = parameters{7};
mi            = parameters{9};

%% Properties of the bed
A   = pi*r^2;
rp  = dp / 2;
lp2 = (rp / 3)^2;

nstages_index = numel(mask);

%% States
FLUID        = x(1:nstages_index);
SOLID        = x(nstages_index+1:2*nstages_index);
ENTHALPY_RHO = x(2*nstages_index+1:3*nstages_index);
PRESSURE     = x(3*nstages_index+1);

%% Temperature
TEMP = T_u * ones(nstages_index, 1);

%% Fluid properties
Z        = Compressibility(TEMP, PRESSURE, parameters);
RHO      = rhoPB_Comp(TEMP, PRESSURE, Z, parameters);
VELOCITY = Velocity(F_u, RHO(round(nstages_index/2)), parameters);

%% Thermal properties
MU = Viscosity(TEMP, RHO);
RE = dp .* RHO .* VELOCITY ./ MU .* 1.3;

CP     = SpecificHeatComp(TEMP, PRESSURE, Z, RHO, parameters);
KRHOCP = kRHOcp_Comp(TEMP, PRESSURE, Z, RHO, CP, epsi_mask, parameters);

%% Extraction kinetics - POWER MODEL with explicit parameters
Csolid_percentage_left = (1 - SOLID./C0solid) .* mask;

% Two-kinetic model with passed parameters
k_w = k_w0 .* (RHO ./ 800).^a_w .* ((F_u * 1e5) ./ 5).^b_w .* 1e-4;
beta = 1 ./ ((Csolid_percentage_left + 1).^n_k);
re = (k_w .* beta) .* SOLID;

%% Boundary conditions
Cf_0 = 0;
Cf_B = FLUID(nstages_index);
T_0  = T_u;
T_B  = TEMP(nstages_index);

% Inlet properties
Z_0   = Compressibility(T_0, PRESSURE, parameters);
rho_0 = rhoPB_Comp(T_0, P_u, Z_0, parameters);
H_0   = SpecificEnthalpy(T_0, PRESSURE, Z_0, rho_0, parameters);
enthalpy_rho_0 = rho_0 .* H_0;

%% Spatial derivatives
dz = L / nstages_index;

d2Tdz2       = central_diff_2_order(TEMP, T_0, T_B, dz);
dHdz         = upwind_2nd_order(VELOCITY .* ENTHALPY_RHO, VELOCITY .* enthalpy_rho_0, [], dz);
d_cons_CF_dz = upwind_2nd_order(VELOCITY .* FLUID, VELOCITY .* Cf_0, [], dz);

%% Temporal derivative
dPdt = backward_diff_1_order(P_u, PRESSURE, [], dt) * 1e2;

%% Model equations
xdot = [
    % Fluid phase concentration
    (-d_cons_CF_dz + epsi_mask .* re) ./ one_minus_epsi_mask;

    % Solid phase concentration
    -mask .* re;

    % Enthalpy
    -dHdz ./ one_minus_epsi_mask + dPdt + KRHOCP .* d2Tdz2;

    % Pressure
    dPdt;

    % Output: mass flow rate [g/s]
    F_u ./ RHO(nstages_index) .* FLUID(nstages_index) * 1e3;
];

end
