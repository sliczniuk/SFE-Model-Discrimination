%% Parameter Estimation for Supercritical Fluid Extraction (SFE) - Chamomile
% This script performs parameter estimation for a supercritical fluid
% extraction model using CasADi optimization framework.
%
% The model estimates RBF parameters by minimizing the residuals between
% simulated and experimental extraction yields.

%% Initialization
startup;
delete(gcp('nocreate'));

addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
import casadi.*

%% Load Data
Parameters_table = readtable('Parameters.csv');
Parameters       = num2cell(Parameters_table{:,3});
Parameters_sym   = MX(cell2mat(Parameters));
LabResults       = xlsread('dataset_2.xlsx');

%% Physical Parameters
m_total = 3.0;  % Total mass [g]

% Bed geometry
before = 0.04;  % Fraction of length before bed (empty)
bed    = 0.92;  % Fraction of length occupied by fixed bed

%% Time Configuration
PreparationTime = 0;
ExtractionTime  = 600;
timeStep        = 5;  % [minutes]

simulationTime   = PreparationTime + ExtractionTime;
timeStep_in_sec  = timeStep * 60;
Time_in_sec      = (timeStep:timeStep:simulationTime) * 60;
Time             = [0 Time_in_sec/60];
N_Time           = length(Time_in_sec);

%% Sample Time Matching
SAMPLE = LabResults(6:19, 1);

% Robust sample index lookup (tolerates small rounding differences)
sample_tol = 1e-3; % minutes
N_Sample   = zeros(size(SAMPLE));
for ii = 1:numel(SAMPLE)
    [delta, idx] = min(abs(Time - SAMPLE(ii)));
    if delta > sample_tol
        error('Sample time mismatch at index %d (delta=%.3g min)', ii, delta);
    end
    N_Sample(ii) = idx;
end

%% Extractor Geometry
nstages = Parameters{1};
r       = Parameters{3};  % Radius [m]
epsi    = Parameters{4};  % Porosity [-]
L       = Parameters{6};  % Total length [m]

% Stage indices
nstagesbefore = 1:floor(before * nstages);
nstagesbed    = nstagesbefore(end)+1 : nstagesbefore(end) + floor(bed * nstages);
nstagesafter  = nstagesbed(end)+1 : nstages;

% Bed mask
bed_mask                = nan(nstages, 1);
bed_mask(nstagesbefore) = 0;
bed_mask(nstagesbed)    = 1;
bed_mask(nstagesafter)  = 0;

%% Volume Calculations
L_nstages = linspace(0, L, nstages);
V_slice   = (L/nstages) * pi * r^2;

V_before = V_slice * numel(nstagesbefore);
V_after  = V_slice * numel(nstagesafter);
V_bed    = V_slice * numel(nstagesbed);

V_before_fluid = repmat(V_before * 1          / numel(nstagesbefore), numel(nstagesbefore), 1);
V_bed_fluid    = repmat(V_bed    * (1 - epsi) / numel(nstagesbed),    numel(nstagesbed),    1);
V_after_fluid  = repmat(V_after  * 1          / numel(nstagesafter),  numel(nstagesafter),  1);
V_fluid        = [V_before_fluid; V_bed_fluid; V_after_fluid];

L_bed_after_nstages = L_nstages(nstagesbed(1):end);
L_bed_after_nstages = L_bed_after_nstages - L_bed_after_nstages(1);
L_end               = L_bed_after_nstages(end);

%% State and Input Dimensions
Nx = 3 * nstages + 2;          % States: C_f, C_s, H per stage + P(t) + yield
Nu = 3 + numel(Parameters);    % Inputs: T_in, P, F + parameters

%% Symbolic Variables
x = MX.sym('x', Nx);
u = MX.sym('u', Nu);

%% Initial Conditions
msol_max   = m_total;
mSol_ratio = 1;

mSOL_s = msol_max * mSol_ratio;
mSOL_f = msol_max * (1 - mSol_ratio);

C0solid       = mSOL_s * 1e-3 / (V_bed * epsi);
Parameters{2} = C0solid;

G       = @(x) -(2*mSOL_f / L_end^2) * (x - L_end);
m_fluid = G(L_bed_after_nstages) * L_bed_after_nstages(2);
m_fluid = [zeros(1, numel(nstagesbefore)) m_fluid];
C0fluid = m_fluid * 1e-3 ./ V_fluid';

%% Precompute bed mask constants (passed to modelSFE)
epsi_mask           = epsi .* bed_mask;
one_minus_epsi_mask = 1 - epsi_mask;

%% Build Integrator
% Note: N parameter not used by Langmuir model, pass empty []
f_linear = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, 'Linear_model', epsi_mask, one_minus_epsi_mask);
F_linear = buildIntegrator(f_linear, [Nx, Nu], timeStep_in_sec, 'cvodes');

f_power = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec, 'Power_model', epsi_mask, one_minus_epsi_mask);
F_power = buildIntegrator(f_power, [Nx, Nu], timeStep_in_sec, 'cvodes');

% Create mapaccum for efficient sequential simulation
F_accum_linear = F_linear.mapaccum('F_accum_linear', N_Time);
F_accum_power  = F_power.mapaccum('F_accum_power', N_Time);

%% Set operating conditions
T0homog   = 30+273;
feedPress = 200;
Flow      = 6.67 * 1e-5;

%% Set inital state and the vector of inputs
Z            = Compressibility(T0homog, feedPress, Parameters);
rho          = rhoPB_Comp(T0homog, feedPress, Z, Parameters);
enthalpy_rho = rho .* SpecificEnthalpy(T0homog, feedPress, Z, rho, Parameters);

feedTemp  = T0homog   * ones(1, N_Time);
feedPress = feedPress * ones(1, N_Time);
feedFlow  = Flow      * ones(1, N_Time);

uu = [feedTemp', feedPress', feedFlow'];

% Initial state
x0 = [C0fluid';
    C0solid * bed_mask;
    enthalpy_rho * ones(nstages, 1);
    feedPress(1);
    0];

% Build input matrix for all time steps [Nu x N_Time]
U_all = [uu'; repmat(cell2mat(Parameters), 1, N_Time)];

%% Simulation using mapaccum (more efficient than loop)
X_all_linear = F_accum_linear(x0, U_all);
X_all_power  = F_accum_power(x0, U_all);

%%
hold on
plot(full(X_all_linear(end,:)))
plot(full(X_all_power(end,:)))
hold off































