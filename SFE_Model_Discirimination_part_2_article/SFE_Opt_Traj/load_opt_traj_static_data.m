function static = load_opt_traj_static_data(config)
%LOAD_OPT_TRAJ_STATIC_DATA Load and precompute seed-independent data.

t_total = tic;
t_io = tic;

Parameters_table = readtable('Parameters.csv');
Parameters       = num2cell(Parameters_table{:,3});

if exist('readmatrix', 'file') == 2
    try
        LabResults = readmatrix('dataset_2.xlsx');
    catch
        LabResults = xlsread('dataset_2.xlsx');
    end
else
    LabResults = xlsread('dataset_2.xlsx');
end

t_file_io = toc(t_io);
t_pre = tic;

%% Parameter indexing
which_k = (0:9) + 44;  % Indices of parameters to fit (44-53)
k1_val  = cell2mat(Parameters((0:3) + 44));
k2_val  = cell2mat(Parameters((4:9) + 44));

%% Time grid and sample matching
timeStep  = config.timeStep;
finalTime = config.finalTime;
Time      = 0:timeStep:finalTime;

SAMPLE = LabResults(6:19, 1);
SAMPLE = SAMPLE(2:end);

N_Sample = zeros(size(SAMPLE));
for ii = 1:numel(SAMPLE)
    [delta, idx] = min(abs(Time - SAMPLE(ii)));
    if delta > config.sample_tol
        error('Sample time mismatch at index %d (delta=%.3g min)', ii, delta);
    end
    N_Sample(ii) = idx;
end

%% Parameter covariance matrices
% Power model covariance (4x4): [k_w0, a_w, b_w, n_k]
Cov_power_cum = [
    1.0035e-02,  1.1795e-02,  1.8268e-03,  2.5611e-02;
    1.1795e-02,  5.6469e-02,  3.1182e-03,  2.8266e-02;
    1.8268e-03,  3.1182e-03,  5.7241e-03,  6.4459e-03;
    2.5611e-02,  2.8266e-02,  6.4459e-03,  7.1744e-02
    ];

Cov_power_diff = [
    3.2963e-03,  1.2094e-03, -2.5042e-03,  6.8414e-03;
    1.2094e-03,  1.0981e-01, -5.7125e-04,  2.3381e-03;
    -2.5042e-03, -5.7125e-04,  1.3915e-02, -2.7301e-04;
    6.8414e-03,  2.3381e-03, -2.7301e-04,  3.8686e-02
    ];

Cov_power_norm = [
    2.9603e-03,  6.8345e-03,  8.7769e-07,  5.6113e-03;
    6.8345e-03,  7.7672e-02,  2.0806e-03,  1.3146e-03;
    8.7769e-07,  2.0806e-03,  8.2066e-03, -2.9670e-04;
    5.6113e-03,  1.3146e-03, -2.9670e-04,  3.1794e-02
    ];

% Linear model covariance (6x6): [D_i(0), D_i(Re), D_i(F), Ups(0), Ups(Re), Ups(F)]
Cov_linear_cum = [
    2.7801e-02,  3.5096e-02, -6.9596e-03,  7.1573e-02,  1.0992e-02, -1.2661e-02;
    3.5096e-02,  6.8482e-01, -5.0531e-02, -4.8187e-02,  3.9209e-01, -2.6206e-02;
    -6.9596e-03, -5.0531e-02,  4.5693e-03, -7.7054e-03, -1.5915e-02,  3.3012e-03;
    7.1573e-02, -4.8187e-02, -7.7054e-03,  2.9254e-01,  6.5758e-02, -4.6300e-02;
    1.0992e-02,  3.9209e-01, -1.5915e-02,  6.5758e-02,  2.9506e+00, -1.3133e-01;
    -1.2661e-02, -2.6206e-02,  3.3012e-03, -4.6300e-02, -1.3133e-01,  1.2975e-02
    ];

Cov_linear_diff = [
    2.2178e-02,  1.0828e-02, -4.3832e-03,  4.3992e-02,  4.4695e-03, -7.4634e-03;
    1.0828e-02,  4.3513e-01, -2.4832e-02,  3.0423e-03,  6.7633e-01, -3.3289e-02;
    -4.3832e-03, -2.4832e-02,  2.1282e-03, -7.3766e-03, -3.2298e-02,  2.8884e-03;
    4.3992e-02,  3.0423e-03, -7.3766e-03,  3.1429e-01, -6.2258e-02, -4.7085e-02;
    4.4695e-03,  6.7633e-01, -3.2298e-02, -6.2258e-02,  6.1032e+00, -2.4975e-01;
    -7.4634e-03, -3.3289e-02,  2.8884e-03, -4.7085e-02, -2.4975e-01,  1.8474e-02
    ];

Cov_linear_norm = [
    1.2717e-02,  2.8660e-02, -4.0477e-03,  2.6437e-02,  1.8916e-02, -5.8348e-03;
    2.8660e-02,  4.4243e-01, -3.1689e-02,  2.2351e-02,  6.5357e-01, -4.0257e-02;
    -4.0477e-03, -3.1689e-02,  2.6805e-03, -5.9892e-03, -3.8209e-02,  3.3147e-03;
    2.6437e-02,  2.2351e-02, -5.9892e-03,  2.0283e-01, -7.3693e-02, -3.1444e-02;
    1.8916e-02,  6.5357e-01, -3.8209e-02, -7.3693e-02,  5.9933e+00, -2.6958e-01;
    -5.8348e-03, -4.0257e-02,  3.3147e-03, -3.1444e-02, -2.6958e-01,  1.8713e-02
    ];

sigma2_cases = [2.45e-2, 1.386e-3, 1.007e-2];

%% Setup simulation infrastructure
m_total = 3.0;
before  = 0.04;
bed     = 0.92;

Time_in_sec     = (timeStep:timeStep:finalTime) * 60;
Time            = [0, Time_in_sec/60];
N_Time          = length(Time_in_sec);
timeStep_in_sec = timeStep * 60;

nstages = Parameters{1};
r       = Parameters{3};
epsi    = Parameters{4};
L       = Parameters{6};

nstagesbefore = 1:floor(before * nstages);
nstagesbed    = nstagesbefore(end)+1 : nstagesbefore(end) + floor(bed * nstages);
nstagesafter  = nstagesbed(end)+1 : nstages;

bed_mask = nan(nstages, 1);
bed_mask(nstagesbefore) = 0;
bed_mask(nstagesbed)    = 1;
bed_mask(nstagesafter)  = 0;

V_slice = (L/nstages) * pi * r^2;
V_bed   = V_slice * numel(nstagesbed);

V_before_fluid = repmat(V_slice * numel(nstagesbefore) / numel(nstagesbefore), numel(nstagesbefore), 1);
V_bed_fluid    = repmat(V_bed * (1-epsi) / numel(nstagesbed), numel(nstagesbed), 1);
V_after_fluid  = repmat(V_slice * numel(nstagesafter) / numel(nstagesafter), numel(nstagesafter), 1);
V_fluid        = [V_before_fluid; V_bed_fluid; V_after_fluid];

C0solid = m_total * 1e-3 / (V_bed * epsi);
Parameters{2} = C0solid;

m_fluid = zeros(1, nstages);
C0fluid = m_fluid * 1e-3 ./ V_fluid';

epsi_mask           = epsi .* bed_mask;
one_minus_epsi_mask = 1 - epsi_mask;

Nx = 3 * nstages + 2;
Nu = 3 + numel(Parameters);

%% Input normalization constants
T_min = config.T_min;
T_max = config.T_max;
F_min = config.F_min;
F_max = config.F_max;

T_mid  = 0.5 * (T_min + T_max);
T_half = 0.5 * (T_max - T_min);
F_mid  = 0.5 * (F_min + F_max);
F_half = 0.5 * (F_max - F_min);

feedPress = config.feedPressValue * ones(1, N_Time);

t_precompute = toc(t_pre);

static = struct;
static.Parameters       = Parameters;
static.which_k          = which_k;
static.k1_val           = k1_val;
static.k2_val           = k2_val;
static.LabResults       = LabResults;
static.SAMPLE           = SAMPLE;
static.N_Sample         = N_Sample;
static.Cov_power_cum    = Cov_power_cum;
static.Cov_power_diff   = Cov_power_diff;
static.Cov_power_norm   = Cov_power_norm;
static.Cov_linear_cum   = Cov_linear_cum;
static.Cov_linear_diff  = Cov_linear_diff;
static.Cov_linear_norm  = Cov_linear_norm;
static.sigma2_cases     = sigma2_cases;
static.Time             = Time;
static.N_Time           = N_Time;
static.Time_in_sec      = Time_in_sec;
static.timeStep         = timeStep;
static.finalTime        = finalTime;
static.timeStep_in_sec  = timeStep_in_sec;
static.nstages          = nstages;
static.bed_mask         = bed_mask;
static.epsi_mask        = epsi_mask;
static.one_minus_epsi_mask = one_minus_epsi_mask;
static.C0solid          = C0solid;
static.C0fluid          = C0fluid;
static.Nx               = Nx;
static.Nu               = Nu;
static.feedPress        = feedPress;
static.T_min            = T_min;
static.T_max            = T_max;
static.F_min            = F_min;
static.F_max            = F_max;
static.T_mid            = T_mid;
static.T_half           = T_half;
static.F_mid            = F_mid;
static.F_half           = F_half;

static.timings = struct;
static.timings.t_file_io     = t_file_io;
static.timings.t_precompute  = t_precompute;
static.timings.t_static_total = t_file_io + t_precompute;
static.timings.t_total       = toc(t_total);

end
