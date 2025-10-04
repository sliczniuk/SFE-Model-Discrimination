%addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
startup;
delete(gcp('nocreate'));

addpath('\\home.org.aalto.fi\sliczno1\data\Documents\casadi-3.6.3-windows64-matlab2018b');
%addpath('casadi_folder')
import casadi.*

%% Create the solver
Iteration_max               = 20;                                         % Maximum number of iterations for optimzer
%Time_max                    = 24;                                         % Maximum time of optimization in [h]

nlp_opts                    = struct;
nlp_opts.ipopt.max_iter     = Iteration_max;
%nlp_opts.ipopt.max_cpu_time = Time_max*3600;
nlp_opts.ipopt.hessian_approximation ='limited-memory';

%%
Parameters_table        = readtable('Parameters.csv') ;                     % Table with prameters
Parameters              = num2cell(Parameters_table{:,3});                  % Parameters within the model + (m_max), m_ratio, sigma

LabResults              = xlsread('dataset_2.xlsx');

sigma                   = 3e-3;

Q_RBF                   = readmatrix('Q_RBF_1.txt');
Q_FP                    = readmatrix('Q_FP_1.txt');

%% Load paramters
m_total                 = 3.0;

% Bed geometry
before                  = 0.04;                                             % Precentage of length before which is empty
bed                     = 0.92;                                              % Percentage of length occupied by fixed bed

% Set time of the simulation
PreparationTime         = 0;
ExtractionTime          = 600;
timeStep                = 10;                                                % Minutes
OP_change_Time          = 30;
Sample_Time             = 10;

simulationTime          = PreparationTime + ExtractionTime;

timeStep_in_sec         = timeStep * 60;                                    % Seconds
Time_in_sec             = (timeStep:timeStep:simulationTime)*60;            % Seconds
Time                    = [0 Time_in_sec/60];                               % Minutes

N_Time                  = length(Time_in_sec);

SAMPLE                  = Sample_Time:Sample_Time:ExtractionTime;

OP_change               = 0:OP_change_Time:ExtractionTime-OP_change_Time;

% Check if the number of data points is the same for both the dataset and the simulation
N_Sample                = [];
for i = 1:numel(SAMPLE)
    N_Sample            = [N_Sample ; find(round(Time,3) == round(SAMPLE(i))) ];
end
if numel(N_Sample) ~= numel(SAMPLE)
    keyboard
end

%% Specify parameters to estimate
nstages                 = Parameters{1};

nstagesbefore           = 1:floor(before*nstages);
nstagesbed              = nstagesbefore(end)+1 : nstagesbefore(end) + floor(bed*nstages);
nstagesafter            = nstagesbed(end)+1:nstages;

bed_mask                = nan(nstages,1);
bed_mask(nstagesbefore) = 0;
bed_mask(nstagesbed)    = 1;
bed_mask(nstagesafter)  = 0;

%% Number of variables
Nx_RBF                  = (1+36)*(3 * nstages+3);                           % 3*Nstages(C_f, C_s, H) + P(t) + yield
Nx_FP                   = (1+6)*(3 * nstages+3);                            % 3*Nstages(C_f, C_s, H) + P(t) + yield
Nu                      = 3 + numel( Parameters );                          % T_in, P, F + numel(Parameters)

%% Extractor geometry
r                       = Parameters{3};                                    % Radius of the extractor  [m]
epsi                    = Parameters{4};                                    % Fullness [-]
dp                      = Parameters{5};                                    % Paritcle diameter
L                       = Parameters{6};                                    % Total length of the extractor [m]

L_nstages               = linspace(0,L,nstages);
V                       = L  * pi * r^2;                                    % Total volume of the extractor [m3]
A                       = pi *      r^2;                                    % Extractor cross-section

%--------------------------------------------------------------------
V_slice                 = (L/nstages) * pi * r^2;

V_before                = V_slice * numel(nstagesbefore);
V_after                 = V_slice * numel(nstagesafter);
V_bed                   = V_slice * numel(nstagesbed);                      % Volume of the fixed bed [m3]

V_before_solid          = repmat(V_before * 0          / numel(nstagesbefore), numel(nstagesbefore),1);
V_bed_solid             = repmat(V_bed    * epsi       / numel(nstagesbed)   , numel(nstagesbed)   ,1);
V_after_solid           = repmat(V_after  * 0          / numel(nstagesbed)   , numel(nstagesafter) ,1);

V_solid                 = [V_before_solid; V_bed_solid; V_after_solid];

V_before_fluid          = repmat(V_before * 1          / numel(nstagesbefore), numel(nstagesbefore),1);
V_bed_fluid             = repmat(V_bed    * (1 - epsi) / numel(nstagesbed)   , numel(nstagesbed)   ,1);
V_after_fluid           = repmat(V_after  * 1          / numel(nstagesafter) , numel(nstagesafter) ,1);

V_fluid                 = [V_before_fluid; V_bed_fluid; V_after_fluid];

L_bed_after_nstages     = L_nstages(nstagesbed(1):end);
L_bed_after_nstages     = L_bed_after_nstages - L_bed_after_nstages(1);
L_end                   = L_bed_after_nstages(end);

%% symbolic variables
x_RBF                   = MX.sym('x', Nx_RBF);
x_FP                    = MX.sym('x', Nx_FP);
u                       = MX.sym('u', Nu);

%% Set inital state and inital conditions
msol_max                = m_total;                                          % g of product in solid and fluid phase
mSol_ratio              = 1;

mSOL_s                  = msol_max*mSol_ratio;                              % g of product in biomass
mSOL_f                  = msol_max*(1-mSol_ratio);                          % g of biomass in fluid

C0solid                 = mSOL_s * 1e-3 / ( V_bed * epsi)  ;                % Solid phase kg / m^3
Parameters{2}           = C0solid;

G                       =@(x) -(2*mSOL_f / L_end^2) * (x-L_end) ;

m_fluid                 = G(L_bed_after_nstages)*( L_bed_after_nstages(2) ); % Lienarly distirubuted mass of solute in fluid phase, which goes is zero at the outlet. mass*dz
m_fluid                 = [zeros(1,numel(nstagesbefore)) m_fluid];
C0fluid                 = m_fluid * 1e-3 ./ V_fluid';

%% Set Integrator - RBF
f_corr                   = @(x, u) modelSFE_Corr_sensitivity(x, u, bed_mask, timeStep_in_sec);
f_RBF                    = @(x, u) modelSFE_RBF_sensitivity(x, u, bed_mask, timeStep_in_sec);

% Integrator
[F_corr]                 = buildIntegrator(f_corr, [Nx_FP,Nu] , timeStep_in_sec);
[F_RBF]                  = buildIntegrator(f_RBF, [Nx_RBF,Nu] , timeStep_in_sec);

%%
%theta                   = MX.sym('theta', size(Parameters));
Parameters_sym          = MX(cell2mat(Parameters));

% Set operating conditions
OPT_solver                  = casadi.Opti();
ocp_opts                    = {'nlp_opts', nlp_opts};
OPT_solver.solver(             'ipopt'   , nlp_opts)

T0homog                 = OPT_solver.variable(numel(OP_change))';
OPT_solver.subject_to( 30+273 <= T0homog <= 40+273 );

Flow                    = OPT_solver.variable(numel(OP_change))';
OPT_solver.subject_to( 3.33 <= Flow <= 6.67 );

feedPress               = 100;               % MPa -> bar

feedTemp                = repmat(T0homog,OP_change_Time/timeStep,1);
feedTemp                = feedTemp(:)';
feedTemp                = [ feedTemp, T0homog(end)*ones(1,N_Time - numel(feedTemp)) ];

feedPress               = feedPress * ones(1,numel(feedTemp)) + 0 ;     % Bars

F_0                     = 5;
feedFlow                = repmat(Flow,OP_change_Time/timeStep,1);
feedFlow                = feedFlow(:)';
feedFlow                = [ feedFlow, Flow(end)*ones(1,N_Time - numel(feedFlow)) ];

Z                       = Compressibility( T0homog, feedPress,         Parameters );
rho                     = rhoPB_Comp(      T0homog, feedPress, Z,      Parameters );

enthalpy_rho            = rho.*SpecificEnthalpy(T0homog, feedPress, Z, rho, Parameters );

uu                      = [feedTemp', feedPress', feedFlow'];

Z                       = Compressibility(      feedTemp(1), feedPress(1),         Parameters );
rho                     = rhoPB_Comp(           feedTemp(1), feedPress(1), Z,      Parameters );
enthalpy_rho            = rho.*SpecificEnthalpy(feedTemp(1), feedPress(1), Z, rho, Parameters ) ;

% Initial conditions
x0_RBF                  = [ C0fluid'                         ;
                            C0solid         * bed_mask       ;
                            enthalpy_rho    * ones(nstages,1);
                            feedPress(1)                     ;
                            0                                ;
                            0                                ;
                            zeros(Nx_RBF - (3*nstages+3), 1) ;
                            ];

x0_FP                   = [ C0fluid'                         ;
                            C0solid         * bed_mask       ;
                            enthalpy_rho    * ones(nstages,1);
                            feedPress(1)                     ;
                            0                                ;
                            0                                ;
                            zeros(Nx_FP  - (3*nstages+3), 1) ;
                            ];

Parameters_init_time   = [uu repmat(cell2mat(Parameters),1,N_Time)'];
[xx_R]                 = simulateSystem(F_RBF, [], x0_RBF, Parameters_init_time );
[xx_F]                 = simulateSystem(F_corr , [], x0_FP, Parameters_init_time );


%%

XX_FP     = xx_F(3*nstages+2:3*nstages+3:Nx_FP,:);
yy_FP     = XX_FP(1,N_Sample);
S_FP      = XX_FP(2:end,N_Sample);
FI_FP     = sigma*eye(length(N_Sample)) + (S_FP' * pinv(Q_FP./sigma) * S_FP);
SIGMA_FP  = pinv(FI_FP);

%{\
XX_RBF    = xx_R(3*nstages+2:3*nstages+3:Nx_RBF,:);
yy_RBF    = XX_RBF(1,N_Sample);
S_RBF     = XX_RBF(2:end,N_Sample);
FI_RBF    = sigma*eye(length(N_Sample)) + (S_RBF' * pinv(Q_RBF./sigma) * S_RBF);
SIGMA_RBF = pinv(FI_RBF);

residuals = yy_RBF - yy_FP;

JJ_1      = trace(FI_FP * SIGMA_RBF + FI_RBF * SIGMA_FP - 2*eye(size(FI_FP)));
JJ_2      = residuals * [SIGMA_RBF + SIGMA_FP] * residuals';


%{
%%
T0 = ( (40-30).*rand(1,numel(T0homog)*N_exp) + 30 )+273;
F0 = ( (3.33-6.67).*rand(1,numel(Flow)*N_exp) + 3.33 );

JJ = -(JJ_1 + JJ_2);

%ControlEffort_F = diff(Flow)    * (diag(ones(1,numel(diff(Flow))))    ) * diff(Flow)'   ;
%ControlEffort_T = diff(T0homog) * (diag(ones(1,numel(diff(T0homog)))) ) * diff(T0homog)';

JJ = JJ; %+ ControlEffort_T + ControlEffort_F;

%% Solve the optimization problem
OPT_solver.minimize(JJ);

H = OPT_solver.to_function('H',{T0homog,Flow},{T0homog,Flow});

%parpool()
H_map = H.map(N_exp,'thread',N_core);
[HH_1, HH_2] = H_map(T0, F0);

HH_1 = full(HH_1);
HH_2 = full(HH_2);

tic
GG        = Function('GG',{T0homog,Flow}, {JJ} );
COST_0    = full(GG(T0, F0 ));
COST      = full(GG(HH_1, HH_2 ));

FY_RBF    = Function('FY_RBF',{T0homog, Flow}, {XX_RBF(1,:)} );
Y_RBF_0   = full(FY_RBF(T0, F0));
Y_RBF     = full(FY_RBF(HH_1,HH_2));

FY_FP     = Function('FY_FP',{T0homog, Flow}, {XX_FP(1,:)} );
Y_FP_0    = full(FY_FP(T0, F0));
Y_FP      = full(FY_FP(HH_1, HH_2));

Fisher    = Function('Fisher',{T0homog, Flow}, {S_RBF} );
Fisher_RBF= full(Fisher(HH_1, HH_2 ));

Fisher    = Function('Fisher',{T0homog, Flow}, {S_FP} );
Fisher_FP = full(Fisher(HH_1, HH_2 ));

toc

writematrix([COST_0; COST]'	, ['COST_',num2str(feedPress(1)),'.txt'])

writematrix(T0              , ['CONTROL_T_0_',num2str(feedPress(1)),'.txt'])
writematrix(F0     			, ['CONTROL_F_0_',num2str(feedPress(1)),'.txt'])

writematrix(HH_1   			, ['CONTROL_T_',num2str(feedPress(1)),'.txt'])
writematrix(HH_2   			, ['CONTROL_F_',num2str(feedPress(1)),'.txt'])

writematrix(Y_FP   			, ['FP_',num2str(feedPress(1)),'.txt'])
writematrix(Y_RBF  			, ['RBF_',num2str(feedPress(1)),'.txt'])
writematrix(Y_FP_0 			, ['FP_0_',num2str(feedPress(1)),'.txt'])
writematrix(Y_RBF_0			, ['RBF_0_',num2str(feedPress(1)),'.txt'])

writematrix(Fisher_FP	    , ['S_FP_',num2str(feedPress(1)),'.txt'])
writematrix(Fisher_RBF		, ['S_RBF_',num2str(feedPress(1)),'.txt'])

datetime
%}