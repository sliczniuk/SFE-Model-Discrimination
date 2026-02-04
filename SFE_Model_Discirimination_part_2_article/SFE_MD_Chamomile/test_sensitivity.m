startup;
delete(gcp('nocreate'));
% %p = Pushbullet(pushbullet_api);

%addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
addpath('\\home.org.aalto.fi\sliczno1\data\Documents\casadi-3.6.3-windows64-matlab2018b');
import casadi.*

excel_file = 'Chamomile_Di_Gamma_2.xls';
%rng(69)

%%
Parameters_table        = readtable('Parameters.csv') ;                     % Table with prameters
Parameters              = num2cell(Parameters_table{:,3});                  % Parameters within the model + (m_max), m_ratio, sigma

LabResults              = xlsread('dataset_2.xlsx');
%Q_RBF = readmatrix('Q_RBF.txt');
%Q_FP  = readmatrix('Q_FP.txt');

%% Create the solver
Iteration_max               = 15;                                         % Maximum number of iterations for optimzer
Time_max                    = 14;                                           % Maximum time of optimization in [h]

nlp_opts                    = struct;
%nlp_opts.ipopt.max_iter     = Iteration_max;
nlp_opts.ipopt.max_cpu_time = Time_max*3600;
nlp_opts.ipopt.hessian_approximation ='limited-memory';

%% Load paramters
m_total                 = 3.0;

% Bed geometry
before                  = 0.04;                                             % Precentage of length before which is empty
bed                     = 0.92;                                              % Percentage of length occupied by fixed bed

% Set time of the simulation
PreparationTime         = 0;
ExtractionTime          = 600;
timeStep                = 5;                                                % Minutes
OP_change_Time          = 15; 
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
Nx_RBF                  = (1+36)*(3 * nstages+3);                                % 3*Nstages(C_f, C_s, H) + P(t) + yield
Nx_FP                   = (1+6)*(3 * nstages+3);                                % 3*Nstages(C_f, C_s, H) + P(t) + yield
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
%{
OPT_solver                  = casadi.Opti();
ocp_opts                    = {'nlp_opts', nlp_opts};
OPT_solver.solver(             'ipopt'   , nlp_opts)

T0homog                 = OPT_solver.variable(numel(OP_change))';
                          OPT_solver.subject_to( 30+273 <= T0homog <= 40+273 );
    
Flow                    = OPT_solver.variable(numel(OP_change))';
                          OPT_solver.subject_to( 3.33 <= Flow <= 6.67 );

feedPress               = 150;               % MPa -> bar

%Pressure                = OPT_solver.variable(numel(OP_change_P))';
%                          OPT_solver.subject_to( 100 <= Pressure <= 200 );

T_0                     = 35+273;%feedTemp(1);   
feedTemp                = repmat(T0homog,OP_change_Time/timeStep,1);
feedTemp                = feedTemp(:)';
%T_0 = feedTemp(1)       ;
feedTemp                = [ feedTemp, T0homog(end)*ones(1,N_Time - numel(feedTemp)) ];   

feedPress               = feedPress * ones(1,numel(feedTemp)) + 0 ;     % Bars
%feedPress                = repmat(Pressure,OP_change_Time_P/timeStep,1);
%feedPress                = feedPress(:)';
%feedPress                = [ feedPress, Pressure(end)*ones(1,N_Time - numel(feedPress)) ];    

F_0                     = 5;
feedFlow                = repmat(Flow,OP_change_Time/timeStep,1);
feedFlow                = feedFlow(:)';
%F_0 = feedFlow(1) ;
feedFlow                = [ feedFlow, Flow(end)*ones(1,N_Time - numel(feedFlow)) ];    
%}
%T0homog                 = 35+273;                    % K
feedPress               = 150;                       % bar
Flow                    = 5 ;                        % kg/s
%feedTemp                = T0homog   * ones(1,length(Time_in_sec)) + 0 ;     % Kelvin
feedTemp                = linspace(30+273,40+273,N_Time);
T0homog                 = feedTemp(1)  ;

feedPress               = feedPress * ones(1,N_Time) + 0 ;     % Bars
    
%feedFlow                = Flow * ones(1,length(Time_in_sec));               % kg/s
feedFlow                = linspace(3.3,6.6, N_Time);

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
%[xx_R]                 = simulateSystem(F_RBF, [], x0_RBF, Parameters_init_time );
[xx_F]                 = simulateSystem(F_corr , [], x0_FP, Parameters_init_time );


%%
sigma = 3e-3;

Q_RBF = readmatrix('Q_RBF_1.txt');
Q_FP  = readmatrix('Q_FP_1.txt');

XX_FP    = xx_F(3*nstages+2:3*nstages+3:Nx_FP,:);
yy_FP    = XX_FP(1,:);
S_FP     = XX_FP(2:end,:);

%{
XX_RBF   = xx_R(3*nstages+2:3*nstages+3:Nx_RBF,:);
yy_RBF   = XX_RBF(1,:);
S_RBF    = XX_RBF(2:end,:);
FI_RBF   = 0*eye(N_Time+1) + (S_RBF' * pinv(Q_RBF./sigma) * S_RBF);
SIGMA_RBF = inv(FI_RBF);

%subplot(2,2,1);imagesc(FI_RBF);subplot(2,2,2);imagesc(FI_FP);
%subplot(2,2,3);imagesc(SIGMA_RBF);subplot(2,2,4);imagesc(SIGMA_FP);

%{\
residuals            = yy_RBF - yy_FP;

JJ_1                = trace(FI_FP * SIGMA_RBF + FI_RBF * SIGMA_FP - 2*eye(size(FI_FP)));
JJ_2                = residuals * [SIGMA_RBF + SIGMA_FP] * residuals';

%%
T0 = ( (40-30).*rand(1,numel(T0homog)) + 30 )+273;
F0 = ( (3.33-6.67).*rand(1,numel(Flow)) + 3.33 );

%{
tic
AA    = Function('AA', {[T0homog,Flow]}, {[JJ_1]} );
FI_R  = full(AA([T0, F0])); toc

tic
AA    = Function('AA', {[T0homog,Flow]}, {[JJ_2]} );
FI_F  = full(AA([T0, F0]));
toc

%subplot(2,1,1);pcolor((FI_R)); shading interp; colorbar
%subplot(2,1,2);pcolor((FI_F)); shading interp; colorbar

%%


%% Simulate system and extract yield and sensitivity data
X_RBF                   = MX(Nx_RBF,N_Time+1);
X_RBF(:,1)              = x0_RBF;

% Symbolic integration
for j=1:N_Time
    X_RBF(:,j+1)=F_RBF(X_RBF(:,j), [uu(j,:)'; Parameters_sym] );
end

X_FP                    = MX(Nx_FP,N_Time+1);
X_FP(:,1)               = x0_FP;

% Symbolic integration
for j=1:N_Time
    X_FP(:,j+1)=F_corr(X_FP(:,j), [uu(j,:)'; Parameters_sym] );
end

S_RBF  = X_RBF(X_RBF(2*152:152:37*152,N_Sample));
FI_RBF = S_RBF' * pinv(Q_RBF) * S_RBF;

S_FP  = X_FP(2*152:152:7*152,N_Sample);
FI_FP = S_FP' * pinv(Q_FP) * S_FP;

%[xout_corr]            = simulateSystem(F_corr, [], x0, Parameters_init_time );
yy_corr                = X_FP(3*nstages+2,N_Sample);
%sT_corr                = X_FP(6*nstages+4,N_Sample);
%sF_corr                = X_FP(9*nstages+6,N_Sample);

%[xout_RBF]             = simulateSystem(F_RBF , [], x0, Parameters_init_time );
yy_RBF                 = X_RBF(3*nstages+2,N_Sample);
%sT_RBF                 = X_RBF(6*nstages+4,N_Sample);
%sF_RBF                 = X_RBF(9*nstages+6,N_Sample);

residuals              = yy_RBF - yy_corr;

%%
%FI_C_T                 = sT_corr' * sT_corr ./0.01^2;
%FI_C_F                 = sF_corr' * sF_corr ./0.01^2;
%FI_R_T                 = sT_RBF'  * sT_RBF  ./0.01^2;
%FI_R_F                 = sF_RBF'  * sF_RBF  ./0.01^2;

%FI_C                   = FI_C_T   + FI_C_T;
%FI_R                   = FI_R_T   + FI_R_T;

%%
T0 = (35+273) * ones(1,numel(T0homog));
F0 = 5 * ones(1,numel(Flow));
%T0 = ( (40-30).*rand(1,numel(T0homog)) + 30 )+273;
%F0 = ( (3.33-6.67).*rand(1,numel(Flow)) + 3.33 );

%%
%FI_C_I               = tikhonov_inverse(FI_FP);
%FI_R_I               = tikhonov_inverse(FI_RBF);

%JJ_1                 = FI_FP * FI_R_I + FI_RBF * FI_C_I - 2*eye(size(FI_FP)) ;
%AA                   = Function('AA', {[T0homog,Flow]}, {JJ_1} );
%FI_JJ_1              = full(AA([T0, F0]));
%trace(FI_JJ_1)

%JJ_2                 = residuals * [FI_R_I + FI_C_I] * residuals';
tic
AA                   = Function('AA', {[T0homog,Flow]}, {[xx_R]} );
FI_JJ_2              = full(AA([T0, F0]));
toc
%}
%% Solve the optimization problem
OPT_solver.minimize(-(JJ_1 + JJ_2));
OPT_solver.set_initial([T0homog, Flow], [T0, F0] );

%%
tic
try
    sol  = OPT_solver.solve();
    KOUT = full(sol.value([T0homog, Flow])) ;
catch
    KOUT = OPT_solver.debug.value([T0homog, Flow]);
end
toc

%%
AA        = Function('AA', {[T0homog,Flow]}, {xx_F([3*nstages+2],:)} );
res_FP_0  = full(AA([T0, F0]));
res_FP    = full(AA(KOUT));

AA        = Function('AA', {[T0homog,Flow]}, {xx_R([3*nstages+2],:)} );
res_RBF_0 = full(AA([T0, F0]));
res_RBF   = full(AA(KOUT));

%%
subplot(3,1,1); 
hold on; 
plot(Time, res_FP_0,'r--'); 
plot(Time, res_FP,'r'); 
plot(Time, res_RBF_0,'b--'); 
plot(Time, res_RBF,'b'); 
hold off; 

legend('Inital FR', 'Final FP', 'Inital RBF', 'Final RBF', 'box', 'off')
subplot(3,1,2)
hold on;
stairs(OP_change, T0-273)
stairs(OP_change, KOUT(1:numel(OP_change))-273)
hold off

subplot(3,1,3)
hold on;
stairs(OP_change, F0)
stairs(OP_change, KOUT(numel(OP_change)+1:end))
hold off
legend('Inital', 'Final', 'box', 'off')
%}