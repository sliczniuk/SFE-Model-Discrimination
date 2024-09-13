startup;
delete(gcp('nocreate'));
% %p = Pushbullet(pushbullet_api);

%addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
addpath('\\home.org.aalto.fi\sliczno1\data\Documents\casadi-3.6.3-windows64-matlab2018b');
import casadi.*

excel_file = 'Chamomile_Di_Gamma_2.xls';
rng(69)

%%
Parameters_table        = readtable('Parameters.csv') ;                     % Table with prameters
Parameters              = num2cell(Parameters_table{:,3});                  % Parameters within the model + (m_max), m_ratio, sigma

LabResults              = xlsread('dataset_2.xlsx');
N                       = 6;

%% Create the solver
Iteration_max               = 2;                                         % Maximum number of iterations for optimzer
Time_max                    = 15;                                           % Maximum time of optimization in [h]

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
timeStep                = 1;                                                % Minutes
OP_change_Time          = 10; 

simulationTime          = PreparationTime + ExtractionTime;

timeStep_in_sec         = timeStep * 60;                                    % Seconds
Time_in_sec             = (timeStep:timeStep:simulationTime)*60;            % Seconds
Time                    = [0 Time_in_sec/60];                               % Minutes

N_Time                  = length(Time_in_sec);

SAMPLE                  = LabResults(6:19,1);

OP_change               = OP_change_Time:OP_change_Time:ExtractionTime;

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
Nx                      = 3 * nstages+2;                                    % 3*Nstages(C_f, C_s, H) + P(t) + yield
Nu                      = 3 + numel( Parameters );                          % T_in, P, F + numel(Parameters)

%% Extractor geometry
r                       = Parameters{3};                                    % Radius of the extractor  [m]
epsi                    = Parameters{4};                                    % Fullness [-]
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
x                       = MX.sym('x', Nx);
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
f_RBF                   = @(x, u) modelSFE_RBF(x, u, bed_mask, timeStep_in_sec, N);
% Integrator
F_RBF                   = buildIntegrator(f_RBF, [Nx,Nu] , timeStep_in_sec);

%% Set Integrator - FP
f_FP                    = @(x, u) modelSFE_Corr(x, u, bed_mask, timeStep_in_sec);
% Integrator
F_FP                    = buildIntegrator(f_FP, [Nx,Nu] , timeStep_in_sec);

%%
Parameters_sym          = MX(cell2mat(Parameters));

% Set operating conditions

OPT_solver                  = casadi.Opti();
ocp_opts                    = {'nlp_opts', nlp_opts};
OPT_solver.solver(             'ipopt'   , nlp_opts)

T0homog                 = OPT_solver.variable(numel(OP_change))';
                          OPT_solver.subject_to( 30+273 <= T0homog <= 40+273 );
    
Flow                    = OPT_solver.variable(numel(OP_change))';
                          OPT_solver.subject_to( 3.33 <= Flow <= 6.67 );

feedPress               = 150;               % MPa -> bar

feedTemp                = repmat(T0homog,OP_change_Time/timeStep,1);
feedTemp                = feedTemp(:)';
feedTemp                = [ feedTemp, T0homog(end)*ones(1,N_Time - numel(feedTemp)) ];    
T_0                     = feedTemp(1);   

feedPress               = feedPress * ones(1,length(Time_in_sec)) + 0 ;     % Bars

feedFlow                = repmat(Flow,OP_change_Time/timeStep,1) * 1e-5;
feedFlow                = feedFlow(:)';
feedFlow                = [ feedFlow, Flow(end)*ones(1,N_Time - numel(feedFlow)) ];    

uu                      = [feedTemp', feedPress', feedFlow'];

Z                       = Compressibility( T_0, feedPress(1),         Parameters );
rho                     = rhoPB_Comp(      T_0, feedPress(1), Z,      Parameters );
enthalpy_rho            = rho.*SpecificEnthalpy(T_0, feedPress(1), Z, rho, Parameters ) ;

% Initial conditions
x0                      = [ C0fluid'                         ;
                            C0solid         * bed_mask       ;
                            enthalpy_rho    * ones(nstages,1);
                            feedPress(1)                     ;
                            0                                ;
                            ];

% Set the inital simulation and plot it against the corresponding dataset
%Parameters_init_time   = [uu repmat(cell2mat(Parameters),1,N_Time)'];
%[xx_RBF_0]             = simulateSystem(F_RBF, [], x0, Parameters_init_time );
%[xx_FP_0]              = simulateSystem(F_FP , [], x0, Parameters_init_time );

X_RBF                   = MX(Nx,N_Time+1);
X_RBF(:,1)              = x0;

% Symbolic integration
for j=1:N_Time
    X_RBF(:,j+1)=F_RBF(X_RBF(:,j), [uu(j,:)'; Parameters_sym] );
end

X_FP                    = MX(Nx,N_Time+1);
X_FP(:,1)               = x0;

% Symbolic integration
for j=1:N_Time
    X_FP(:,j+1)=F_FP(X_FP(:,j), [uu(j,:)'; Parameters_sym] );
end

%% Find the measurment from the simulation
Yield_estimate_RBF     = X_RBF(Nx,N_Sample);
Yield_estimate_FP      = X_FP(Nx,N_Sample);

residual = Yield_estimate_FP - Yield_estimate_RBF;

JJ = - residual * residual';

%% Defin intial guesses
%T0 = linspace(30,40,numel(T0homog))+273;
T0 = ( (40-30).*rand(1,numel(T0homog)) + 30 )+273;
%F0 = linspace(3.33,6.67,numel(Flow));
F0 = ( (6.67-3.33).*rand(1,numel(Flow)) + 3.33 ) ;
%P0 = ( (200 -100) .*rand(1,numel(Press)) + 100  ) ;

%% Solve the optimization problem
OPT_solver.minimize(JJ);
OPT_solver.set_initial([T0homog, Flow], [T0, F0] );

%%
try
    sol  = OPT_solver.solve();
    KOUT = full(sol.value([T0homog, Flow])) 
catch
    KOUT = OPT_solver.debug.value([T0homog, Flow])
end

figure()
hold on
stairs([0 OP_change],[KOUT(1:numel(OP_change)) KOUT(numel(OP_change))]-273, 'LineWidth', 2)
stairs([0 OP_change],[T0 T0(end)]-273, 'LineWidth', 2)
hold off
ylabel('Temperture C')
xlabel('Time min')
%legend('Optimized solution','Inital guess', 'Location','Best', 'box', 'off')
%set(gcf,'PaperOrientation','landscape'); print(figure(1),['2.pdf'],'-dpdf','-bestfit')
%exportgraphics(figure(1), ['2.png'], "Resolution",300);
%close all

figure()
hold on
stairs([0 OP_change],[KOUT(numel(OP_change)+1:end) KOUT(end)]* 1e-5, 'LineWidth', 2)
stairs([0 OP_change],[F0 F0(end)]* 1e-5, 'LineWidth', 2)
hold off
ylabel('Mass flow rate kg/s')
xlabel('Time min')

%Plotting
%figure(which_dataset)
%hold on
%title(['Dataset: ',num2str(which_dataset)])
%plot(SAMPLE, data_org,     ['o' ,COLORS(ii)], 'LineWidth',2, 'DisplayName',[num2str(round(T0homog-273)),'$^\circ C$, ',num2str(feedPress(1)),' bar'] )
%plot(Time, xx_FP_0(end,:), ['--',COLORS(ii)], 'LineWidth',2, 'HandleVisibility','off' )
%plot(Time, xx_RBF_0(end,:),[':' ,COLORS(ii)], 'LineWidth',2, 'HandleVisibility','off' )
