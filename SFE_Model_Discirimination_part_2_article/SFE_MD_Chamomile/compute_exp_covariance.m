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
Parameters_sym          = MX(cell2mat(Parameters));
LabResults              = xlsread('dataset_2.xlsx');

%% Load paramters
m_total                 = 3.0;

% Bed geometry
before                  = 0.04;                                             % Precentage of length before which is empty
bed                     = 0.92;                                              % Percentage of length occupied by fixed bed

% Set time of the simulation
PreparationTime         = 0;
ExtractionTime          = 600;
timeStep                = 1;                                                % Minutes
Sample_Time             = 10;

simulationTime          = PreparationTime + ExtractionTime;

timeStep_in_sec         = timeStep * 60;                                    % Seconds
Time_in_sec             = (timeStep:timeStep:simulationTime)*60;            % Seconds
Time                    = [0 Time_in_sec/60];                               % Minutes

N_Time                  = length(Time_in_sec);

SAMPLE                  = Sample_Time:Sample_Time:ExtractionTime;

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
X_FP_1                  = nan(1,  7, N_Time+1);
X_RBF_1                 = nan(1, 37, N_Time+1);

X_FP_2                  = nan(1,  7, N_Time+1);
X_RBF_2                 = nan(1, 37, N_Time+1);

parfor which_dataset=1:12
    
    T0homog                 = LabResults(2,which_dataset+1);                    % K
    feedPress               = LabResults(3,which_dataset+1) * 10;               % MPa -> bar
    Flow                    = LabResults(4,which_dataset+1) ;                   % kg/s
    
    Z                       = Compressibility( T0homog, feedPress,         Parameters );
    rho                     = rhoPB_Comp(      T0homog, feedPress, Z,      Parameters );
        
    enthalpy_rho            = rho.*SpecificEnthalpy(T0homog, feedPress, Z, rho, Parameters );
        
    feedTemp                = T0homog   * ones(1,length(Time_in_sec)) + 0 ;     % Kelvin
        
    feedPress               = feedPress * ones(1,length(Time_in_sec)) + 0 ;     % Bars
        
    feedFlow                = Flow * ones(1,length(Time_in_sec));               % kg/s
    
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
                                0;
                                zeros(Nx_RBF - (3*nstages+3), 1) ;
                                ];
    
    x0_FP                   = [ C0fluid'                         ;
                                C0solid         * bed_mask       ;
                                enthalpy_rho    * ones(nstages,1);
                                feedPress(1)                     ;
                                0;
                                0;
                                zeros(Nx_FP  - (3*nstages+3), 1) ;
                                ];
    
    Parameters_init_time   = [uu repmat(cell2mat(Parameters),1,N_Time)'];
    [xx_R]                 = simulateSystem(F_RBF, [], x0_RBF, Parameters_init_time );
    [xx_F]                 = simulateSystem(F_corr , [], x0_FP, Parameters_init_time );

    X_FP_1(which_dataset,:,:)= round(xx_F(1*(3*nstages+2):(3*nstages+3):Nx_FP, :),10);
    X_RBF_1(which_dataset,:,:)= round(xx_R(1*(3*nstages+2):(3*nstages+3):Nx_RBF, :),10);

    X_FP_2(which_dataset,:,:)= round(xx_F(1*(3*nstages+3):(3*nstages+3):Nx_FP, :),10);
    X_RBF_2(which_dataset,:,:)= round(xx_R(1*(3*nstages+3):(3*nstages+3):Nx_RBF, :),10);

end
%%
Q_RBF_1 = 0; Q_FP_1 = 0; Q_RBF_2 = 0; Q_FP_2 = 0; 
for ii=1:N_Time+1
    AA_FP_1  = X_FP_1(:, 2:end,ii) ; 
    AA_FP_2  = X_FP_2(:, 2:end,ii) ; 
    AA_RBF_1 = X_RBF_1(:,2:end,ii) ;
    AA_RBF_2 = X_RBF_2(:,2:end,ii) ;

    Q_FP_1   = Q_FP_1  + (AA_FP_1' *  AA_FP_1);
    Q_RBF_1  = Q_RBF_1 + (AA_RBF_1' *  AA_RBF_1);

    Q_FP_2   = Q_FP_2  + (AA_FP_2' *  AA_FP_2);
    Q_RBF_2  = Q_RBF_2 + (AA_RBF_2' *  AA_RBF_2);
    
end
%%
figure()
subplot(2,2,1)
pcolor(pinv(Q_FP_1)); colorbar; shading interp
subplot(2,2,2)
pcolor(pinv(Q_RBF_1)); colorbar; shading interp
subplot(2,2,3)
pcolor(pinv(Q_FP_2)); colorbar; shading interp
subplot(2,2,4)
pcolor(pinv(Q_RBF_2)); colorbar; shading interp

%%
%{\
writematrix(Q_RBF_1, 'Q_RBF_1.txt')
writematrix(Q_RBF_2, 'Q_RBF_2.txt')
writematrix(Q_FP_1 , 'Q_FP_1.txt')
writematrix(Q_FP_2 , 'Q_FP_2.txt')
%%}