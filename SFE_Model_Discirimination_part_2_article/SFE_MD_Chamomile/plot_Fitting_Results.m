startup;
delete(gcp('nocreate'));

%addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
addpath('\\home.org.aalto.fi\sliczno1\data\Documents\casadi-3.6.3-windows64-matlab2018b');
import casadi.*

%%
AA = xlsread('Regression.xlsx');
DI = [0.71383013	1.076801997	2.179470155	2.475532632	1.390707877	1.336111172	1.882954204	2.457886055	0.564935512	1.542106938	0.835725102	0.87349666];
GG = [4.229739602	3.091520556	2.359538225	1.132795818	2.204975712	2.739220425	1.868538631	1.69935869	3.452308202	1.995905641	3.012676539	2.596460037];
RE = [0.4632, 0.3783, 0.3029, 0.2619, 0.3579, 0.3140, 0.2635, 0.2323, 0.1787, 0.1160, 0.1889, 0.1512];
TT = [313, 313, 313, 313, 303, 303, 303, 303, 303, 303, 313, 313];
Tr = TT ./ 304;
RHO= [630, 691, 793, 840, 772, 802, 856, 891, 772, 891, 691, 793];

%%
Parameters_table        = readtable('Parameters.csv') ;                     % Table with prameters
Parameters              = num2cell(Parameters_table{:,3});                  % Parameters within the model + (m_max), m_ratio, sigma

LabResults              = xlsread('dataset_2.xlsx');

which_k                 = [44, 46];

%% Load paramters
m_total                 = 3.0;

% Bed geometry
before                  = 0.04;                                             % Precentage of length before which is empty
bed                     = 0.92;                                             % Percentage of length occupied by fixed bed

% Set time of the simulation
PreparationTime         = 0;
ExtractionTime          = 600;
timeStep                = 10;                                               % Minutes

simulationTime          = PreparationTime + ExtractionTime;

timeStep_in_sec         = timeStep * 60;                                    % Seconds
Time_in_sec             = (timeStep:timeStep:simulationTime)*60;            % Seconds
Time                    = [0 Time_in_sec/60];                               % Minutes

N_Time                  = length(Time_in_sec);

SAMPLE                  = LabResults(6:19,1);

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

%% Set Integrator
f                       = @(x, u) modelSFE(x, u, bed_mask, timeStep_in_sec);
xdot                    = modelSFE(x, u, bed_mask, timeStep_in_sec);

% Integrator
F                       = buildIntegrator(f, [Nx,Nu] , timeStep_in_sec);

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

for jj = 9:12
    which_dataset           = jj;

    data_org                = LabResults(6:19,which_dataset+1)';
    data_diff               = diff(data_org);

    %% Set operating conditions
    T0homog                 = LabResults(2,which_dataset+1);                    % K
    feedPress               = LabResults(3,which_dataset+1) * 10;               % MPa -> bar
    Flow                    = LabResults(4,which_dataset+1) * 1e-5 ;            % kg/s

    Z                       = Compressibility( T0homog, feedPress,         Parameters );
    rho                     = rhoPB_Comp(      T0homog, feedPress, Z,      Parameters );

    enthalpy_rho            = rho.*SpecificEnthalpy(T0homog, feedPress, Z, rho, Parameters );

    feedTemp                = T0homog   * ones(1,length(Time_in_sec)) + 0 ;     % Kelvin

    feedPress               = feedPress * ones(1,length(Time_in_sec)) + 0 ;     % Bars

    feedFlow                = Flow * ones(1,length(Time_in_sec));               % kg/s

    uu                      = [feedTemp', feedPress', feedFlow'];

    % Initial conditions
    x0                      = [ C0fluid'                         ;
                                C0solid         * bed_mask       ;
                                enthalpy_rho    * ones(nstages,1);
                                feedPress(1)                     ;
                                0                                ;
                                ];

    %KOUT = DATA_K_OUT(:,ii);
    %KOUT = [ 0.7, 3.8 ];
    D =  0.190 -  8.188 * RE(jj) + 0.620  * feedFlow(1) * 10^5;
    G =  3.158 + 11.922 * RE(jj) - 0.6868 * feedFlow(1) * 10^5;
    %KOUT = [ DI(jj) , GG(jj) ];
    KOUT = [ D , G ];
    Parameters_opt = Parameters;
    for i=1:numel(which_k)
        Parameters_opt{which_k(i)}  = KOUT(i);
    end

    Parameters_init_time   = [uu repmat(cell2mat(Parameters_opt),1,N_Time)'];
    [xx_0]                 = simulateSystem(F, [], x0, Parameters_init_time );

    hold on; 
    plot(Time,xx_0(end,:),'LineWidth',2, 'DisplayName',[num2str(round(T0homog-273)),'$[^\circ C]$, ',num2str(feedPress(1)),'[bar]']); 
    plot(SAMPLE, data_org,'ko','LineWidth',2,'HandleVisibility','off'); 
    hold off
    
    xlabel('t~[min]')
    ylabel('$y~[g]$')

    set(gca,'FontSize',12)
end

ylim([0 3])

legend1 = legend('show');
set(legend1,...
    'box','off',...
    'Position',[-0.0100674570863917 0.938730158389562 1.01721031422925 0.0476190464837211],...
    'Orientation','horizontal',...
    'FontSize',10);

exportgraphics(figure(1), ['Fit_Di_Gamma_9_12_correlation.png'], "Resolution",300);
close all