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
dp                      = Parameters{5};
mi                      = Parameters{9};
rp                      = dp / 2;
lp2                     = (rp / 3)^2;

LabResults              = xlsread('dataset_2.xlsx');
N                       = 5;

Csolid_percentage_left  = linspace(0,1,20);
TT                      = linspace(30,40,3)+273;
PP                      = linspace(200,100,3);

FF                      = linspace(3.33,6.67,25)*1e-5;
RE = []; RHO = [];

MM = table2array(combinations(TT, PP));

indx = 1:2:2*size(MM,1);
indy = 2:2:2*size(MM,1);

tiledlayout(9,2)

for kk=1:size(MM,1)
    AA            =     MM(kk,:);
    T = AA(1); P = AA(2);

    Z             =     Compressibility(T, P,    Parameters);
    rho           =     rhoPB_Comp(     T, P, Z, Parameters);   
    %RHO           =     [RHO, rho];
    MU            =     Viscosity(T,rho);
    VELOCITY      =     Velocity(FF, rho, Parameters);
    RE            =     dp .* rho .* VELOCITY ./ MU .* 1.3;

    %RE            =     [RE, re];

    re_RBF = nan(numel(Csolid_percentage_left), numel(FF));
    re_FP  = nan(numel(Csolid_percentage_left), numel(FF));
    for ii=1:numel(Csolid_percentage_left)
        for jj = 1:numel(FF)
            
            x = Csolid_percentage_left(ii);
            y = RE(jj);

            Di            = Diffusion(y, FF(jj), Parameters) .* 1e-13;
            gamma         = Decay_Function_Coe(y, FF(jj), Parameters);
            Sat_coe       = Saturation_Concentration(x, gamma, Di);
            SOLID         = (1 - y) * 20.8391;
            

            re_RBF(ii,jj) = RBF_Function_3D_Single_Layer( x, y, rho./800, N, [])*1e-3;
            re_FB(ii,jj)  = (Sat_coe ./ mi ./ lp2)  .* ( SOLID );
        end
    end

    %figure(1)
    %subplot(size(MM,1),2,indx(kk))
    nexttile
    p = pcolor(RE, Csolid_percentage_left, re_RBF);
    shading interp; 
    hold on
    contour(RE, Csolid_percentage_left, re_RBF, 'k', 'LineWidth', 1, 'ShowText','on', 'LevelList', -1:0.01:0 );
    hold off
    title(['T = ',num2str(round(T-273)),' $^\circ C$, P =',num2str(round(P)),' bar,  $\rho_f$=',num2str(round(rho)),'~$kg/m^3$'])
    colorbar
    colormap jet
    %caxis([0 4]);
    xlabel('Re')
    ylabel(['$1-\frac{c_s}{c_{s0}}$'])
    axis tight
    set(gca,'FontSize',10)

    %{\
    %figure(2)
    %subplot(size(MM,1),2,indy(kk))
    nexttile
    p = pcolor(RE, Csolid_percentage_left, re_FB);
    shading interp; 
    hold on
    contour(RE, Csolid_percentage_left, re_FB, 'k', 'LineWidth', 1, 'ShowText','on', 'LevelList', -1:0.01:0 );
    hold off
    title(['T = ',num2str(round(T-273)),' $^\circ C$, P =',num2str(round(P)),' bar,  $\rho_f$=',num2str(round(rho)),'~$kg/m^3$'])
    colorbar
    colormap jet
    %caxis([0 4]);
    xlabel('Re')
    ylabel(['$1-\frac{c_s}{c_{s0}}$'])
    axis tight
    set(gca,'FontSize',10)

    %}

end

%}