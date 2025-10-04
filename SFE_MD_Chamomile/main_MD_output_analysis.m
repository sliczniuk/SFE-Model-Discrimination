startup;
delete(gcp('nocreate'));
% %p = Pushbullet(pushbullet_api);

%addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
addpath('\\home.org.aalto.fi\sliczno1\data\Documents\casadi-3.6.3-windows64-matlab2018b');
import casadi.*


%% Load data
Parameters_table        = readtable('Parameters.csv') ;                     % Table with prameters
Parameters              = num2cell(Parameters_table{:,3});                  % Parameters within the model + (m_max), m_ratio, sigma
r                       = Parameters{3};                                    % Radius of the extractor  [m]
epsi                    = Parameters{4};                                    % Fullness [-]
dp                      = Parameters{5};                                    % Paritcle diameter
L                       = Parameters{6};                                    % Total length of the extractor [m]

V                       = L  * pi * r^2;                                    % Total volume of the extractor [m3]
A                       = pi *      r^2;                                    % Extractor cross-section

%--------------------------------------------------------------------

N_exp                   = 4;
COLORS                  = ['b','r','k','m','g'];
GROUP    = []; Yield      = []; COST = [];
%PP                      = [100, 125, 150, 175, 200];
PP                      = [100];

figure(3)
tiledlayout(numel(PP),2)

%hold on
for ii = 1:numel(PP)
    PRES = PP(ii);
    AA       = readmatrix(['COST_',num2str(PRES),'.txt']);
    COST     = [COST; AA];
    ind      = find( AA(:,2) == min(AA(:,2)) );

    % Plots of controls
    BB_T       = readmatrix(['CONTROL_T_',num2str(PRES),'.txt']);
    BB_T       = reshape(BB_T(:), [], N_exp);
    BB_F       = readmatrix(['CONTROL_F_',num2str(PRES),'.txt']);
    BB_F       = reshape(BB_F(:), [], N_exp);

    TempCont = BB_T(:,ind);
    FlowCont = BB_F(:,ind) * 1e-5;
    Time     = linspace(0,600,length(TempCont));

    CC = readmatrix(['FP_',num2str(PRES),'.txt']);
    CC = reshape(CC(:),[],N_exp);
    Y_FP     = CC(:,ind);
    
    DD = readmatrix(['RBF_',num2str(PRES),'.txt']);
    DD = reshape(DD(:),[],N_exp);
    Y_RBF     = DD(:,ind);

    GROUP    = [GROUP; repmat(['P = ', num2str(PRES),' bar'], N_exp,1) ];

    sigma = 0.03;
    N_Sample = 60; 

    S_FP  = readmatrix(['S_FP_',num2str(PRES),'.txt']);
    S_FP  = reshape(S_FP(:),[],N_exp);
    S_FP  = reshape(S_FP(:,ind), 6, []); 

    FI_FP    = sigma*eye(length(N_Sample)) + (S_FP' * pinv(Q_FP./sigma) * S_FP);
    SIGMA_FP = pinv(FI_FP);

    S_RBF  = readmatrix(['S_RBF_',num2str(PRES),'.txt']);
    S_RBF  = reshape(S_RBF(:),[],N_exp);
    S_RBF  = reshape(S_RBF(:,ind), 36, []); 

    FI_RBF    = sigma*eye(length(N_Sample)) + (S_RBF' * pinv(Q_RBF./sigma) * S_RBF);
    SIGMA_RBF = pinv(FI_RBF);
    
    %{\
    %subplot(3,1,1)
    figure(1)
    hold on
    stairs(Time, TempCont-273, 'LineWidth', 2 ,'Color',COLORS(ii));
    hold off
    ylabel('$T^{in}~^\circ$C')
    xlabel('Time min')
    %legend('P = 100 bar','P = 125 bar','P = 150 bar','P = 175 bar','P = 200 bar', 'Location','best','NumColumns',5)
    %legend box off
    %axis square
    set(gca,'FontSize',16)
    hold off
    
    figure(2)
    %subplot(3,1,2)
    hold on
    stairs(Time, FlowCont, 'LineWidth', 2, 'Color',COLORS(ii));
    hold off
    ylabel('$F~kg/s$')
    xlabel('Time min')
    %axis square
    set(gca,'FontSize',16)
    hold off
    %}
    %{'
    %Yield = [Yield; Yield_Plot(PRES,TempCont(:, ind),FlowCont(:, ind))];

    Time     = linspace(0,600,size(Y_FP,1));

    figure(3)
    %subplot(5,1,ii)
    %nexttile
    hold on
    plot(Time, Y_RBF, 'LineWidth',3, 'Color', COLORS(ii), 'LineStyle',':');
    plot(Time, Y_FP, 'LineWidth',3, 'Color', COLORS(ii), 'LineStyle','-');
    %plot(Time, YY_RBF_0(:,ii), 'LineWidth',3, 'Color', 'b', 'LineStyle',':');
    %plot(Time, YY_FP_0( :,ii), 'LineWidth',3, 'Color', 'b', 'LineStyle','-');
    hold off
    xlabel('Time min')
    ylabel('y gram')
    ylim([0 3.2])
    %title(['P =',num2str(round(PRES)),' bar'])
    set(gca,'FontSize',16);
    hold off

    figure(4)
    %subplot(5,1,ii)
    %nexttile
    hold on
    plot(Time(1:end-1), diff(Y_RBF), 'LineWidth',3, 'Color', COLORS(ii),'LineStyle',':');
    plot(Time(1:end-1), diff(Y_FP), 'LineWidth',3, 'Color',  COLORS(ii),'LineStyle','-');
    %plot(Time(1:end-1), diff(YY_FP(:,ii)), 'LineWidth',3, 'Color', 'r','LineStyle','-');
    %plot(Time(1:end-1), diff(YY_RBF_0(:,ii)), 'LineWidth',3, 'Color', 'b','LineStyle',':');
    %plot(Time(1:end-1), diff(YY_FP_0(:,ii)), 'LineWidth',3, 'Color', 'b','LineStyle','-');
    %hold off
    xlabel('Time min')
    ylabel('$\frac{dy}{dt}$ gram/s')
    %ylim([0 0.13])
    %title(['P =',num2str(round(PRES)),' bar'])
    set(gca,'FontSize',16);
    hold off
%}

end
%{\
%figure(1); legend({'100 bar','125 bar','150 bar','175 bar','200 bar'}, 'Location', 'northwest', 'NumColumns',5); legend('boxoff')
%exportgraphics(figure(1), ['Profile_T_MD.png'], "Resolution",300); 

%figure(2); legend({'100 bar','125 bar','150 bar','175 bar','200 bar'}, 'Location', 'northwest', 'NumColumns',5); legend('boxoff')
%exportgraphics(figure(2), ['Profile_F_MD.png'], "Resolution",300); 

%figure(3); legend({'100 bar','100 bar','125 bar','125 bar','150 bar', '150 bar','175 bar','175 bar','200 bar: RBF','200 bar: FP'}, 'Location', 'northwest', 'NumColumns',5, 'FontSize', 7); legend('boxoff')
%figure(3); legend({'100 bar','100 bar','125 bar','125 bar','150 bar: RBF', '150 bar: FP'}, 'Location', 'northwest', 'NumColumns',5); legend('boxoff')
%exportgraphics(figure(3), ['yield_MD.png'], "Resolution",300); 

%figure(4); legend({'100 bar','100 bar','125 bar','125 bar','150 bar', '150 bar','175 bar','175 bar','200 bar: RBF','200 bar: FP'}, 'Location', 'northwest', 'NumColumns',5, 'FontSize', 7); legend('boxoff')
%figure(4); legend({'100 bar','100 bar','125 bar','125 bar','150 bar: RBF', '150 bar: FP'}, 'Location', 'northwest', 'NumColumns',5); legend('boxoff')
%exportgraphics(figure(4), ['diff_yield_MD.png'], "Resolution",300); 
%}
%%
%{
figure(5)
s = scatterhist(-COST(:,2), -COST(:,1), 'Group', GROUP, 'Kernel', 'on', 'LineWidth',3, 'MarkerSize',6, 'Color',COLORS(1:numel(PP)), 'LineStyle',{'-','-','-','-','-','-'} );
%s = scatter(COST_F, COST_I, 'Group', GROUP, 'LineWidth',3, 'MarkerSize',6, 'Color',COLORS, 'LineStyle',{'-','-','-','-','-','-'} );
s(1).Children(5).MarkerFaceColor = 'b';
s(1).Children(4).MarkerFaceColor = 'r';
s(1).Children(3).MarkerFaceColor = 'k';
s(1).Children(2).MarkerFaceColor = 'm';
s(1).Children(1).MarkerFaceColor = 'g';

legend box off

ylabel('Inital value of $\ln j_D$ [-]');
xlabel('Final value of $\ln j_D$ [-]');
set(gca,'FontSize',16);

%exportgraphics(figure(5), ['scatter_MD.png'], "Resolution",300);
%close all

%}