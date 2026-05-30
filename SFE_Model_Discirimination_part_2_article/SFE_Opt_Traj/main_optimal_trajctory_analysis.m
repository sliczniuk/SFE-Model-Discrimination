startup;

COLOR = [
    0.00 0.45 0.74
    0.85 0.33 0.10
    0.93 0.69 0.13
    0.49 0.18 0.56
    0.47 0.67 0.19
    0.00 0.00 0.00
];

set(gca,'ColorOrder',COLOR,'NextPlot','replacechildren')

PRESS = {100, 130, 150, 175, 200, 'MuliPrees'};
Time = 0:10:600;

for ii = 1:numel(PRESS)
    
    if ii == 6
        load( "multi_pressure_results_smooth.mat");
        legend_name = 'Multi pressure';
    else
        load( [num2str(PRESS(ii)+"_results_smooth.mat")]);
        legend_name = num2str(cell2mat(PRESS(ii)))+" bar";
    end
    

    RES = cell2mat(results);
    
    j_f = vertcat(RES.j);
    j_i = vertcat(RES.j_initial);
    
    figure(1)
    hold on
    plot(j_f, j_i,'o', MarkerEdgeColor=COLOR(ii, :), MarkerFaceColor=COLOR(ii, :), MarkerSize=8, DisplayName=legend_name )
    hold off
    
    indx_best = find(max(j_f) == j_f);
    
    figure(2)
    feedTemp       = RES(indx_best).feedTemp;
    feedTemp_plot  = [feedTemp, feedTemp(end)]-273;
    hold on
    stairs(Time, feedTemp_plot, 'Color', COLOR(ii, :), 'LineWidth',2) %, DisplayName=[num2str(PRESS(ii))+" bar"])
    hold off
    
    figure(3)
    feedFlow      = RES(indx_best).feedFlow;
    feedFlow_plot = [feedFlow, feedFlow(end)];
    hold on
    stairs(Time, feedFlow_plot, 'Color', COLOR(ii, :), 'LineWidth',2) %, DisplayName=[num2str(PRESS(ii))+" bar"])
    hold off

    figure(4)
    if ii == 6
        feedPress_plot = RES(indx_best).feedPress;
        feedPress_plot = [feedPress_plot, feedPress_plot(end)];
    else
        feedPress_plot = cell2mat(PRESS(ii)) * ones(1, numel(Time));
    end

    hold on
    stairs(Time, feedPress_plot, 'Color', COLOR(ii, :), 'LineWidth',2) %, DisplayName=[num2str(PRESS(ii))+" bar"])
    hold off

    %out = trajectory(feedTemp, feedFlow, PRESS(ii));
    %{\
    if ii == 6
        Y_P  = RES(indx_best).Y_cum_P;
        Y_L  = RES(indx_best).Y_cum_L;
        figure(5)
        hold on
        plot(Time, Y_P, 'Color', COLOR(ii, :), 'LineWidth',2) %, DisplayName=[num2str(PRESS(ii))+" bar, P model"])
        plot(Time, Y_L, 'Color', COLOR(ii, :), 'LineStyle', '-.', 'LineWidth',2) %, DisplayName=[num2str(PRESS(ii))+" bar, L model"])
        hold off
    else
        out = trajectory(feedTemp, feedFlow, cell2mat(PRESS(ii)) );
        figure(5)
        hold on
        plot(out.Time, out.Y_P, 'Color', COLOR(ii, :), 'LineWidth',2) %, DisplayName=[num2str(PRESS(ii))+" bar, P model"])
        plot(out.Time, out.Y_L, 'Color', COLOR(ii, :), 'LineStyle', '-.', 'LineWidth',2) %, DisplayName=[num2str(PRESS(ii))+" bar, L model"])
        hold off
    end
    %}
end

figure(1); legend box off; legend(Location="northoutside", NumColumns=3); 
fontsize(gcf, 16, "points"); xlabel('Final value of the cost function [-]'); ylabel('Inital value of the cost function [-]');
print(figure(1),'OptTraj_scatter_smooth.png','-dpng', '-r500'); 

figure(2); %legend box off; legend(Location="northoutside", NumColumns=numel(PRESS))
fontsize(gcf, 16, "points"); xlabel('Time min'); ylabel('T $^\circ$C');
print(figure(2),'OptTraj_temp_smooth.png','-dpng', '-r500'); 

figure(3); %legend box off; legend(Location="northoutside", NumColumns=numel(PRESS))
fontsize(gcf, 16, "points"); xlabel('Time min'); ylabel('F kg/s');
print(figure(3),'OptTraj_flow_smooth.png','-dpng', '-r500'); 

figure(4); %legend box off; legend(Location="northoutside", NumColumns=numel(PRESS))
fontsize(gcf, 16, "points"); xlabel('Time min'); ylabel('P bar');
print(figure(4),'OptTraj_press_smooth.png','-dpng', '-r500'); 

figure(5); %fontsize(gcf, 16, "points"); legend box off; legend(Location="northoutside", NumColumns=2)
pbaspect([2 1 1])
fontsize(gcf, 16, "points"); xlabel('Time min'); ylabel('Yield gram');
print(figure(5),'OptTraj_yield_smooth.png','-dpng', '-r500');

close all

%{
%% Figure 6 – CO2 density map (pcolor) with multi-pressure optimal trajectory
% Load Parameters for the Peng-Robinson EOS
Parameters_table = readtable('Parameters.csv');
Parameters       = num2cell(Parameters_table{:,3});

% Density grid: T in [30 40] °C → [303 313] K, P in [100 200] bar
T_vec = linspace(303, 313, 300);   % [K]
P_vec = linspace(100, 200, 300);   % [bar]
[T_grid, P_grid] = meshgrid(T_vec, P_vec);

Z_grid   = Compressibility(T_grid, P_grid, Parameters);
rho_grid = rhoPB_Comp(T_grid, P_grid, Z_grid, Parameters);   % [kg/m³]

% Load multi-pressure result and extract optimal trajectory
load("multi_pressure_results.mat");
RES_mp  = cell2mat(results);
j_f_mp  = vertcat(RES_mp.j);
indx_mp = find(max(j_f_mp) == j_f_mp, 1);

feedTemp_mp  = RES_mp(indx_mp).feedTemp;    % [K], 1×N
feedPress_mp = RES_mp(indx_mp).feedPress;   % [bar], 1×N

% Time vector for colour-coding (one colour per control step)
N_steps    = numel(feedTemp_mp);
time_steps = linspace(0, 600, N_steps);     % [min]

figure(6);
pcolor(T_grid - 273.15, P_grid, rho_grid);
shading interp
cb = colorbar;
cb.Label.String = 'CO_2 density [kg/m^3]';
cb.Label.FontSize = 16;
colormap(turbo)
hold on

% Trajectory line
plot(feedTemp_mp - 273.15, feedPress_mp, 'w-', 'LineWidth', 1.2);

%{
% Control-step points coloured by time (uses a second axes trick via scatter)
scatter(feedTemp_mp - 273.15, feedPress_mp, 50, time_steps, ...
    'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.6);
%clim([0 600]);
cb2 = colorbar('Location', 'southoutside');
cb2.Label.String = 'Time [min]';
cb2.Label.FontSize = 14;
colormap(gca, parula);
%}

% Mark start (square) and end (triangle)
plot(feedTemp_mp(1)   - 273.15, feedPress_mp(1),   'ws', ...
    'MarkerFaceColor', [0.2 0.8 0.2], 'MarkerSize', 10, 'LineWidth', 1.2);
plot(feedTemp_mp(end) - 273.15, feedPress_mp(end), 'w^', ...
    'MarkerFaceColor', [0.9 0.2 0.2], 'MarkerSize', 10, 'LineWidth', 1.2);

hold off
fontsize(gcf, 16, "points");
xlabel('Temperature [°C]');
ylabel('Pressure [bar]');
legend({'Trajectory', '', 'Start', 'End'}, 'Location', 'northeast', 'Box', 'off');
%print(figure(6), 'OptTraj_density_map.png', '-dpng', '-r500');

%close all
%}