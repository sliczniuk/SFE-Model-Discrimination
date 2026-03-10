startup;

COLOR = ['k', 'r', 'b', 'g', 'm'];
PRESS = [100, 130, 150, 175, 200];

for ii = 1:numel(PRESS)
    if ii == 5
        load( [num2str(PRESS(ii)+"_bar_results.mat")]);
    else
        load( [num2str(PRESS(ii)+"_bar_results_extended.mat")]);
    end
    RES = cell2mat(results);
    
    j_f = vertcat(RES.j);
    j_i = vertcat(RES.j_initial);
    
    figure(1)
    hold on
    plot(j_f, j_i,'o', MarkerEdgeColor=COLOR(ii), MarkerFaceColor=COLOR(ii), MarkerSize=8, DisplayName=[num2str(PRESS(ii))+" bar"] )
    hold off
    
    indx_best = find(max(j_f) == j_f);
    
    figure(2)
    feedTemp  = RES(indx_best).feedTemp;
    hold on
    stairs(feedTemp-273, COLOR(ii), 'LineWidth',2, DisplayName=[num2str(PRESS(ii))+" bar"])
    hold off
    
    figure(3)
    feedFlow  = RES(indx_best).feedFlow;
    hold on
    stairs(feedFlow, COLOR(ii), 'LineWidth',2, DisplayName=[num2str(PRESS(ii))+" bar"])
    hold off

    out = trajectory(feedTemp, feedFlow, PRESS(ii));
    figure(4)
    hold on
    plot(out.Time, out.Y_P, COLOR(ii), 'LineWidth',2, DisplayName=[num2str(PRESS(ii))+" bar, P model"])
    plot(out.Time, out.Y_L, COLOR(ii), 'LineStyle', '-.', 'LineWidth',2, DisplayName=[num2str(PRESS(ii))+" bar, L model"])
    hold off

end

figure(1); legend box off; legend(Location="northoutside", NumColumns=numel(PRESS)); 
fontsize(gcf, 16, "points"); xlabel('Final value of the cost function [-]'); ylabel('Inital value of the cost function [-]');
%print(figure(1),['OptTraj_scatter.png'],'-dpng', '-r500'); 

figure(2); legend box off; legend(Location="northoutside", NumColumns=numel(PRESS))
fontsize(gcf, 16, "points"); xlabel('Time min'); ylabel('Temperature $^\circ$ C');
%print(figure(2),['OptTraj_temp.png'],'-dpng', '-r500'); 

figure(3); legend box off; legend(Location="northoutside", NumColumns=numel(PRESS))
fontsize(gcf, 16, "points"); xlabel('Time min'); ylabel('Mass flow rate [kg/s]');
%print(figure(3),['OptTraj_flow.png'],'-dpng', '-r500'); 

figure(4); fontsize(gcf, 16, "points"); legend box off; legend(Location="northoutside", NumColumns=2)
fontsize(gcf, 16, "points"); xlabel('Time min'); ylabel('Yield g');
%print(figure(4),['OptTraj_yield.png'],'-dpng', '-r500'); 