startup;
delete(gcp('nocreate'));
% %p = Pushbullet(pushbullet_api);

%addpath('C:\Dev\casadi-3.6.3-windows64-matlab2018b');
addpath('\\home.org.aalto.fi\sliczno1\data\Documents\casadi-3.6.3-windows64-matlab2018b');
import casadi.*

%% 
DI = [0.71383013	1.076801997	2.179470155	2.475532632	1.390707877	1.336111172	1.882954204	2.457886055	0.564935512	1.542106938	0.835725102	0.87349666];
GG = [4.229739602	3.091520556	2.359538225	1.132795818	2.204975712	2.739220425	1.868538631	1.69935869	3.452308202	1.995905641	3.012676539	2.596460037];
RE = [0.4632, 0.3783, 0.3029, 0.2619, 0.3579, 0.3140, 0.2635, 0.2323, 0.1787, 0.1160, 0.1889, 0.1512];
FF = [6.67, 6.67, 6.67, 6.67, 6.67, 6.67, 6.67, 6.67, 3.33, 3.33, 3.33, 3.33];

DI_fun = @(Re, F) max(0.190 -  8.188 * Re + 0.620 * F, 0)* 10^(-13);

x1 = [0.4632, 6.67]; y1 = DI_fun(x1(1), x1(2)); z1 = [x1, y1];
x2 = [0.2323, 6.67]; y2 = DI_fun(x2(1), x2(2)); z2 = [x2, y2];
x3 = [0.116 , 3.33]; y3 = DI_fun(x3(1), x3(2)); z3 = [x3, y3];
x4 = [0.1889, 3.33]; y4 = DI_fun(x4(1), x4(2)); z4 = [x4, y4];

Z = [z1; z2; z3; z4];
Z = [Z; Z(1,:)];

%% Di function
figure(1);clf;
set(gcf,'Visible','on')

alphaVal = 0.5;
fsurf(DI_fun, [0 1 0 10], 'EdgeColor','none')
hold on
plot3( Z(:,1) ,Z(:,2), Z(:,3) , 'w--', 'LineWidth', 2 )
scatter3(RE,FF, 4*ones(1,12), 'filled', 'MarkerFaceColor', 'k', 'MarkerEdgeColor','w')
hold off

colormap turbo 
c = colorbar;
set(c,'TickLabelInterpreter','latex');

title('$D^R_i = \max(0, 0.190 -  8.188 \cdot Re + 0.620 \cdot F) \cdot 10^{-13} $')
subtitle('$R^2 = 0.868$')
xlabel('Re [-]')
ylabel('F $\cdot 10^{-5}$ [kg/s]')

set(gca,'FontSize',12);

view(2)
exportgraphics(figure(1), ['Di.png'], "Resolution",300);

%% Gamma function

GG_fun = @(Re, F) max(3.158 + 11.922 * Re - 0.686 * F, 0);

x1 = [0.4632, 6.67]; y1 = GG_fun(x1(1), x1(2)); z1 = [x1, y1];
x2 = [0.2323, 6.67]; y2 = GG_fun(x2(1), x2(2)); z2 = [x2, y2];
x3 = [0.116 , 3.33]; y3 = GG_fun(x3(1), x3(2)); z3 = [x3, y3];
x4 = [0.1889, 3.33]; y4 = GG_fun(x4(1), x4(2)); z4 = [x4, y4];

Z = [z1; z2; z3; z4];
Z = [Z; Z(1,:)];

figure(2);clf;
set(gcf,'Visible','on')

fsurf(GG_fun, [0 1 0 10], 'EdgeColor','none')
hold on
plot3( Z(:,1) ,Z(:,2), Z(:,3) , 'w--', 'LineWidth', 2 )
scatter3(RE,FF, 5*ones(1,12), 'filled', 'MarkerFaceColor', 'k', 'MarkerEdgeColor','w')
hold off

colormap turbo 
c = colorbar;
set(c,'TickLabelInterpreter','latex');

title('$\Upsilon = \max(0, 3.158 - 11.922 \cdot Re + 0.686 \cdot F) $')
subtitle('$R^2 = 0.823$')
xlabel('Re [-]')
ylabel('F $\cdot 10^{-5}$ [kg/s]')

set(gca,'FontSize',12);

view(2)
exportgraphics(figure(2), ['Gamma.png'], "Resolution",300);