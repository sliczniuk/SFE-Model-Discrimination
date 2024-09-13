startup;
delete(gcp('nocreate'));

%%
AA = xlsread('Regression.xlsx');
DI = [0.71383013	1.076801997	2.179470155	2.475532632	1.390707877	1.336111172	1.882954204	2.457886055	0.564935512	1.542106938	0.835725102	0.87349666];
GG = [4.229739602	3.091520556	2.359538225	1.132795818	2.204975712	2.739220425	1.868538631	1.69935869	3.452308202	1.995905641	3.012676539	2.596460037];
RE = [0.4632, 0.3783, 0.3029, 0.2619, 0.3579, 0.3140, 0.2635, 0.2323, 0.1787, 0.1160, 0.1889, 0.1512];
TT = [313, 313, 313, 313, 303, 303, 303, 303, 303, 303, 313, 313];
Tr = TT ./ 304;
RHO= [630, 691, 793, 840, 772, 802, 856, 891, 772, 891, 691, 793];

delta = 2.79 .* sqrt(7.38) .* Tr.^(1/4) .* (RHO./470);

FF = [6.67, 6.67, 6.67, 6.67, 6.67, 6.67, 6.67, 6.67, 3.33, 3.33, 3.33, 3.33];

%%
x1 = RE(1:8) ; y1 = DI(1:8);
x2 = RE(9:12); y2 = DI(9:12);

[mdl1,stuff1] = fit(x1',y1','poly1');
[mdl2,stuff2] = fit(x2',y2','poly1');

p1 = coeffvalues(mdl1);
p2 = coeffvalues(mdl2);

xtest  = linspace(0.1,0.5);
ytest1 = polyval(p1,xtest);
ytest2 = polyval(p2,xtest);

%{\
hold on
plt1 = plot(x1,y1,'ko', 'LineWidth',2);
plt2 = plot(x2,y2,'kd', 'LineWidth',2);
plot(xtest,ytest1, 'LineWidth',2);
plot(xtest,ytest2, 'LineWidth',2);
yline(0, 'k', 'LineWidth',2);
hold off

%text(0.3, 4.0, sprintf('~$~\\Upsilon = %.3f \\cdot + Re %.3f$ \n $R^2=%.3f$',[p1, stuff1.rsquare]))
%text(0.15, 6.0, sprintf('~$~ \\Upsilon = %.3f \\cdot Re + %.3f$ \n $R^2=%.3f$',[p2, stuff2.rsquare]))

text(0.35, 2.0, sprintf('$D_i^R = %.3f \\cdot Re + %.3f$ \n $R^2=~~%.3f$',[p1, stuff1.rsquare]))
text(0.15, -1.0, sprintf('$D_i^R = %.3f \\cdot Re + %.3f$ \n $R^2=~~%.3f$',[p2, stuff2.rsquare]))

legend([plt1(1),plt2(1)],'$6.67\times 10^{-5}[kg/s]$','$3.33\times 10^{-5}[kg/s]$','Location','best');
legend box off

ylabel('$D_i^R \times 10^{-13} [m^2/s]$')
%ylabel('$\Upsilon [-]$')
xlabel('Re [-]')
set(gca,'FontSize',12)

%exportgraphics(figure(1), ['Correlation_Di_Re.png'], "Resolution",300);
%exportgraphics(figure(1), ['Correlation_Gamma_Re.png'], "Resolution",300);
%close all
%}

%% Surface

[sf, ml] = fit([RE', FF'],GG','poly11');

% Create figure
figure1 = figure;
% Create axes
axes1 = axes('Parent',figure1);

plt = plot(sf, [RE', FF'],GG');

% Create zlabel
%zlabel({'$Di \times 10^{-13} [m/s^2]$'});
zlabel({'$\Upsilon [-]$'});

% Create ylabel
ylabel({'$F \times 10^{-5}[kg/s]$'}, 'Position',[-0.1 3.5 2]);

% Create xlabel
xlabel({'Re[-]'}, 'Position',[0.15 1.0 2]);

%text(-0.1, 5, 8.5, sprintf('$D_i^R = %.3f  %.3f \\cdot Re + %.3f \\cdot F \\times 10^{5} $ \n $R^2=~%.3f$',[coeffvalues(sf), ml.rsquare]))
text(-0.1, 5, 14.5, sprintf('$~~\\Upsilon = %.3f + %.3f \\cdot Re %.3f \\cdot F \\times 10^{5} $ \n $R^2=%.3f$',[coeffvalues(sf), ml.rsquare]))

%set(gca,'FontSize',12)

set(plt, 'EdgeAlpha',0, 'FaceAlpha',0.5)
%exportgraphics(figure(1), ['Di_Re_F.png'], "Resolution",300);