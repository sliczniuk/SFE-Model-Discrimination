startup;

Time = 0:10:600;
load( "multi_pressure_results_T_lag.mat");

RES = cell2mat(results);

j_f = vertcat(RES.j);
j_i = vertcat(RES.j_initial);

figure(1)
hold on
scatter(j_f, j_i, 64, 'k', 'o', 'filled', MarkerFaceAlpha=0.5, MarkerEdgeAlpha=0.5);
hold off

indx_best = find(max(j_f) == j_f);

figure(2)

hold on

for ii = 1:64
    feedTemp       = RES(ii).feedTemp;
    feedTemp_plot  = [feedTemp, feedTemp(end)]-273;
    [xs, ys] = stairs(Time, feedTemp_plot);
    h = plot(xs, ys, 'Color', [0.7 0.7 0.7], 'LineWidth', 2);
end

feedTemp       = RES(indx_best).feedTemp;
feedTemp_plot  = [feedTemp, feedTemp(end)]-273;
stairs(Time, feedTemp_plot, 'Color', 'k', 'LineWidth',2)
hold off

figure(3)
hold on
for ii = 1:64
    feedFlow      = RES(ii).feedFlow;
    feedFlow_plot = [feedFlow, feedFlow(end)];
    [xs, ys] = stairs(Time, feedFlow_plot);
    plot(xs, ys, 'Color', [0.7 0.7 0.7], 'LineWidth', 2);
end
feedFlow      = RES(indx_best).feedFlow;
feedFlow_plot = [feedFlow, feedFlow(end)];
[xs, ys] = stairs(Time, feedFlow_plot);
plot(xs, ys, 'Color', 'k', 'LineWidth', 2)
hold off

figure(4)
hold on
for ii = 1:64
    feedPress_plot = RES(ii).feedPress;
    feedPress_plot = [feedPress_plot, feedPress_plot(end)];
    [xs, ys] = stairs(Time, feedPress_plot);
    plot(xs, ys, 'Color', [0.7 0.7 0.7], 'LineWidth', 2);
end
feedPress_plot = RES(indx_best).feedPress;
feedPress_plot = [feedPress_plot, feedPress_plot(end)];
[xs, ys] = stairs(Time, feedPress_plot);
plot(xs, ys, 'Color', 'k', 'LineWidth', 2)
hold off

figure(5)
hold on
for ii = 1:64
    Y_P = RES(ii).Y_cum_P;
    Y_L = RES(ii).Y_cum_L;
    plot(Time, Y_P, 'Color', [1 0 0 0.1], 'LineWidth', 2, HandleVisibility='off');
    plot(Time, Y_L, 'Color', [0 0 1 0.1], 'LineWidth', 2, HandleVisibility='off');
end
Y_P  = RES(indx_best).Y_cum_P;
Y_L  = RES(indx_best).Y_cum_L;
plot(Time, Y_P, 'Color', [1 0 0], 'LineWidth', 2, DisplayName='Power model')
plot(Time, Y_L, 'Color', [0 0 1], 'LineWidth', 2, DisplayName='Linear model')
hold off

figure(1); %legend box off; legend(Location="northoutside", NumColumns=3); 
fontsize(gcf, 16, "points"); xlabel('Final value of the cost function [-]'); ylabel('Inital value of the cost function [-]');
print(figure(1),'OptTraj_scatter.png','-dpng', '-r500'); 

figure(2); %legend box off; legend(Location="northoutside", NumColumns=numel(PRESS))
fontsize(gcf, 16, "points"); xlabel('Time min'); ylabel('T $^\circ$C');
print(figure(2),'OptTraj_temp.png','-dpng', '-r500'); 

figure(3); %legend box off; legend(Location="northoutside", NumColumns=numel(PRESS))
fontsize(gcf, 16, "points"); xlabel('Time min'); ylabel('F kg/s');
print(figure(3),'OptTraj_flow.png','-dpng', '-r500'); 

figure(4); %legend box off; legend(Location="northoutside", NumColumns=numel(PRESS))
fontsize(gcf, 16, "points"); xlabel('Time min'); ylabel('P bar');
print(figure(4),'OptTraj_press.png','-dpng', '-r500'); 

figure(5); legend('Power model', 'Linear model', Location='best'); legend box off;
%pbaspect([2 1 1])
fontsize(gcf, 16, "points"); xlabel('Time min'); ylabel('Yield gram');
print(figure(5),'OptTraj_yield.png','-dpng', '-r500');

%% Figure 6 – Output distributions at final time (best solution, Monte Carlo)
%  Propagates parameter + measurement uncertainty via MC sampling
N_mc = 1500;
out = trajectory_varying_pressure( ...
    RES(indx_best).feedTemp, ...
    RES(indx_best).feedFlow, ...
    RES(indx_best).feedPress, N_mc);

% Final-time samples: parameter uncertainty only (model) and with measurement noise (observed)
Y_P_model = out.Y_P_mc(:, end);
Y_L_model = out.Y_L_mc(:, end);
Y_P_obs   = out.Y_P_obs_mc(:, end);
Y_L_obs   = out.Y_L_obs_mc(:, end);

% --- Figure 6a: parameter uncertainty only ---
[pdf_P_m, y_P_m] = ksdensity(Y_P_model);
[pdf_L_m, y_L_m] = ksdensity(Y_L_model);

%%
figure(6)
hold on
fill([y_P_m, y_P_m(end), y_P_m(1)], [pdf_P_m, 0, 0], [1 0 0], FaceAlpha=0.3, EdgeColor='none', HandleVisibility='off');
fill([y_L_m, y_L_m(end), y_L_m(1)], [pdf_L_m, 0, 0], [0 0 1], FaceAlpha=0.3, EdgeColor='none', HandleVisibility='off');
plot(y_P_m, pdf_P_m, 'Color', [1 0 0], 'LineWidth', 2, DisplayName='Power model')
plot(y_L_m, pdf_L_m, 'Color', [0 0 1], 'LineWidth', 2, DisplayName='Linear model')
hold off
legend(Location='northoutside', NumColumns=2); legend box off;
fontsize(gcf, 16, "points");
xlabel('Yield at final time [g]'); ylabel('Probability density');
%title('Parameter uncertainty only');
print(figure(6),'OptTraj_dist_final_model.png','-dpng', '-r500');

% --- Figure 7: parameter + measurement uncertainty ---
[pdf_P_o, y_P_o] = ksdensity(Y_P_obs);
[pdf_L_o, y_L_o] = ksdensity(Y_L_obs);

figure(7)
hold on
fill([y_P_o, y_P_o(end), y_P_o(1)], [pdf_P_o, 0, 0], [1 0 0], FaceAlpha=0.3, EdgeColor='none', HandleVisibility='off');
fill([y_L_o, y_L_o(end), y_L_o(1)], [pdf_L_o, 0, 0], [0 0 1], FaceAlpha=0.3, EdgeColor='none', HandleVisibility='off');
plot(y_P_o, pdf_P_o, 'Color', [1 0 0], 'LineWidth', 2, DisplayName='Power model')
plot(y_L_o, pdf_L_o, 'Color', [0 0 1], 'LineWidth', 2, DisplayName='Linear model')
hold off
legend(Location='northoutside', NumColumns=2); legend box off;
fontsize(gcf, 16, "points");
xlabel('Yield at final time [g]'); ylabel('Probability density');
%title('Parameter + measurement uncertainty');
print(figure(7),'OptTraj_dist_final_obs.png','-dpng', '-r500');

%% Discrimination metrics at final time
fprintf('\n=== Discrimination metrics at t = %d min ===\n', Time(end));
fprintf('%-35s %10s %10s\n', '', 'Model', 'Observed');
fprintf('%-35s %10s %10s\n', '', '(param)', '(param+meas)');
fprintf('%s\n', repmat('-', 1, 57));

[m_AUC, m_JS, m_KS, m_OVL, m_Cohen, m_Bhatt, m_Power] = disc_metrics(Y_P_model, Y_L_model);
[o_AUC, o_JS, o_KS, o_OVL, o_Cohen, o_Bhatt, o_Power] = disc_metrics(Y_P_obs,   Y_L_obs);

fprintf('%-35s %10.4f %10.4f\n', 'AUC (0.5=same, 1=perfect)',        m_AUC,   o_AUC);
fprintf('%-35s %10.4f %10.4f\n', 'JS divergence (0=same, ln2=max)',   m_JS,    o_JS);
fprintf('%-35s %10.4f %10.4f\n', 'KS statistic (0=same, 1=max)',      m_KS,    o_KS);
fprintf('%-35s %10.4f %10.4f\n', 'Overlap coeff (0=none, 1=identical)', m_OVL, o_OVL);
fprintf('%-35s %10.4f %10.4f\n', 'Cohen''s d',                        m_Cohen, o_Cohen);
fprintf('%-35s %10.4f %10.4f\n', 'Bhattacharyya distance',            m_Bhatt, o_Bhatt);
fprintf('%-35s %10.4f %10.4f\n', 'Statistical power (alpha=0.05)',    m_Power, o_Power);

%% Bayes factor over the sampled trajectory
sigma2 = 2.45e-2;   % empirical measurement noise variance (cumulative yield)

% Subsample trajectory to specified measurement times
t_sample = [30 60 90 120 150 180 240 300 360 420 480 540 600];
t_idx    = t_sample / 10 + 1;   % indices into out.Time (step = 10 min)

Y_P_mc_sub = out.Y_P_mc(:, t_idx);
Y_L_mc_sub = out.Y_L_mc(:, t_idx);

% Parameter uncertainty only (sigma2 = 0 → fitted multivariate Gaussian)
BF_model = compute_trajectory_bayes_factor(Y_P_mc_sub, Y_L_mc_sub, 0);

% Parameter + measurement uncertainty
BF_obs   = compute_trajectory_bayes_factor(Y_P_mc_sub, Y_L_mc_sub, sigma2);

fprintf('\n=== Trajectory Bayes Factor (log scale, equal priors) ===\n');
fprintf('%-40s %10s %10s\n', '',                    'Model',   'Observed');
fprintf('%-40s %10s %10s\n', '',                    '(param)', '(param+meas)');
fprintf('%s\n', repmat('-', 1, 62));
fprintf('%-40s %10.3f %10.3f\n', 'E[log BF | M_P true]  (> 0 = correct)', BF_model.E_logBF_if_P, BF_obs.E_logBF_if_P);
fprintf('%-40s %10.3f %10.3f\n', 'E[log BF | M_L true]  (< 0 = correct)', BF_model.E_logBF_if_L, BF_obs.E_logBF_if_L);
fprintf('%-40s %10.4f %10.4f\n', 'P(correct | M_P true)',                  BF_model.prob_correct_P, BF_obs.prob_correct_P);
fprintf('%-40s %10.4f %10.4f\n', 'P(correct | M_L true)',                  BF_model.prob_correct_L, BF_obs.prob_correct_L);
fprintf('%-40s %10.4f %10.4f\n', 'P(correct) overall',                     BF_model.prob_correct,   BF_obs.prob_correct);

fprintf('\n=== Trajectory Bayes Factor (linear scale, L1/L2) ===\n');
fprintf('%-40s %10s %10s\n', '',                    'Model',   'Observed');
fprintf('%-40s %10s %10s\n', '',                    '(param)', '(param+meas)');
fprintf('%s\n', repmat('-', 1, 62));
fprintf('%-40s %10.3e %10.3e\n', 'E[BF | M_P true]  (> 1 = correct)', BF_model.E_BF_if_P, BF_obs.E_BF_if_P);
fprintf('%-40s %10.3e %10.3e\n', 'E[BF | M_L true]  (< 1 = correct)', BF_model.E_BF_if_L, BF_obs.E_BF_if_L);

% Jeffreys scale interpretation for E[log BF] when P is true
fprintf('\nJeffreys scale (|log BF|): <1 barely, 1–3 substantial, 3–5 strong, >5 decisive\n');

%% Multivariate JS divergence over the sampled trajectory
Y_P_obs_sub = out.Y_P_obs_mc(:, t_idx);
Y_L_obs_sub = out.Y_L_obs_mc(:, t_idx);

JS_model = mvn_js_mc(Y_P_mc_sub,  Y_L_mc_sub);
JS_obs   = mvn_js_mc(Y_P_obs_sub, Y_L_obs_sub);

fprintf('\n=== Multivariate JS divergence over trajectory (0 = identical, ln2 = %.4f = max) ===\n', log(2));
fprintf('  Parameter uncertainty only:    JS = %.4f\n', JS_model);
fprintf('  Parameter + measurement noise: JS = %.4f\n', JS_obs);

% Gaussian goodness-of-fit diagnostic (Mahalanobis d² ~ chi2(d))
fprintf('\n=== Gaussian fit diagnostic (d² should be ~ chi2(%d)) ===\n', numel(t_idx));
fprintf('%-30s %8s %8s %8s %8s %8s\n', 'Distribution', 'mean(d²)', 'exp(d²)', 'std(d²)', 'exp std', 'KS p-val');
fprintf('%s\n', repmat('-', 1, 72));
mvn_gof_print(Y_P_mc_sub,  'Power  (param only)');
mvn_gof_print(Y_L_mc_sub,  'Linear (param only)');
mvn_gof_print(Y_P_obs_sub, 'Power  (param+meas)');
mvn_gof_print(Y_L_obs_sub, 'Linear (param+meas)');

%% -----------------------------------------------------------------------
function mvn_gof_print(X, label)
%MVN_GOF_PRINT  Mahalanobis-distance diagnostic for multivariate Gaussian fit.
%  Under H0 (X ~ MVN), squared Mahalanobis distances d²_i ~ chi2(d).
%  Reports mean and std of d² vs chi2 expectations, plus KS p-value.

    [N, d] = size(X);
    mu = mean(X, 1);
    S  = cov(X) + 1e-8 * eye(d);
    R  = chol(S);
    Z  = (X - mu) / R;
    d2 = sum(Z.^2, 2);          % N×1, should be ~ chi2(d)

    % Kolmogorov-Smirnov test against chi2(d) CDF (no toolbox needed)
    d2_sorted = sort(d2);
    ecdf_vals = (1:N)' / N;
    tcdf_vals = chi2cdf(d2_sorted, d);
    ks_stat   = max(abs(ecdf_vals - tcdf_vals));
    ks_pval   = exp(-2 * N * ks_stat^2);   % Kolmogorov approximation

    flag = '';
    if ks_pval < 0.05; flag = ' (*)'; end

    fprintf('%-30s %8.2f %8d %8.2f %8.2f %8.4f%s\n', ...
        label, mean(d2), d, std(d2), sqrt(2*d), ks_pval, flag);
end

%% -----------------------------------------------------------------------
function js = mvn_js_mc(X, Y)
%MVN_JS_MC  Multivariate JS divergence estimated via fitted MVN densities.
%  Fits N(mu_X,S_X) and N(mu_Y,S_Y) to N×d sample matrices X and Y,
%  then evaluates JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M) using the
%  existing samples as quadrature points (M = 0.5*P + 0.5*Q).

    d    = size(X, 2);
    mu_X = mean(X, 1);
    mu_Y = mean(Y, 1);
    S_X  = cov(X) + 1e-8 * eye(d);
    S_Y  = cov(Y) + 1e-8 * eye(d);

    R_X = chol(S_X);  ld_X = 2 * sum(log(diag(R_X)));
    R_Y = chol(S_Y);  ld_Y = 2 * sum(log(diag(R_Y)));

    % Evaluate log densities at samples drawn from P
    lp_XX = mvn_logpdf_rows(X, mu_X, R_X, ld_X);   % log p(x), x ~ P
    lp_XY = mvn_logpdf_rows(X, mu_Y, R_Y, ld_Y);   % log q(x), x ~ P
    kl_PM = mean(lp_XX - logmix(lp_XX, lp_XY));     % KL(P || M)

    % Evaluate log densities at samples drawn from Q
    lp_YX = mvn_logpdf_rows(Y, mu_X, R_X, ld_X);   % log p(y), y ~ Q
    lp_YY = mvn_logpdf_rows(Y, mu_Y, R_Y, ld_Y);   % log q(y), y ~ Q
    kl_QM = mean(lp_YY - logmix(lp_YX, lp_YY));     % KL(Q || M)

    js = 0.5 * (kl_PM + kl_QM);
end

%% -----------------------------------------------------------------------
function ll = mvn_logpdf_rows(X, mu, R, logdet)
%MVNLOGPDF_ROWS  Log MVN density for each row of X.
%  R = chol(Sigma) upper triangular; logdet = 2*sum(log(diag(R)))
    d  = size(X, 2);
    Z  = (X - mu) / R;                                       % N×d
    ll = -0.5*d*log(2*pi) - 0.5*logdet - 0.5*sum(Z.^2, 2); % N×1
end

%% -----------------------------------------------------------------------
function lm = logmix(la, lb)
%LOGMIX  log(0.5*exp(la) + 0.5*exp(lb)) elementwise, numerically stable.
    M  = max(la, lb);
    lm = -log(2) + M + log1p(exp(min(la, lb) - M));
end

%% -----------------------------------------------------------------------
function [AUC, JS_div, KS_stat, OVL, cohen_d, bhatt_dist, stat_power] = disc_metrics(sP, sL)
%DISC_METRICS  Compute discrimination metrics between two sample vectors.

    N = numel(sP);

    % Common KDE grid
    y_common = linspace(min([sP; sL]) - 0.5, max([sP; sL]) + 0.5, 1000);
    p = ksdensity(sP, y_common);
    q = ksdensity(sL, y_common);
    dy = y_common(2) - y_common(1);
    p = p / (sum(p) * dy);
    q = q / (sum(q) * dy);

    % AUC (ROC)
    labels = [ones(N,1); zeros(N,1)];
    scores = [sP; sL];
    [~, si] = sort(scores, 'descend');
    ls = labels(si);
    tp = cumsum(ls)  / sum(ls);
    fp = cumsum(~ls) / sum(~ls);
    AUC = trapz(fp, tp);
    if AUC < 0.5; AUC = 1 - AUC; end

    % Jensen-Shannon divergence
    m_  = 0.5 * (p + q);
    ip  = p > 0 & m_ > 0;
    iq  = q > 0 & m_ > 0;
    KL_pm  = sum(p(ip) .* log(p(ip) ./ m_(ip))) * dy;
    KL_qm  = sum(q(iq) .* log(q(iq) ./ m_(iq))) * dy;
    JS_div = 0.5 * (KL_pm + KL_qm);

    % KS statistic
    [~, ~, KS_stat] = kstest2(sP, sL);

    % Overlap coefficient
    OVL = sum(min(p, q)) * dy;

    % Cohen's d
    pooled_std = sqrt(0.5 * (var(sP) + var(sL)));
    cohen_d    = abs(mean(sP) - mean(sL)) / pooled_std;

    % Bhattacharyya distance
    BC = sum(sqrt(p .* q)) * dy;
    bhatt_dist = -log(max(BC, eps));

    % Statistical power (two-sample t-test, alpha = 0.05)
    n1 = numel(sP);  n2 = numel(sL);
    se = pooled_std * sqrt(1/n1 + 1/n2);
    t_crit = 1.96;
    ncp    = abs(mean(sP) - mean(sL)) / se;   % non-centrality parameter
    stat_power = 1 - normcdf(t_crit - ncp) + normcdf(-t_crit - ncp);
end 
