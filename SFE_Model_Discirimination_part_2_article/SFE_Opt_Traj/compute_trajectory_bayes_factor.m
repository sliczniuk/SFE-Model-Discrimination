function BF = compute_trajectory_bayes_factor(Y_P_mc, Y_L_mc, sigma2)
%COMPUTE_TRAJECTORY_BAYES_FACTOR  Expected log Bayes factor over the full trajectory.
%
%  BF = COMPUTE_TRAJECTORY_BAYES_FACTOR(Y_P_mc, Y_L_mc, sigma2)
%
%  Inputs:
%    Y_P_mc  — N_mc × N_time, MC model output samples for Power model
%    Y_L_mc  — N_mc × N_time, MC model output samples for Linear model
%    sigma2  — scalar measurement noise variance
%              sigma2 > 0: uses MC-based marginal likelihood with Gaussian kernel
%              sigma2 = 0: uses fitted multivariate Gaussian (parameter uncertainty only)
%
%  Output struct:
%    BF.logBF_if_P      — N_mc×1, log BF for each simulated observation from P
%    BF.logBF_if_L      — N_mc×1, log BF for each simulated observation from L
%    BF.E_logBF_if_P    — expected log BF when P is true (should be > 0)
%    BF.E_logBF_if_L    — expected log BF when L is true (should be < 0)
%    BF.prob_correct_P  — fraction of P observations correctly identified
%    BF.prob_correct_L  — fraction of L observations correctly identified
%    BF.prob_correct    — overall probability of correct model selection

N_mc = size(Y_P_mc, 1);
N_t  = size(Y_P_mc, 2);

logBF_if_P = zeros(N_mc, 1);
logBF_if_L = zeros(N_mc, 1);

rng(123);

if sigma2 > 0
    %% MC-based marginal likelihood with measurement noise kernel
    %  P(y | M_k) = (1/N_mc) sum_i N(y; Y_k_mc(i,:), sigma2 * I)
    %  For each test observation, simulate y = Y_mc(i,:) + noise

    fprintf('Computing trajectory Bayes factors (with measurement noise)...\n');
    for i = 1:N_mc
        % Observation from Power model
        y_P = Y_P_mc(i, :) + sqrt(sigma2) * randn(1, N_t);
        logML_PP = log_marginal_mc(y_P, Y_P_mc, sigma2);
        logML_PL = log_marginal_mc(y_P, Y_L_mc, sigma2);
        logBF_if_P(i) = logML_PP - logML_PL;

        % Observation from Linear model
        y_L = Y_L_mc(i, :) + sqrt(sigma2) * randn(1, N_t);
        logML_LP = log_marginal_mc(y_L, Y_P_mc, sigma2);
        logML_LL = log_marginal_mc(y_L, Y_L_mc, sigma2);
        logBF_if_L(i) = logML_LP - logML_LL;

        if mod(i, 200) == 0
            fprintf('  %d / %d done\n', i, N_mc);
        end
    end
else
    %% Fitted multivariate Gaussian (no measurement noise)
    %  P(y | M_k) = N(y; mu_k, Sigma_k)
    fprintf('Computing trajectory Bayes factors (parameter uncertainty only)...\n');

    mu_P = mean(Y_P_mc, 1);
    mu_L = mean(Y_L_mc, 1);

    Sigma_P = cov(Y_P_mc) + 1e-8 * eye(N_t);
    Sigma_L = cov(Y_L_mc) + 1e-8 * eye(N_t);

    % Precompute Cholesky factors
    R_P = chol(Sigma_P);
    R_L = chol(Sigma_L);
    logdet_P = 2 * sum(log(diag(R_P)));
    logdet_L = 2 * sum(log(diag(R_L)));

    for i = 1:N_mc
        logBF_if_P(i) = log_mvnpdf_chol(Y_P_mc(i,:), mu_P, R_P, logdet_P) ...
                       - log_mvnpdf_chol(Y_P_mc(i,:), mu_L, R_L, logdet_L);

        logBF_if_L(i) = log_mvnpdf_chol(Y_L_mc(i,:), mu_P, R_P, logdet_P) ...
                       - log_mvnpdf_chol(Y_L_mc(i,:), mu_L, R_L, logdet_L);
    end
end

fprintf('Bayes factor computation complete.\n');

%% Assemble output
% Log-scale: ln(L1) - ln(L2)
BF.logBF_if_P     = logBF_if_P;
BF.logBF_if_L     = logBF_if_L;
BF.E_logBF_if_P   = mean(logBF_if_P);
BF.E_logBF_if_L   = mean(logBF_if_L);

% Linear scale: L1/L2 = exp(ln BF)
BF.BF_if_P        = exp(logBF_if_P);
BF.BF_if_L        = exp(logBF_if_L);
BF.E_BF_if_P      = mean(exp(logBF_if_P));
BF.E_BF_if_L      = mean(exp(logBF_if_L));

BF.prob_correct_P = mean(logBF_if_P > 0);   % P correctly identified
BF.prob_correct_L = mean(logBF_if_L < 0);   % L correctly identified
BF.prob_correct   = 0.5 * (BF.prob_correct_P + BF.prob_correct_L);

end

%% -----------------------------------------------------------------------
function logML = log_marginal_mc(y, Y_ref, sigma2)
%LOG_MARGINAL_MC  Log marginal likelihood via MC with diagonal Gaussian kernel.
%  log P(y | M_k) = -log(N) + logsumexp_i[ log N(y; Y_ref(i,:), sigma2*I) ]

    N = size(Y_ref, 1);
    d = size(Y_ref, 2);

    % ||y - Y_ref(i,:)||^2 for each reference sample
    diff_sq = sum((Y_ref - y).^2, 2);   % N × 1

    log_liks = -0.5 * d * log(2 * pi * sigma2) - diff_sq / (2 * sigma2);

    logML = -log(N) + logsumexp(log_liks);
end

%% -----------------------------------------------------------------------
function s = logsumexp(x)
    m = max(x);
    s = m + log(sum(exp(x - m)));
end

%% -----------------------------------------------------------------------
function ll = log_mvnpdf_chol(x, mu, R, logdet)
%LOG_MVNPDF_CHOL  Log multivariate normal PDF using precomputed Cholesky.
%  R = chol(Sigma), logdet = 2*sum(log(diag(R)))

    d = length(x);
    z = R' \ (x - mu)';
    ll = -0.5 * d * log(2*pi) - 0.5 * logdet - 0.5 * (z' * z);
end
