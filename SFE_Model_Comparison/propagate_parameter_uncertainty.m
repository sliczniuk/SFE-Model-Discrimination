function [uncertainty] = propagate_parameter_uncertainty(method, Y_nominal, theta, Cov_theta, ...
    simulate_func, varargin)
% PROPAGATE_PARAMETER_UNCERTAINTY
% Propagates parameter uncertainties to model output using various methods.
%
% Methods:
%   'delta'      - First-order Taylor expansion (delta method)
%   'montecarlo' - Monte Carlo sampling from parameter distribution
%   'sigma'      - Sigma-point (unscented) transform
%
% Inputs:
%   method        - 'delta', 'montecarlo', or 'sigma'
%   Y_nominal     - Nominal output value(s) at theta
%   theta         - Nominal parameter vector (n_params x 1)
%   Cov_theta     - Parameter covariance matrix (n_params x n_params)
%   simulate_func - Function handle: Y = simulate_func(theta_perturbed)
%                   Returns scalar or vector output
%
% Optional parameters (name-value pairs):
%   'N_MC'        - Number of Monte Carlo samples (default: 1000)
%   'delta_rel'   - Relative perturbation for numerical Jacobian (default: 1e-4)
%   'alpha'       - Sigma-point spread parameter (default: 1e-3)
%   'beta'        - Prior distribution parameter (default: 2 for Gaussian)
%   'kappa'       - Secondary scaling parameter (default: 0)
%
% Outputs:
%   uncertainty   - Structure with:
%                   .mean     - Mean output (MC/sigma) or nominal (delta)
%                   .std      - Standard deviation of output
%                   .var      - Variance of output
%                   .CI_95    - 95% confidence interval [lower, upper]
%                   .samples  - (MC only) All sampled outputs
%                   .J        - (delta only) Jacobian matrix

%% Parse inputs
p = inputParser;
addRequired(p, 'method', @(x) ismember(x, {'delta', 'montecarlo', 'sigma'}));
addRequired(p, 'Y_nominal');
addRequired(p, 'theta');
addRequired(p, 'Cov_theta');
addRequired(p, 'simulate_func');
addParameter(p, 'N_MC', 1000);
addParameter(p, 'delta_rel', 1e-4);
addParameter(p, 'alpha', 1e-3);
addParameter(p, 'beta', 2);
addParameter(p, 'kappa', 0);

parse(p, method, Y_nominal, theta, Cov_theta, simulate_func, varargin{:});
opts = p.Results;

n_params = length(theta);
n_outputs = length(Y_nominal);

uncertainty = struct();

%% Select method
switch lower(method)
    case 'delta'
        %% ================================================================
        %  DELTA METHOD (First-Order Taylor Expansion)
        %  ================================================================
        % Var(Y) ≈ J * Cov(theta) * J'
        % where J = dY/d(theta) is the Jacobian

        % Compute Jacobian numerically using central differences
        J = zeros(n_outputs, n_params);

        for i = 1:n_params
            % Perturbation size
            delta = max(abs(theta(i)) * opts.delta_rel, 1e-10);

            % Perturbed parameters
            theta_plus = theta;
            theta_plus(i) = theta(i) + delta;

            theta_minus = theta;
            theta_minus(i) = theta(i) - delta;

            % Evaluate at perturbed points
            try
                Y_plus = simulate_func(theta_plus);
                Y_minus = simulate_func(theta_minus);

                % Central difference
                J(:, i) = (Y_plus - Y_minus) / (2 * delta);
            catch
                % If simulation fails, use forward difference
                try
                    Y_plus = simulate_func(theta_plus);
                    J(:, i) = (Y_plus - Y_nominal) / delta;
                catch
                    warning('Sensitivity computation failed for parameter %d', i);
                    J(:, i) = 0;
                end
            end
        end

        % Propagate variance
        Var_Y = J * Cov_theta * J';

        % Handle vector output
        if n_outputs > 1
            uncertainty.var = diag(Var_Y);
            uncertainty.std = sqrt(max(diag(Var_Y), 0));
            uncertainty.cov = Var_Y;
        else
            uncertainty.var = Var_Y;
            uncertainty.std = sqrt(max(Var_Y, 0));
        end

        uncertainty.mean = Y_nominal;
        uncertainty.CI_95 = [Y_nominal - 1.96*uncertainty.std, Y_nominal + 1.96*uncertainty.std];
        uncertainty.J = J;
        uncertainty.method = 'delta';

    case 'montecarlo'
        %% ================================================================
        %  MONTE CARLO SAMPLING
        %  ================================================================
        % Sample parameters from N(theta, Cov_theta) and propagate

        N_MC = opts.N_MC;

        % Cholesky decomposition for sampling
        try
            L = chol(Cov_theta + 1e-10*eye(n_params), 'lower');
        catch
            warning('Covariance not positive definite, using diagonal');
            L = diag(sqrt(max(diag(Cov_theta), 0)));
        end

        % Generate samples: theta_sample = theta + L * z, where z ~ N(0, I)
        Z = randn(n_params, N_MC);
        theta_samples = repmat(theta(:), 1, N_MC) + L * Z;

        % Preallocate output samples
        Y_samples = zeros(n_outputs, N_MC);
        valid_samples = true(1, N_MC);

        % Run simulations
        for i = 1:N_MC
            try
                Y_samples(:, i) = simulate_func(theta_samples(:, i));
            catch
                valid_samples(i) = false;
            end
        end

        % Remove failed samples
        Y_samples = Y_samples(:, valid_samples);
        n_valid = sum(valid_samples);

        if n_valid < 10
            warning('Only %d valid MC samples. Results may be unreliable.', n_valid);
        end

        % Compute statistics
        uncertainty.mean = mean(Y_samples, 2);
        uncertainty.std = std(Y_samples, 0, 2);
        uncertainty.var = var(Y_samples, 0, 2);

        % Percentile-based confidence intervals (more robust for non-Gaussian)
        uncertainty.CI_95 = [prctile(Y_samples, 2.5, 2), prctile(Y_samples, 97.5, 2)];

        % Store samples for further analysis
        uncertainty.samples = Y_samples;
        uncertainty.theta_samples = theta_samples(:, valid_samples);
        uncertainty.n_valid = n_valid;
        uncertainty.method = 'montecarlo';

    case 'sigma'
        %% ================================================================
        %  SIGMA-POINT (UNSCENTED) TRANSFORM
        %  ================================================================
        % Uses 2n+1 deterministic sigma points to capture mean and covariance

        alpha = opts.alpha;
        beta = opts.beta;
        kappa = opts.kappa;

        % Scaling parameters
        lambda = alpha^2 * (n_params + kappa) - n_params;
        gamma = sqrt(n_params + lambda);

        % Weights for mean and covariance
        W_m = zeros(2*n_params + 1, 1);
        W_c = zeros(2*n_params + 1, 1);

        W_m(1) = lambda / (n_params + lambda);
        W_c(1) = lambda / (n_params + lambda) + (1 - alpha^2 + beta);

        for i = 2:(2*n_params + 1)
            W_m(i) = 1 / (2 * (n_params + lambda));
            W_c(i) = 1 / (2 * (n_params + lambda));
        end

        % Compute square root of covariance matrix
        try
            S = chol(Cov_theta, 'lower');
        catch
            % Add small regularization if not positive definite
            S = chol(Cov_theta + 1e-10*eye(n_params), 'lower');
        end

        % Generate sigma points
        sigma_points = zeros(n_params, 2*n_params + 1);
        sigma_points(:, 1) = theta(:);

        for i = 1:n_params
            sigma_points(:, i+1) = theta(:) + gamma * S(:, i);
            sigma_points(:, n_params+i+1) = theta(:) - gamma * S(:, i);
        end

        % Propagate sigma points through model
        Y_sigma = zeros(n_outputs, 2*n_params + 1);
        valid_points = true(1, 2*n_params + 1);

        for i = 1:(2*n_params + 1)
            try
                Y_sigma(:, i) = simulate_func(sigma_points(:, i));
            catch
                valid_points(i) = false;
                Y_sigma(:, i) = Y_nominal;  % Use nominal as fallback
            end
        end

        if ~all(valid_points)
            warning('%d sigma points failed to evaluate', sum(~valid_points));
        end

        % Compute weighted mean
        Y_mean = zeros(n_outputs, 1);
        for i = 1:(2*n_params + 1)
            Y_mean = Y_mean + W_m(i) * Y_sigma(:, i);
        end

        % Compute weighted covariance
        Y_cov = zeros(n_outputs, n_outputs);
        for i = 1:(2*n_params + 1)
            diff = Y_sigma(:, i) - Y_mean;
            Y_cov = Y_cov + W_c(i) * (diff * diff');
        end

        % Extract results
        uncertainty.mean = Y_mean;
        if n_outputs > 1
            uncertainty.var = diag(Y_cov);
            uncertainty.std = sqrt(max(diag(Y_cov), 0));
            uncertainty.cov = Y_cov;
        else
            uncertainty.var = Y_cov;
            uncertainty.std = sqrt(max(Y_cov, 0));
        end

        uncertainty.CI_95 = [Y_mean - 1.96*uncertainty.std, Y_mean + 1.96*uncertainty.std];
        uncertainty.sigma_points = sigma_points;
        uncertainty.Y_sigma = Y_sigma;
        uncertainty.weights_mean = W_m;
        uncertainty.weights_cov = W_c;
        uncertainty.method = 'sigma';

    otherwise
        error('Unknown method: %s', method);
end

end
