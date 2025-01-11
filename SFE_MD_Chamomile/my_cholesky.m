function L = my_cholesky(A)
    % Custom Cholesky decomposition compatible with CasADi variables
    % Inputs:
    %   A - A symmetric positive definite matrix (CasADi SX or MX variable)
    % Outputs:
    %   L - Lower triangular matrix such that A = L * L'

    import casadi.*

    % Get the size of the matrix A
    [n, m] = size(A);

    % Ensure A is square
    assert(n == m, 'Matrix must be square');

    % Initialize L as an n x n symbolic matrix
    L = MX.zeros(n, n);  % You can also use MX for larger systems

    % Compute each element of L symbolically using vectorized operations and CasADi's if_else
    for i = 1:n
        % Diagonal elements
        L(i, i) = sqrt(A(i, i) - sum(L(i, 1:i-1).^2));
        
        % Off-diagonal elements
        for j = i+1:n
            L(j, i) = (A(j, i) - L(j, 1:i-1) * L(i, 1:i-1)') / L(i, i);
        end
    end
end
