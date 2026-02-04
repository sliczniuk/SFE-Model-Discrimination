function [U, S, V] = custom_svd(A)
    % Custom SVD function compatible with CasADi variables
    import casadi.*
    
    % Get the dimensions of the input matrix A
    [m, n] = size(A);
    
    % Ensure A is a CasADi matrix
    assert(isa(A, 'casadi.MX') || isa(A, 'casadi.SX'), 'Input must be a CasADi variable');

    % Compute A^T * A (Gramian matrix), size will be (n x n)
    AtA = A' * A;

    % Perform custom eigenvalue decomposition on A^T * A
    [V, D] = custom_eig(AtA);  % V will be (n x n) and D (n x n)

    % Singular values are the square roots of the eigenvalues
    singular_values = sqrt(diag(D));  % Singular values (vector of size n)

    % Sort singular values and corresponding vectors
    [singular_values, idx] = sort(singular_values, 'descend');
    V = V(:, idx);  % Rearrange V columns based on sorted singular values
    
    % Construct the singular value matrix S
    %S = MX.zeros(m, n);  % S should have the size (m x n)
    %for i = 1:m
    %    S(i, i) = singular_values(i);  % Place singular values along the diagonal of S
    %end
    S = diag(singular_values);

    % Compute U using the relation A * V * inv(S) for non-zero singular values
    U = A * V(:, 1:min(m, n)) * diag(1 ./ singular_values(1:min(m, n)));

    % Ensure the dimensions of U are (m x m)
    if m > n
        % If A has more rows than columns, we need to pad U to be (m x m)
        extra_cols = MX.zeros(m, m - n);
        U = [U, extra_cols];
    end
end
