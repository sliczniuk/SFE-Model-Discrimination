function P_inv = moore_penrose_pinv(A)
    % This function computes the Moore-Penrose pseudoinverse of a CasADi matrix A
    import casadi.*
        
    % Get the dimensions of A
    [m, n] = size(A);

    % Perform custom SVD
    foo = MySVD('foo', n, m, true);
    [U, S, V] = foo(A);
    S = diag(S);
    
    % Initialize pseudoinverse of S (which is diagonal)
    S_inv = MX.zeros(n, m);  % Pseudoinverse of S will be (n x m)

    % Invert the singular values with CasADi's conditional statement
    for i = 1:m
        % If the singular value is greater than a small threshold, invert it
        S_inv(i, i) = if_else(S(i, i) > 1e-12, 1 / S(i, i), 0);
    end
    
    % Compute the Moore-Penrose pseudoinverse using V, S_inv, and U'
    % V: n x n, S_inv: n x m, U': m x m
    P_inv = V .* S_inv .* diag(U');
end
