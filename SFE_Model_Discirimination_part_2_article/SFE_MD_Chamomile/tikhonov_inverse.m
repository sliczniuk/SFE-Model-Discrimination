function X = tikhonov_inverse(A)
    % This function computes the inverse of a matrix using Tikhonov regularization
    % A: The matrix to be inverted (ill-conditioned or close to singular)
    % lambda: The regularization parameter (must be a positive scalar)

    lambda = 1e-4;
    
    % Ensure A is a square matrix
    [m, n] = size(A);
    if m ~= n
        error('Matrix A must be square');
    end
    
    % Ensure lambda is positive
    if lambda <= 0
        error('Regularization parameter lambda must be positive');
    end
    
    % Compute the regularized inverse
    I = eye(n);  % Identity matrix of size n
    % Regularized matrix A_reg = A' * A + lambda * I
    A_reg = A' * A + lambda * I;
    
    % QR decomposition of A_reg (QR decomposition of a square matrix)
    [Q, R] = qr(A_reg);  % This performs QR decomposition of A_reg
    
    % Solve R * X = Q' * A'
    X = R \ (Q' * A');

end
