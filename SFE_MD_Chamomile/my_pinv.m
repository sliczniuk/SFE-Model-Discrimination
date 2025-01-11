function X = my_pinv(A,tol)
import casadi.*

    [m, n] = size(A);
    foo = MySVD('foo', n, m, true);
    [U,S,V] = foo(A);
    % Handle the case when S is a vector (rank-1 matrix)
    if_else_case = if_else(min(size(S)) == 1, S(1), diag(S));  % Convert S to a diagonal matrix if it's not already
    
    % If no tolerance is provided, set a default tolerance based on matrix size
    if nargin == 1
        tol = max(m, n) * S(1) * eps;  % Use the largest singular value to compute the tolerance
    end
    
    % Count the number of singular values greater than tolerance
    singular_value_check = S > tol;
    r = sum(singular_value_check);
    
    % If no singular values are greater than tolerance, return a zero matrix
    if_else_case_r = if_else(r == 0, MX.zeros(n, m), V(1:end, 1:r) * diag(1 ./ S(1:r)) * U(1:end, 1:r)');
    
    % Assign the final result to X
    X = if_else_case_r;
end