function A_pinv_return = pseudoinverse(FI)
    % Compute the Moore-Penrose pseudoinverse of matrix A
    % without using the built-in pinv() function
    import casadi.*

    A = MX.sym('A',size(FI));

    % Compute the Singular Value Decomposition
    %[U, S, V] = svd(A, 'econ');
    foo = MySVD('foo', size(FI,1), size(FI,2), true);
    [U, S, V] = foo(A);
    S = diag(S);

    % Set a tolerance to identify negligible singular values
    %tol = 7e-09;%1e-12;%
    tol = max(size(A)) * eps * max(S);

    % Invert the singular values, ignoring those below the tolerance
    S_inv = MX(zeros(size(S')));
    for i = 1:min(size(S))
        S_inv(i, i) = if_else(S(i, i) > tol,  1 / S(i, i), 0);
    end

    % Compute the pseudoinverse
    A_pinv = V * S_inv * U';

    BB = Function('BB',{A},{A_pinv});
    A_pinv_return = full(BB(FI));

end
