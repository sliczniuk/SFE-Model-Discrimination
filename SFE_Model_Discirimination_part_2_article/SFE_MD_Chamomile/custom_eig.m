function [V, D] = custom_eig(A)
    % Custom eigenvalue decomposition for CasADi variables using Power Iteration
    import casadi.*
    
    % Ensure A is a square CasADi matrix
    assert(size(A, 1) == size(A, 2), 'Matrix must be square');
    assert(isa(A, 'casadi.MX') || isa(A, 'casadi.SX'), 'Input must be a CasADi variable');
    
    n = size(A, 1);
    V = MX.zeros(n, n);  % Initialize eigenvector matrix
    D = MX.zeros(n, n);  % Initialize diagonal matrix of eigenvalues
    
    % Parameters for power iteration
    num_iterations = 100;  % Number of iterations for convergence
    tol = 1e-6;            % Tolerance for convergence
    
    % Iterative process to find eigenvalues and eigenvectors
    for i = 1:n
        % Start with a random vector
        v = MX.sym(['v' num2str(i)], n, 1);  % Symbolic eigenvector
        
        % Power iteration to find the dominant eigenvalue
        for k = 1:num_iterations
            v_new = A * v;
            v_new = v_new / norm(v_new);  % Normalize the vector
            
            % Check convergence
            %if norm(v_new - v) < tol
            %    break;
            %end
            v = v_new;
        end
        
        % Eigenvalue corresponding to the eigenvector
        lambda = (v' * A * v) / (v' * v);
        
        % Store eigenvalue and eigenvector
        D(i, i) = lambda;
        V(:, i) = v;
        
        % Deflate the matrix by removing the found eigenvalue/eigenvector
        A = A - lambda * (v * v');
    end
end
