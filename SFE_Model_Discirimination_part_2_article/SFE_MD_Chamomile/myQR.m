function [Q_val, R_val] = myQR(A_val)

    import casadi.*
    % CasADi symbolic variable for matrix A
    [n, m] = size(A_val);

    A = MX.sym('A', n, m); % SX is used for CasADi symbolic expressions

    % Initialize an identity matrix
    Q = MX.eye(n);
    R = MX(A); % Initially, R is A
    
    % Perform the Gram-Schmidt process to get Q and R
    for i = 1:m
        % Orthogonalization
        norm_val = sqrt(dot(R(:,i), R(:,i)));
        Q(:,i) = R(:,i) / norm_val;  % Normalize the column
        
        for j = i+1:m
            proj = dot(Q(:,i), R(:,j));
            R(:,j) = R(:,j) - proj * Q(:,i);  % Subtract projection
        end
        
        % Set the lower part of the matrix to zero manually
        for k = i+1:n
            R(k,i) = 0;  % Manually set the lower triangular part to zero
        end
        R(i,i) = norm_val;  % Store the norm as part of R
    end

    f_qr = Function('f_qr', {A}, {Q, R});
    [Q_val, R_val] = f_qr(A_val);
    Q_val = full(Q_val);
    R_val = full(R_val);

end