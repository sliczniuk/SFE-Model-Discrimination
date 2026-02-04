function d = myDet(A)
    % This function calculates the determinant of matrix A using Laplace expansion.
    % Adjusted to be compatible with CasADi variables.

    import casadi.*  % Make sure to import CasADi

    [n, m] = size(A);

    % Ensure the matrix is square
    if n ~= m
        error('Matrix must be square');
    end

    % Base case for 1x1 matrix
    if n == 1
        d = A(1,1);
        return;
    end

    % Initialize determinant to zero
    d = MX.zeros(1, 1);

    % Loop over first row elements for expansion
    for j = 1:n
        % Create the minor matrix for A(1,j)
        % Using all rows except the first, and all columns except the j-th
        Minor = A([2:n], [1:j-1, j+1:n]);

        % Recursive call for the minor determinant
        MinorDet = myDet(Minor);

        % Add to the total determinant
        d = d + (-1)^(1+j) * A(1,j) * MinorDet;
    end
end
