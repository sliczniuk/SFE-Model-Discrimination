function [A_inv]=tik_inv(A)

    delta = 1e-7;

    [m, n] = size(A);
    A_inv = (A' * A + delta * eye(n)) \ A';
    
end