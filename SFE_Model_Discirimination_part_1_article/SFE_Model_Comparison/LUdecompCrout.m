function LU = LUdecompCrout(A)
    n = length(A);
    LU = A;
    % decomposition of matrix, Doolittle's Method
    for i = 1:1:n
        for j = 1:(i - 1)
            LU(i,j) = (LU(i,j) - LU(i,1:(j - 1))*LU(1:(j - 1),j)) / LU(j,j);
        end
        j = i:n;
        LU(i,j) = LU(i,j) - LU(i,1:(i - 1))*LU(1:(i - 1),j);
    end
    %LU = L+U-I
end
