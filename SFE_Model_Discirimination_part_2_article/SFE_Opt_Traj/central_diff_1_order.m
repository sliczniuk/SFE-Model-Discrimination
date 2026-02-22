function d = central_diff_1_order(U, U_0, U_B, dz)

    d = [U(2:end); U_B] - [U_0; U(1:end-1)];

    d = d ./ (2*dz);

end