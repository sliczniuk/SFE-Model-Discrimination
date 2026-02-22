function d = backward_diff_1_order(U, U_0, U_B, dz)

    d = U - [U_0; U(1:end-1)];

    d = d ./ dz;

end