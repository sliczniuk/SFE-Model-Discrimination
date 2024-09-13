function d = central_diff_2_order(U, U_0, U_B, dz)

    d = [ U_0;  U(1:end-1) ]    - 2*U      + [ U(2:end) ; U_B   ];

    d = d ./ (dz^2);

end