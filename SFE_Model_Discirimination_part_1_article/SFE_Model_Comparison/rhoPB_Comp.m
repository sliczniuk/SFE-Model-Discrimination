function [rho] = rhoPB_Comp(t, p, Z, theta)

    R  = theta{12};
    MW = theta{14};

    rho_mol = p ./ (R .* t .* Z);       % Density in mol/m3
    rho = MW * rho_mol;                 % Density in kg/m3

end

