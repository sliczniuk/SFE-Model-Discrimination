function D_L = axial_diffusion(rho, parameters)

    a = parameters{48};
    b = parameters{49};
    
    D_L = a .* rho + b;

end