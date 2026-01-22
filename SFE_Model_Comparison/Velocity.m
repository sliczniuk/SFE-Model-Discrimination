function v = Velocity(F,rho,theta)

    r = theta{3};        % Extractor radius (m)

    A = pi * r^2 ;       % Cross-section of the extractor (m^2)
    
    v = F ./ (A .* rho);

end