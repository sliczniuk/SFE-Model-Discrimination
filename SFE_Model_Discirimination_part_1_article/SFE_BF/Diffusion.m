function D = Diffusion(Re, F, parameters)

    a = parameters{48};
    b = parameters{49};
    c = parameters{50};

    %D =  a -  b * Re + c  * F * 10^5;
    D =  a + b * Re + c  * F * 10^5;
    D = max(D,0);
end