function D = Diffusion(Re, F, parameters)

    a = 0.19;
    b = -8.188;
    c = 0.62;

    %D =  a -  b * Re + c  * F * 10^5;
    D =  a + b * Re + c  * F * 10^5;
    D = max(D,0);
end