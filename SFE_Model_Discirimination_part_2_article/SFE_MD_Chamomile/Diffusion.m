function D = Diffusion(Re, F, parameters)

    a = parameters{44};%0.19;
    b = parameters{45};%8.188;
    c = parameters{46};%0.62;

    D =  a -  b * Re + c  * F * 10^5;
    D = max(D,0);
end