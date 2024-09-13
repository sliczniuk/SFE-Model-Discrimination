function D = Diffusion(Re, F, parameters)

    a = 0.19;%parameters{44};
    b = 8.188;%parameters{45};
    c = 0.62;%parameters{46};

    D =  a -  b * Re + c  * F * 10^5;
    
end