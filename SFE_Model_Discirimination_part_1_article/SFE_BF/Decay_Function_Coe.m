function [Upsilon] = Decay_Function_Coe(Re, F, parameters)

    a = parameters{51};
    b = parameters{52};
    c = parameters{53};

    Upsilon  = a + b * Re + c * F * 10^5;

end