function [Upsilon] = Decay_Function_Coe(Re, F, parameters)

    a = 3.158;
    b = 11.922;
    c = -0.6868;

    Upsilon  = a + b * Re + c * F * 10^5;

end