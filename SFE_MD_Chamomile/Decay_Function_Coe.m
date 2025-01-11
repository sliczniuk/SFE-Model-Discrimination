function [Upsilon] = Decay_Function_Coe(Re, F, parameters)

    a = parameters{47};%3.158;
    b = parameters{48};%11.922;
    c = parameters{49};%0.6868;

    Upsilon  = a + b * Re - c * F * 10^5;

end