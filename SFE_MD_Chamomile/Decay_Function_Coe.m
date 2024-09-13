function [Upsilon] = Decay_Function_Coe(Re, F, parameters)

    a = 3.158;%parameters{47};
    b = 11.922;%parameters{48};
    c = 0.6868;%parameters{49};

    Upsilon  = a + b * Re - c * F * 10^5;

end