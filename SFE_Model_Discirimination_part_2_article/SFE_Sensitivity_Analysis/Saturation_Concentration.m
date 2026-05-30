function [FLUID_SAT] = Saturation_Concentration(SOLID, shape, Di)

    FLUID_SAT = Di .* exp( -shape .* ( SOLID ) );
end