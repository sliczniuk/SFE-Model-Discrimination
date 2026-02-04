function [T] = Reconstruct_T_from_enthalpy(h, P, theta)

    %addpath('C:\dev\casadi-windows-matlabR2016a-v3.5.2');
    %addpath('\\home.org.aalto.fi\sliczno1\data\Documents\casadi');
    import casadi.*

    if class(P)=="double"
         % Create symbolic temperature
        T_s             = MX.sym('T_s'  ,length(h)    ,1);
    
        % Based on symbolic T and know P calulate symbolic enthalpy h
        Z               = Compressibility( T_s, P,         theta );
        rho             = rhoPB_Comp(      T_s, P, Z,      theta );
        h_sym           = rho.*SpecificEnthalpy(T_s, P, Z, rho, theta);
        
        % Create function which compare the difference between symbolic h and
        % real h
        H               = h - h_sym;
        
        % Set a rootfinder to make the difference between h and h_sym goes to
        % zero
        g               = Function('g',{T_s},{H});
        G               = rootfinder('G','newton',g);
        
        % Guess value of T and use rootfinder
        T               = G(40+273);

    else

        % Create symbolic temperature
        THETA           = MX.sym('THETA',length(theta)  );
        H_s             = MX.sym('H_s'  ,length(h)      );
        P_s             = MX.sym('P_s'  ,length(P)      );
        T_s             = MX.sym('T_s'  ,length(h)    ,1);
    
        % Based on symbolic T and know P calulate symbolic enthalpy h
        Z               = Compressibility( T_s, P_s,         THETA );
        rho             = rhoPB_Comp(      T_s, P_s, Z,      THETA );
        h_sym           = rho.*SpecificEnthalpy(T_s, P_s, Z, rho, THETA );
        
        % Create function which compare the difference between symbolic h and
        % real h
        H               = H_s - h_sym;
        
        % Set a rootfinder to make the difference between h and h_sym goes to
        % zero
        g               = Function('g',{T_s,P_s,H_s,THETA},{H});
        G               = rootfinder('G','newton',g);
        
        % Guess value of T and use rootfinder
        T               = G(40+273, P, h, theta);
        
    end
end