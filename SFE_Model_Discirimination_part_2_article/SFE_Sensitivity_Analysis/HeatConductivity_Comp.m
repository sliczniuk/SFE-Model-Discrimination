function k = HeatConductivity_Comp(t, RHO)

import casadi.*

%{
function k = HeatConductivity_Comp(t, P, Z, RHO, theta)
A1 = theta{28}; % -105.161;
A2 = theta{29}; % 0.9007;
A3 = theta{30}; % 0.0007;
A4 = theta{31}; % 3.50E-15;
A5 = theta{32}; % 3.76E-10;
A6 = theta{33}; % 0.7500;
A7 = theta{34}; % 0.0017;

k = ( A1 + A2.*rho + A3.*(rho.^2) + A4.*(rho.^3).*(T.^3) + A5.*(rho.^4) + A6.*T + A7.*(T.^2) ) ./ (T.^(1/2));
%}

    T   = MX.sym('T'  ,numel(t)  );
    rho = MX.sym('rho',numel(RHO));

    Tc    = 304.1282;
    Pc    = 73.7650;
    Tref  = 3/2*Tc;
    rho_c = 467.6;
    
    RD = 1.02;
    v = 0.63;
    gamma = 1.239;
    
    Gamma = 0.052;
    xi_0 = 1.50e-10;
    
    q_D = 1/(4e-10);
    kB =  1.380649e-23;
    
    Tr = T/Tc;
    
    Lk = [1.51874307e-2, 2.80674040e-2, 2.28564190e-2, -7.41624210e-3];
    
    B1 = [1.00128e-2,  5.60488e-2, -8.11620e-2,  6.24337e-2, -2.06336e-2,  2.53248e-3]*1e3;
    B2 = [4.30829e-3, -3.58563e-2,  6.71480e-2, -5.22855e-2,  1.74571e-2, -1.96414e-3]*1e3;
    
    Delta_Tc    = Tr - 1;
    Delta_rho_c = rho/rho_c - 1;
    
    k0 = sqrt(Tr) ./ ( Lk(1)./(Tr.^0) + Lk(2)./(Tr.^1) + Lk(3)./(Tr.^2) + Lk(4)./(Tr.^3) );
    
    Delta_k = MX(6,numel(T));
    for i=1:6
        Delta_k(i,:) =  (B1(i) + B2(i).*(T/Tc) ) .* (rho/rho_c).^i;
    end
    Delta_k = sum(Delta_k);
    
    %%
    %{
                [Cp,Cv] = SpecificHeatComp(T, P, Z, rho, theta);
    
                Cp = 1000*Cp;
                Cv = 1000*Cv;
    
                drhodp_Tref = drhodp(Tref, P, rho, theta);
                drhodp_T    = drhodp(T, P, rho, theta);
    
                %%
                xi = xi_0 * (Pc * rho / Gamma / rho_c^2 )^(v/gamma) * (drhodp_T - (Tref/T) * drhodp_Tref )^(v/gamma) ;
    
                Omega  = 2/pi * ( (1-Cv/Cp) *atan(q_D*xi) + Cv/Cp*q_D*xi );
    
                Omega0 = 2/pi * ( 1-exp( - 1/ ( (q_D*xi)^(-1) + ( (q_D*xi*rho_c/rho)^2 ) / 3 ) ) );
    %}
    %%
    %            Delta_k_c = rho*Cp*RD*kB*T/(6*pi*(mu*10^-6)*xi) * (Omega-Omega0)
    
    Delta_k_c_1 = (-17.47 - 44.88 .* Delta_Tc) ./ ( 0.8563 - exp(8.865.*Delta_Tc + 4.16.*Delta_rho_c.^2 + 2.302.*Delta_Tc.*Delta_rho_c - Delta_rho_c.^3) - 0.4503.*Delta_rho_c - 7.197*Delta_Tc );
    
    k = k0 + Delta_k' + Delta_k_c_1;

    fk = Function('fk',{T,rho}, {k});

    k = full(fk(t,RHO));

    %k = Delta_k;

    % Units [ 10^-3 (W / m * K) ]
    %error('Double check units')


end