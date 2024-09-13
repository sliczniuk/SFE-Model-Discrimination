function mu = Viscosity(T,rho)

    % The temperature ranges of the dataare 310–900K and the pressure from 7.5MPa to 101.5MPa.
    % T[K], P[bar]
    %{
    function mu = Viscosity(T,P,rho,theta)
    A1 = theta{35}; % -1.146067E-1;
    A2 = theta{36}; %  6.978380E-7;
    A3 = theta{37}; %  3.976765E-10;
    A4 = theta{38}; %  6.336120E-2;
    A5 = theta{39}; % -1.166119E-2;
    A6 = theta{40}; %  7.142596E-4;
    A7 = theta{41}; %  6.519333E-6;
    A8 = theta{42}; % -3.567559E-1;
    A9 = theta{43}; %  3.180473E-2;
    
    mu = ( A1 + A2.*P + A3.*P.^2 + A4.*log(T) + A5.*(log(T)).^2 + A6.*(log(T)).^3 ) ./ ( 1 + A7.*P + A8.*log(T) + A9.*(log(T)).^2 );
    %}
    
    Na     = 6.022e23;
    sigma  = 0.378421e-9;
    M      = 44.0095e-3;
    Tt     = 216.592;
    rho_tl = 1178.53; % density at the triple point
    
    rho_r  = rho / rho_tl;
    T_star = T   / 200.760;
    Tr     = T   / Tt;
    
    a	   = [ 1749.354893188350, -369.069300007128, 5423856.34887691, -2.21283852168356, -269503.247933569, 73145.021531826, 5.34368649509278];
    
    b      = [ -19.572881, 219.73999, -1015.3226, 2471.0125, -3375.1717, 2491.6597, -787.26086,  14.085455, -0.34664158];
    
    t      = [ 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2.5, 5.5];
    
    gamma  = 8.06282737481277;
    c      = [0.360603235428487, 0.121550806591497 ];
    
    mu_0   = 1.0055.*sqrt(T) ./ ( a(1) + a(2)*T.^(1/6) + a(3).*exp(a(4)*T.^(1/3)) + ( (a(5) + a(6)*T.^(1/3) ) ./exp(T.^(1/3)) ) + a(7).*sqrt(T) );
    
    B      = 0;
    for i=1:8
        B = B + b(i+1) ./ (T_star.^t(i));
    end
    B = b(1) + B;
    
    mu_l   = mu_0 .* B .* sigma^3 .* Na ./ M;
    
    mu_tl  = 0.09436; % rho_tl^(2/3) * sqrt(kB*Tt)/( (M/Na)^(1/6) );
    
    mu_r   = mu_tl .*(  c(1).*Tr.*(rho_r.^3) + (rho_r.^2 + rho_r.^gamma)./(Tr - c(2)) );
    
    mu = mu_0 + rho.*mu_l + mu_r;
    
    % Units are mPa*s -> 1e-3 * Pa*s

    mu = mu.*1e-3;

end