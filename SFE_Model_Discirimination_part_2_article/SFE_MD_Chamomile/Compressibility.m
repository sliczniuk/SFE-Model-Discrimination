function Z = Compressibility(t,p,theta)
    %{
     import casadi.*
    T = MX.sym('T',length(t));
    P = MX.sym('P',length(p));
    THETA = MX.sym('THETA',length(theta));

    %% Unpacking parameters
    % theta = [nstages, C0solid, V, epsi, dp, L, rho_s, km, mi, Tc, Pc, R, kappa];
    %          1        2        3  4     5   6  7      8   9   10  11  12 13

    Tc      = THETA{10};   % Critical temperature [K]
    Pc      = THETA{11};   % Critical pressure [bar]
    R       = THETA{12};   % Universal gas constant, [m3-bar/K-mol]
    kappa   = THETA{13};
    MW      = THETA{14};   % Molar mass [g/mol]
    Tr      = T ./ Tc;
    a       = 0.4572350 .* R.^2 .* Tc.^2 ./ Pc;
    b       = 0.0777961 .* R    .* Tc    ./ Pc;

    alpha = (1 + kappa .* (1 - sqrt(Tr))).^2;

    A = a .* alpha .* P ./ R.^2 ./ T.^2;
    B = b .* P ./ R ./ T;

    %% Creating symbolic variables for CasADi rootfinder
    z = MX.sym('z',length(T));      % Compressibility factor

    %%
    % Cubic polynomial in z from Equation of State
    % Find z such that the polynomial is zero
    pol =                         z.^3 ...
        - (1 - B)              .* z.^2 ...
        + (A - 2.*B - 3.*B.^2) .* z ...
        - ( A .* B - B.^2 - B.^3);

    %disp("Case 1, T and P are symbolic")
    g = Function('g',{z,T,P,THETA},{pol});
    
%{

     * If it failed due to an unrecoverable failure in rhs, then we return
     * the value CV_RHSFUNC_FAIL.
     *
     * Otherwise, a recoverable failure occurred when solving the 
     * nonlinear system (cvNls returned nflag == CONV_FAIL or RHSFUNC_RECVR). 
     * In this case, if ncf is now equal to maxncf or |h| = hmin, 
     * we return the value CV_CONV_FAILURE (if nflag=CONV_FAIL) or
     * CV_REPTD_RHSFUNC_ERR (if nflag=RHSFUNC_RECVR).
     * If not, we set *nflagPtr = PREV_CONV_FAIL and return the value
     * PREDICT_AGAIN, telling cvStep to reattempt the step.
    
    %}
    
    opts = struct;
    %opts.max_iter = 10;

    opts.error_on_fail = 0;
    %opts.abstol  = 1e-4;
    %opts.abstolStep = 1e-4;
    opts.print_iteration = 0;
    G = rootfinder('G','newton',g,opts);
    
    try
        Z = G(unifrnd(1.1,2),t,p,theta);
    catch
        Z = G(unifrnd(0,0.9),t,p,theta);
    end
    %}
    %Z = theta(end);
    Z = Cardano(t,p,theta);
end