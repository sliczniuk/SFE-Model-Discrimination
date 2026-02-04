function xout = SimulateSystemEuler(f,x0,p,dt)

    %addpath('\\home.org.aalto.fi\sliczno1\data\Documents\casadi-windows-matlabR2016a-v3.5.1');
    import casadi.*

    N = 10;
    Nx = size(x0,1);
    xout = zeros(Nx,N+1);  
    xout(:,1) = x0;

    for k = 1:N
        k
        xdot        = f(xout(:,k), p(k,:)); 
        xout(:,k+1) = full(xout(:,k) + xdot*dt) ;
    end

end