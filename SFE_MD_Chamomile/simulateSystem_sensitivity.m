function [xout] = simulateSystem_sensitivity(varargin)
%function [yout,tout,xout] = simulateSystem(varargin)

    % [y, t, x] = Simulate(F, g, x0, u); 
    
    import casadi.*
    
    F = varargin{1}; 
    G = varargin{2}; 
    x0 = varargin{3};
    
    u = varargin{4};
    N = size(u,1);
        
    f = @(x,u) full(F(x,u));
    g = @(x,u) full(G(x,u));
    
    % aux
    Nx = size(x0,1); 
    Ny = size(x0,1);
    
    tout = 1:N;
    
    xout = zeros(Nx,N+1);  
    xout(:,1) = x0;
    
    sout = zeros(Ny,N+1);   
    %sout(:,1) = g(xout(:,1));
    
    % sim
    for k = 1:N
        xout(:,k+1) = f(xout(:,k), u(k,:));
        sout(:,k+1) = g(xout(:,k), u(k,:));
    end
    
    % extract the initial state from the vector that is returned
    xout = xout(:,1:end); 
    sout = sout(:,1:end);
end
