function F = buildIntegrator(varargin)
    
    import casadi.*

    f = varargin{1};
    d = varargin{2};
    
    % Verify if additional arguments were provided
    
    options = struct;
    
    if(numel(varargin) > 2)
        options.tf = varargin{3};
    else
        options.tf = 1;     
    end
    
    if(numel(varargin) > 3)
        method = varargin{4};     
    else                    
        method = 'cvodes';     
    end
    
    x = MX.sym('x',d(1));
    u = MX.sym('u',d(2));
    p = [u];
    
    DAE = struct('x', x, 'p', p, 'ode', f(x,u));
    args = {x,u};
    args_str = {'x','u'};

    
    F = integrator('F', method, DAE, options);
    F_res = F('x0', x, 'p', p);
    
    F = Function('F', args, {F_res.xf}, args_str, {'x_next'});
    
end