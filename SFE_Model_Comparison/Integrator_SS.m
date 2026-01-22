function Results = Integrator_SS(Time, x0, S, p, Sdot, parameters)
    import casadi.*

    Results = MX(length(S),length(Time));
    Results(:,1) = x0;
    
    timeStep_in_seconds = Time(2)-Time(1);
    
    dae = struct('x', S, 'p', p, 'ode', Sdot);

    MyIntegrator = 'cvodes';

    opts = struct();
    opts.tf = timeStep_in_seconds;
    %opts.monitor = '1';
    %opts.nonlinear_solver_iteration = 'functional';
    %opts.abstol  = 1e-10;
    %opts.reltol  = 1e-8;

    I = casadi.integrator('I', MyIntegrator, dae, opts);

    %% Integrator
    for i=1:(length(Time)-1)
        
        Q = Results(:,i);

        res = I('x0', Q, 'p', parameters(i,:) );
        Results(:,i+1) = full(res.xf.');

        if(mod(Time(i)/Time(end),0.05)==0)
                fprintf("Finished: %g [%%]\n",[i/length(Time)*100])
        end

    end


end