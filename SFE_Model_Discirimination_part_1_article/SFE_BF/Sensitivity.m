function [S,p,Sdot] = Sensitivity(x, xdot, theta, which_theta)

    %addpath('\\home.org.aalto.fi\sliczno1\data\Documents\casadi-windows-matlabR2016a-v3.5.1');
    import casadi.*

    s = MX.sym('s',[numel(xdot),numel(which_theta)]);                        % size of sensitivity matrix = xdot * (num of p + num of u)

    Jtheta_p =  jacobian(xdot,theta);                                        % find jacobian of xdot with respect to all p
    Jtheta   =  Jtheta_p(:,which_theta);                                     % select jacobians related to specific p from which_theta
    
    Sdot = jacobian(xdot,x)*s + Jtheta;   
    Sdot = [xdot; Sdot(:)];

    %% Create a function of all equations 
    S = [x(:);s(:)];
    p = [theta(:)];

end