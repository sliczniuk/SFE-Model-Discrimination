function z = RBF_Function_1D(x,N,parameters)
%% Parameters for N RBFs
%cx = opti.variable(N, 1); % Centers of the RBFs in x
%cy = opti.variable(N, 1); % Centers of the RBFs in y
%w = opti.variable(N, 1);  % Weights of the RBFs
%sx = opti.variable(N, 1); % Widths of the RBFs in x (standard deviations)
%sy = opti.variable(N, 1); % Widths of the RBFs in y (standard deviations)
%b = opti.variable();      % Bias term



cx = parameters(44 + (0:N-1) );
%cy = parameters(46:47);
w  = parameters(44 + (N:2*N-1) );
sx = parameters(44 + (2*N:3*N-1) );
%sy = parameters(52:53);
b  = parameters(44 + (3*N)   );

% RBF function
%rbf = @(x, y, cx, cy, sx, sy) exp(-((x - cx).^2) / (2 * sx^2) - ((y - cy).^2) / (2 * sy^2));
rbf = @(x, c, s) exp(-((x - c).^2) / (2 * s^2));

% Model prediction
z = b;
for i = 1:N
    z = z + w(i) * rbf( x, cx(i), sx(i) );
end