function z = RBF_Function_2D_Single_Layer(x,y,N)
%% Parameters for N RBFs
%cx = opti.variable(N, 1); % Centers of the RBFs in x
%cy = opti.variable(N, 1); % Centers of the RBFs in y
%w = opti.variable(N, 1);  % Weights of the RBFs
%sx = opti.variable(N, 1); % Widths of the RBFs in x (standard deviations)
%sy = opti.variable(N, 1); % Widths of the RBFs in y (standard deviations)
%b = opti.variable();      % Bias term

parameters = readmatrix('KOUT.txt');

cx = parameters(1 + (0*N:1*N-1) );
cy = parameters(1 + (1*N:2*N-1) );
w  = parameters(1 + (2*N:3*N-1) );
sx = parameters(1 + (3*N      ) );
sy = parameters(1 + (3*N+1    ) );
b  = parameters(1 + (3*N+2    ) );

% RBF function
rbf = @(x, y, cx, cy, sx, sy) exp(-((x - cx).^2) / (2 * sx^2) - ((y - cy).^2) / (2 * sy^2));

% Model prediction
z = b;
for i = 1:N
    z = z + w(i) * rbf( x, y, cx(i), cy(i), sx, sy );
end