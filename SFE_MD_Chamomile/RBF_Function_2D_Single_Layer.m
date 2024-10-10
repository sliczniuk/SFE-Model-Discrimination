function z = RBF_Function_2D_Single_Layer(x,y,N)
%% Parameters for N RBFs
%cx = opti.variable(N, 1); % Centers of the RBFs in x
%cy = opti.variable(N, 1); % Centers of the RBFs in y
%w = opti.variable(N, 1);  % Weights of the RBFs
%sx = opti.variable(N, 1); % Widths of the RBFs in x (standard deviations)
%sy = opti.variable(N, 1); % Widths of the RBFs in y (standard deviations)
%b = opti.variable();      % Bias term

parameters = readmatrix('KOUT.txt');

%RE            = [0.4632, 0.3783, 0.3029, 0.2619, 0.3579, 0.3140, 0.2635, 0.2323, 0.1787, 0.1160, 0.1889, 0.1512];
%RE            = RE([1,2,4:11]);
%RE            = sort(RE);
%RE            = linspace(min(RE),max(RE),N);
%CF_norm_cent  = linspace(0,1,N);

cx = parameters(1 + (0*N:1*N-1) );
cy = parameters(1 + (1*N:2*N-1) );
w  = parameters(1 + (2*N:3*N-1) );
sx = parameters(1 + (3*N:4*N-1) );
sy = parameters(1 + (4*N:5*N-1) );
b  = parameters(1 + (5*N    ) );

% RBF function
rbf = @(x, y, cx, cy, sx, sy) exp(-((x - cx).^2) / (2 * sx^2) - ((y - cy).^2) / (2 * sy^2));

% Model prediction
z = b;
for i = 1:N
    z = z + w(i) * rbf( x, y, cx(i), cy(i), sx(i), sy(i) );
end