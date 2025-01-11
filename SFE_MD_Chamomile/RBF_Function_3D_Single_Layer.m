function f_out = RBF_Function_3D_Single_Layer(x,y,z,N,parameters)
    %% Parameters for N RBFs
    % Parameters for N RBFs
    %cx = opti.variable(N, 1); % Centers of the RBFs in x
    %cy = opti.variable(N, 1); % Centers of the RBFs in y
    %cz = opti.variable(N, 1); % Centers of the RBFs in z
    %sx = opti.variable(N, 1); % Widths of the RBFs in x (standard deviations)
    %sy = opti.variable(N, 1); % Widths of the RBFs in y (standard deviations)
    %sz = opti.variable(N, 1); % Widths of the RBFs in z (standard deviations)
    %w = opti.variable(N, 1);  % Weights of the RBFs
    %b = opti.variable();      % Bias term

    %parameters = readmatrix('KOUT.txt');
    
    %% Parameters for N RBFs
    cx = parameters(50 + (0*N :1*N-1)); % Centers of the RBFs in x
    cy = parameters(50 + (1*N :2*N-1)); % Centers of the RBFs in y
    cz = parameters(50 + (2*N :3*N-1)); % Centers of the RBFs in z
    w  = parameters(50 + (3*N :4*N-1)); % Weights of the RBFs
    sx = parameters(50 + (4*N :5*N-1)); % Widths of the RBFs in x (standard deviations)
    sy = parameters(50 + (5*N :6*N-1)); % Widths of the RBFs in y (standard deviations)
    sz = parameters(50 + (6*N :7*N-1)); % Widths of the RBFs in z (standard deviations)
    %w  = parameters(44 + (6*N :7*N-1)); % Weights of the RBFs
    b  = parameters(50 + (7*N+0     )); % Bias term
    
    %%
    f_out = model_prediction(x, y, z, cx, cy, cz, sx, sy, sz, w, b, N) ;
    
    %% Auxilary functions
    % RBF function
    function rbf_value = rbf(x, y, z, cx, cy, cz, sx, sy, sz)
        rbf_value = exp(-((x - cx).^2) ./ sx - ((y - cy).^2) ./ sy - ((z - cz).^2) ./ sz );
    end
    
    % Model prediction function
    function f = model_prediction(x, y, z, cx, cy, cz, sx, sy, sz, w, b, N)
        f = b;
        for i = 1:N
            f = f + w(i) * rbf(x, y, z, cx(i), cy(i), cz(i), sx(i), sy(i), sz(i));
        end
    end
end