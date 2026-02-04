function z = RBF_Function_2D_Double_Layer_Double_Hidden(x,y,N,parameters)
    
    import casadi.*

    %% Parameters for N RBFs
    %cx = opti.variable(N, 1); % Centers of the RBFs in x
    %cy = opti.variable(N, 1); % Centers of the RBFs in y
    %w = opti.variable(N, 1);  % Weights of the RBFs
    %sx = opti.variable(N, 1); % Widths of the RBFs in x (standard deviations)
    %sy = opti.variable(N, 1); % Widths of the RBFs in y (standard deviations)
    %b = opti.variable();      % Bias term

    N1 = N;
    N2 = N;
    
    cx1 = parameters(44 + (0*N :1*N-1) );
    cy1 = parameters(44 + (1*N :2*N-1) );
    w11 = parameters(44 + (2*N :3*N-1) );
    w12 = parameters(44 + (3*N :4*N-1) );
    sx1 = parameters(44 + (4*N :5*N-1) );
    sy1 = parameters(44 + (5*N :6*N-1) );
    
    cx2 = parameters(44 + (6*N :7*N-1) );
    cy2 = parameters(44 + (7*N :8*N-1) );
    w2  = parameters(44 + (8*N :9*N-1) );
    sx2 = parameters(44 + (9*N :10*N-1) );
    sy2 = parameters(44 + (10*N:11*N-1) );
    
    b  = parameters(44 + (11*N)   );
    
    z = model_prediction(x, y, cx1, cy1, sx1, sy1, w11, w12, cx2, cy2, sx2, sy2, w2, b, N1, N2);
    
    %% RBF function
    function rbf_value = rbf(x, y, cx, cy, sx, sy)
        rbf_value = exp(-((x - cx).^2) ./ (2 * sx.^2) - ((y - cy).^2) ./ (2 * sy.^2));
    end
    
    % First hidden layer output function
    function [h1_1, h1_2] = hidden_layer1(x, y, cx1, cy1, sx1, sy1, w1_1, w1_2, N1)
        h1_1 = zeros(size(x, 1), 1);
        h1_2 = zeros(size(x, 1), 1);
        for i = 1:N1
            h1_1 = h1_1 + w1_1(i) * rbf(x, y, cx1(i), cy1(i), sx1(i), sy1(i));
            h1_2 = h1_2 + w1_2(i) * rbf(x, y, cx1(i), cy1(i), sx1(i), sy1(i));
        end
    end
    
    % Second hidden layer output function
    function z = hidden_layer2(h1_1, h1_2, cx2, cy2, sx2, sy2, w2, b, N2)
        z = b;
        for i = 1:N2
            z = z + w2(i) * rbf(h1_1, h1_2, cx2(i), cy2(i), sx2(i), sy2(i));
        end
    end
    
    % Model prediction function
    function z = model_prediction(x, y, cx1, cy1, sx1, sy1, w1_1, w1_2, cx2, cy2, sx2, sy2, w2, b, N1, N2)
        [h1_1, h1_2] = hidden_layer1(x, y, cx1, cy1, sx1, sy1, w1_1, w1_2, N1);
        z = hidden_layer2(h1_1, h1_2, cx2, cy2, sx2, sy2, w2, b, N2);
    end



end