function [Q_ret, R_ret] = qrgivens(A)
import casadi.*

[m,n] = size(A);

AA = MX.sym('AA',m,n);

Q = MX.eye(m);
R = MX(AA);

for j = 1:n
    for i = m:-1:j+1
        [c,s] = givensrotation( R(j,j),R(i,j) );
        G = [c s; -s c];
        R([j i],j:n) = G'*R([j i],j:n);
        R(i,j) = 0; % rewrite small number by zero
        Q(j:n,[j i]) = Q(j:n,[j i])*G;
    end
end

F = Function('F',{AA},{Q,R});
[Q_ret, R_ret] = F(A);

Q_ret = full(Q_ret);
R_ret = full(R_ret);

end

% Givens rotation (Algorithm 5.1.3)
function [c,s] = givensrotation(a,b)
% Givens rotation
import casadi.*

r = if_else(abs(b) > abs(a), -a/b, -b/a);
c = if_else(b == 0, 1, if_else(abs(b) > abs(a), (1/sqrt(1+r^2))*r, 1/sqrt(1+r^2)));
s = if_else(b == 0, 0, if_else(abs(b) > abs(a), 1/sqrt(1+r^2), (1/sqrt(1+r^2))*r));

end