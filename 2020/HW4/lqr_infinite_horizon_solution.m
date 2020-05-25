function [L, P] = lqr_infinite_horizon_solution(Q, R)

%% find the infinite horizon L and P through running LQR back-ups
%%   until norm(L_new - L_current, 2) <= 1e-4  
dt = 0.1;
mc = 10; mp = 2.; l = 1.; g= 9.81;

% TODO write A,B matrices
a1 = mp*g/mc;
a2 = (mc+mp)*g/(l*mc);
delf_dels = [0, 0, 1, 0;
             0, 0, 0, 1;
             0, a1, 0, 0;
             0, a2, 0, 0];
delf_delu = [0;
             0;
             1/mc;
             1/(l*mc)];
A = eye(4) + dt*delf_dels;
B = dt* delf_delu;


% Backward Riccati recursion
L_current = zeros(4);
L_new = ones(4);
V_current = Q;
iter = 1;
delta = 1000;
while delta >= 1e-4
    disp(['At iteration: ', num2str(iter), ' delta: ', num2str(delta)])
    L_new = -(R + B'*V_current*B)\(B'*V_current*A);
    V_new = Q + L_new'*R*L_new + (A+B*L_new)'*V_current*(A+B*L_new);
    delta = norm(V_new - V_current, 2);
    iter = iter + 1;
    V_current = V_new;
end
L = L_new;
P = V_current;
end