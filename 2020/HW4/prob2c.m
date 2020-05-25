% Solution for problem 2, based off mpc_example.m from AA203 Examples

clc;
clear;
close all;

%% Set random seed
rng(0)

%% generate A, B matrices

n = 2; m = 1;
A = [1 1;
     0 1]; 
B = [0; 1];


%% objective, constraints, initial state, parameters
Q = eye(n); 
R = 0.01;

x_bar = 10;
u_bar = 1;

%% Solving for infinite horizon LQR

L_current = zeros(2);
L_new = ones(2);
P_current = Q;
iter = 1;
delta = 1000;
while delta >= 1e-4
    disp(['At iteration: ', num2str(iter), ' delta: ', num2str(delta)])
    L_new = -(R + B'*P_current*B)\(B'*P_current*A);
    P_new = Q + L_new'*R*L_new + (A+B*L_new)'*P_current*(A+B*L_new);
    delta = norm(P_new - P_current, 2);
    iter = iter + 1;
    P_current = P_new;
end
L = L_new;
P_inf = P_current;

%% 

P = P_inf;
T = 6;

N = 10;

Qhalf = sqrtm(Q); Rhalf = sqrtm(R); Phalf = sqrtm(P);
xmax = x_bar*ones(n,1); 
xmin = -x_bar*ones(n,1);

umax = u_bar*ones(m,1); 
umin = -u_bar*ones(m,1);

n_grid = 11;
lin_x0 = linspace(-x_bar, x_bar, n_grid);

feasibility = ones(n_grid, n_grid);

for x1_idx = 1:length(lin_x0)
    for x2_idx = 1:length(lin_x0)
        %% MPC with different time-horizons 
        x0 = [lin_x0(x1_idx); lin_x0(x2_idx)];
        disp(['Solving for x0 [', num2str(x0(1)), ', ', num2str(x0(2)), ']']);
        % model predictive control w/ different horizons (T) from before
        optvalmpc = 0; 

        %store solutions
        Xallmpc = zeros(n,N+1); Uallmpc = zeros(m,N);
        x = x0; %reset initial state
        Xallmpc(:,1) = x;


        %step through time
        for i = 1:N
            fprintf('%d, ',i-1);

            %cvx precision
            cvx_precision(max(min(abs(x))/100,1e-6))
            %cvx_precision('default')

            cvx_begin quiet
                variables X(n,T+1) U(m,T)
                max(X') <= xmax'; max(U') <= umax';
                min(X') >= xmin'; min(U') >= umin';
                X(:,2:T+1) == A*X(:,1:T)+B*U;
                X(:,1) == x; %initial state constraint
                %X(:,T+1) == 0; %terminal state constraint
                minimize (norm([Qhalf*X(:,1:T); Rhalf*U],'fro') + norm(Phalf*X(:, T+1), 'fro'))
            cvx_end

            %check feasibility
            if strcmp(cvx_status,'Solved')

                %store control
                u= U(:,1);
                Uallmpc(:,i) = u;

                %accumulate cost
                optvalmpc = optvalmpc + x'*Q*x + u'*R*u;

                %forward propagate state
                x = A*x+B*u;

                %record state
                Xallmpc(:,i+1) = x;

            else
               % break from loop
               optvalmpc = Inf;
               feasibility(x1_idx, x2_idx) = 0;
               break;
            end
        end
        fprintf('\n');
    end
end


%% Plots for MPC

%mpc soln for x1(t) and u(t) w/ T=10
figure()
[XX, YY] = meshgrid(lin_x0, lin_x0);
surf(XX, YY, feasibility)
view(0, 90)