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
R = 10;% 0.01;

x_bar = 10;
u_bar = 1;
x0 = [-4.5; 2];

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

P = eye(2);
N_T = [3, 6, 9];

N = 25;

Qhalf = sqrtm(Q); Rhalf = sqrtm(R); Phalf = sqrtm(P);
xmax = x_bar*ones(n,1); 
xmin = -x_bar*ones(n,1);

umax = u_bar*ones(m,1); 
umin = -u_bar*ones(m,1);

Xallmpc = zeros(n,N+1, length(N_T)); Uallmpc = zeros(m,N, length(N_T));
optvalmpc = zeros(length(N_T), 1);
%% MPC with different time-horizons 
for idx = 1:length(N_T)
    T = N_T(idx);
    % model predictive control w/ different horizons (T) from before

    %store solutions

    x = x0; %reset initial state
    Xallmpc(:,1, idx) = x;


    %step through time
    for i = 1:N
        fprintf('%d, ',i-1);

        %cvx precision
        %cvx_precision(max(min(abs(x))/10,1e-6))
        cvx_precision('default')

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
            Uallmpc(:,i, idx) = u;

            %accumulate cost
            optvalmpc(idx, 1) = optvalmpc(idx, 1) + x'*Q*x + u'*R*u;

            %forward propagate state
            x = A*x+B*u;

            %record state
            Xallmpc(:,i+1, idx) = x;

        else
           % break from loop
           optvalmpc(idx, 1) = Inf;
           break;
        end
    end
    fprintf('\n');
end

%% Plots for MPC

%mpc soln for x1(t) and u(t) w/ T=10
tvec = 0:N;
figure()
subplot(3,1,1);
hold on;
set(gca,'Fontsize',16);
stairs(tvec,Xallmpc(1,:, 1),'k');
stairs(tvec,Xallmpc(1,:, 2),'b');
stairs(tvec,Xallmpc(1,:, 3),'r');
title('MPC State x(1)')
legend('T=3', 'T=6', 'T=9')
axis([0,N,-x_bar,x_bar]); ylabel('x1');
subplot(3, 1,2);
hold on;
set(gca,'Fontsize',16);
stairs(tvec,Xallmpc(2,:, 1),'k');
stairs(tvec,Xallmpc(2,:, 2),'b');
stairs(tvec,Xallmpc(2,:, 3),'r');
title('MPC State x(1)')
legend('T=3', 'T=6', 'T=9')
axis([0,N,-x_bar,x_bar]); ylabel('x2');
subplot(3,1,3); 
hold on;
set(gca,'Fontsize',16);
stairs(tvec(1:end-1),Uallmpc(1,:,1),'k');
stairs(tvec(1:end-1),Uallmpc(1,:,2),'b');
stairs(tvec(1:end-1),Uallmpc(1,:,3),'r');
title('MPC Control u(1)')
legend('T=3', 'T=6', 'T=9')
axis([0,N,-u_bar,u_bar]); xlabel('t'); ylabel('u');