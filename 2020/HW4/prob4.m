clc;
clear;
close all;
%% Part a)
A = [0.99 1;
     0 0.99];
B = [0 ; 1];
xlb = [-5; -5];
xub = [5; 5];
ulb = -0.5;
uub = 0.5;

Q = eye(2);
R = 1;

% Calculating the infinite horizon LQR P and F (gain) to find Xf and P

% Backward Riccati recursion
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
P = P_current;

% For the obtained LQR gain, the control limits must not be violated

tic; 
system = LTISystem('A', A, 'B', B);
system.x.min = xlb;
system.x.max = xub;
system.u.min = ulb;
system.u.max = uub;

system.x.penalty = QuadFunction( Q );
system.u.penalty = QuadFunction( R );

Tset = system.invariantSet();
P_mpt = system.LQRPenalty;
Tset.plot();


%% Part b)
N = 4;

system.x.with('terminalPenalty');
system.x.with('terminalSet');

system.x.terminalPenalty = P_mpt;
system.x.terminalSet = Tset;

x0 = [-4.7; 2];
Tf = 15;

t_vec = 0:1:Tf;

mpc = MPCController(system, N);
toc;
tic;
loop = ClosedLoop(mpc, system);
data = loop.simulate(x0, Tf);
toc;
x = data.X;
u = data.U;
figure();
subplot(3, 1, 1)
stairs(t_vec, x(1, :))
title('Online MPC State x(1)')
subplot(3, 1, 2)
stairs(t_vec, x(2, :))
title('Online MPC State x(2)')
subplot(3, 1, 3)
stairs(t_vec(1:Tf), u)
title('Online MPC Control')

%% Part c)
tic;
expmpc = mpc.toExplicit();
toc;
tic;
loop_exp = ClosedLoop(expmpc, system);

Tf = 15;

data = loop_exp.simulate(x0, Tf);
toc;
% figure()
% plot(data.X');
x = data.X;
u = data.U;
figure();
subplot(3, 1, 1)
stairs(t_vec, x(1, :))
title('Explicit MPC State x(1)')
subplot(3, 1, 2)
stairs(t_vec, x(2, :))
title('Explicit MPC State x(2)')
subplot(3, 1, 3)
stairs(t_vec(1:Tf), u)
title('Explicit MPC Control')
toc

%% Part d)
figure()
expmpc.partition.plot()