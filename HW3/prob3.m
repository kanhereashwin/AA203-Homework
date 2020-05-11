clc;
clear;
close all;
%% Defining problem parameters
A = [0, 1; 1, 0];
B = [0; 1];
C = [0,1];
Q_cost = 3*eye(2);
R_cost = 1;
t_final = 15;
Q_kf = [0, 0; 0, 4];
R_kf = 0.5;
%% LQR Controller Gain
x_init = [10; 10]';
p_vect_init = zeros(2, 2);
[t_lqr, p_vect] = ode45(@p3ricatti_diff, [t_final, 0], p_vect_init);
lqr_gain = zeros(size(t_lqr, 1), 2);
for lqr_idx = 1:size(t_lqr, 1)
    P_mat = [p_vect(lqr_idx,1), p_vect(lqr_idx,3); p_vect(lqr_idx,2), p_vect(lqr_idx,4)];
    lqr_gain(lqr_idx, :) = -(1/R_cost)*B'*P_mat;
end
%% Kalman Filter Gain
%Variables to store covariances and gains
t_kf = flip(t_lqr);
sigma_init = zeros(2);%5*eye(2);
[t_kf, sigma_vect] = ode45(@kf_gain_de, t_kf, sigma_init);
kf_gain = zeros(size(t_kf, 1), 2);
for kf_idx = 1:size(t_kf, 1)
    sigma_mat = [sigma_vect(kf_idx, 1), sigma_vect(kf_idx, 3); sigma_vect(kf_idx, 2), sigma_vect(kf_idx,4)];
    kf_gain(kf_idx, :) = sigma_mat*C'/R_kf;
end
%% Plot LQR and KF Gains
figure;
hold on;
plot(t_lqr, -lqr_gain(:, 1) );
plot(t_lqr, -lqr_gain(:, 2));
plot(t_kf, kf_gain(:, 1));
plot(t_kf, kf_gain(:, 2));
legend('LQR_1', 'LQR_2', 'KF_1', 'KF_2')
title('LQR and KF Gains')
xlabel('Time (s)')
ylabel('Gain values')
%% Simulating closed loop system
Ac = zeros(size(t_kf, 1), 2, 2);
Bc = zeros(size(t_kf, 1), 2, 1);
Cc = zeros(size(t_kf, 1), 1, 2);
p_flip = flip(p_vect, 1);
A_cl = zeros(size(t_kf, 1), 4, 4);
for kf_idx = 1:size(t_kf, 1)
    sigma_mat = [sigma_vect(kf_idx, 1), sigma_vect(kf_idx, 3); sigma_vect(kf_idx, 2), sigma_vect(kf_idx,4)];
    P_mat = [p_flip(kf_idx,1), p_flip(kf_idx,3); p_flip(kf_idx,2), p_flip(kf_idx,4)];
    Ac = A - sigma_mat*(C'*C)/R_kf - B*(1/R_cost)*B'*P_mat;
    Bc = sigma_mat*C'/R_kf;
    Cc = (1/R_cost)*B'*P_mat;
    A_cl(kf_idx,:,:) = [A, -B*Cc; Bc*C, Ac];
end
%% Simulating closed loop system, assuming discretization given by solver time vector
init_xcl = [10; -10; 0; 0];
xcl = zeros(size(t_kf, 1) - 1, 4);
xcl(1, :) = init_xcl;
for i = 1:size(t_kf, 1)-1
    dt = t_kf(i+1) - t_kf(i);
    xcl(i+1, :) = (xcl(i, :)' + squeeze(A_cl(i, :, :))*xcl(i, :)'*dt)';
end
figure()
hold on;
plot(t_kf, xcl(:, 1))
plot(t_kf, xcl(:, 2))
plot(t_kf, xcl(:, 3))
plot(t_kf, xcl(:, 4))
legend('x1', 'x2', 'xc1', 'xc2')
title('Evolution of closed loop system')
xlabel('Time (s)')
ylabel('States')

%% Simulating closed loop system with steady state values
init_xcl = [10; -10; 0; 0];
xcl = zeros(size(t_kf, 1) - 1, 4);
xcl(1, :) = init_xcl;

sigma_mat = [sigma_vect(end, 1), sigma_vect(end, 3); sigma_vect(end, 2), sigma_vect(end,4)];
P_mat = [p_flip(end,1), p_flip(end,3); p_flip(end,2), p_flip(end,4)];
Ac = A - sigma_mat*(C'*C)/R_kf - B*(1/R_cost)*B'*P_mat;
Bc = sigma_mat*C'/R_kf;
Cc = (1/R_cost)*B'*P_mat;

A_cl_steady = [A, -B*Cc; Bc*C, Ac];
for i = 1:size(t_kf, 1)-1
    dt = t_kf(i+1) - t_kf(i);
    xcl(i+1, :) = (xcl(i, :)' + A_cl_steady*xcl(i, :)'*dt)';
end
figure()
hold on;
plot(t_kf, xcl(:, 1))
plot(t_kf, xcl(:, 2))
plot(t_kf, xcl(:, 3))
plot(t_kf, xcl(:, 4))
legend('x1', 'x2', 'xc1', 'xc2')
title('Evolution of closed loop system')
xlabel('Time (s)')
ylabel('States')

%% Steady state eigenvalues
eig_kf = eig(Ac);
eig_lqr = eig(A);
eig_cl = eig(A_cl_steady);