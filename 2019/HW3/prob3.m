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

%% Kalman Filter Gain
%Variables to store covariances and gains
t_kf = flip(t_lqr);
sigma_init = 5*[1, 0; 0, 1];
[t_kf, sigma_kf] = ode45(@kf_gain_de, t_kf, sigma_init);

