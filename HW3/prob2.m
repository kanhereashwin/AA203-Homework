clc;
clear;
close all;
%% Set problem parameters
A = [0, 1; 0, -1];
B = [0; 1];
QN = [0, 0; 0, 4];
Q = [1, 0; 0, 0];
r = 3;
t_final = 100;
x_init = [1; 1]';
p_vect_init = [QN(1); QN(2); QN(3); QN(4)];
[t, p_vect] = ode45(@ricatti_diff, [t_final, 0], p_vect_init);
%% Plotting obtained gains
n = size(p_vect, 1);
K = zeros(n, 2);
for i = 1:n
    P_mat = [p_vect(i, 1), p_vect(i, 3); p_vect(i, 2), p_vect(i, 4)];
    K(i, :) = -B'*P_mat/r;
end
figure;
hold on;
plot(t, K(:, 1));
plot(t, K(:, 2));
legend('K1', 'K2')
title('Gain matrix for LQR plotted against time')
xlabel('Time')
ylabel('Gain values')
%% Some debugging
figure
hold on
plot(t, p_vect(:, 1))
plot(t, p_vect(:, 2))
plot(t, p_vect(:, 3))
plot(t, p_vect(:, 4))
legend('P11', 'P21', 'P12', 'P22')
title('P matrix for LQR plotted against time')
xlabel('Time')
ylabel('Elements of P matrix')