clc;
clear;
close all;

a = -1;
b = 3;
am = 4;
bm = 4;
gamma = 2;
r = 4;%*sin(3*t);

kr_star = bm/b;
ky_star = (a - am)/b;

%% Solve ODE to get y, e, delta_r and delta_y
states_0 = [0; 0; -kr_star; -ky_star];
[t, states] = ode45(@prob5_cl, [0, 20], states_0);

%% Plot y and ym
figure()
hold on;
plot(t, states(:, 1))
plot(t, states(:, 1) - states(:, 2))
xlabel('t (s)')
ylabel('States')
legend('y', 'ym')
title('Evolution of true system and reference system with time')
%% Plot kr, kr_star, ky and ky_star
figure()
hold on;
plot(t, kr_star* ones(size(t)))
plot(t, kr_star + states(:, 3))
plot(t, ky_star* ones(size(t)))
plot(t, ky_star + states(:, 4))
xlabel('t (s)')
ylabel('Gains')
legend('k_r*', 'k_r', 'k_y*', 'k_y')
title('Evolution of ideal and estimated control gains with time')