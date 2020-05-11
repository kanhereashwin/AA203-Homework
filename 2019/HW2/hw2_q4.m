clc;
clear;
close all;
%% Setting up problem parameters
n = 20;
m = 4;
gamma = 0.95;
sigma = 10;
center = [15, 15];
center = center + 1;
goal = [19, 9];
goal = goal +1; % To match convention given in problem with MATLAB
reward = zeros(n ,n);
reward(goal(1), goal(2)) = 1;
act_space = {[-1, 0], [1, 0], [0, -1], [0, 1]};
act_indices = [1, 2, 3, 4];
%% Initializing transition probabilities for each grid point
x = 1:n;
y = 1:n;
[yy, xx] = meshgrid(x, y);
p_vals = exp(-((xx - center(1)).^2 + (yy - center(2)).^2)./(2*sigma^2));
trans_probs = zeros(n, n, m, n, n);
for i = 1:n
    for j = 1:n
        for k = 1:m
            trans_probs(i, j, k, :, :) = 0;
        end
    end
end
size(trans_probs(i,j,k, :, :))
% For each item in the cell, probability of transitioning to another grid
% point is contained
for x = 1:n
    for y = 1:n
        for act = 1:m
            new_x = x+act_space{act}(1);
            new_y = y+act_space{act}(2);
            if new_x > 0 && new_x < n+1 && new_y > 0 && new_y < n+1
                trans_probs(x, y, act, new_x, new_y) = 1 - 3*p_vals(x, y)/4;
            else
                % Action caused you to move out, so stay at the same place
                trans_probs(x, y, act, x, y) = 1 - 3*p_vals(x, y)/4;
            end
            rand_act = setdiff(act_indices, act);
            for j = 1:3
                new_x = x+act_space{rand_act(j)}(1);
                new_y = y+act_space{rand_act(j)}(2);
                if new_x > 0 && new_x < n+1 && new_y > 0 && new_y < n+1
                    trans_probs(x, y, act, new_x, new_y) = p_vals(x, y)/4;
                else
                    trans_probs(x, y, act, x, y) = trans_probs(x, y, act, x, y) + p_vals(x, y)/4;
                end
            end
        end
    end
end
% Modify transition probabilities for goal to all zeros
for act = 1:m
    trans_probs(goal(1), goal(2), act, :, :) = 0;
end


%% Value Iteration
% Initialize value iteration parameters
val = zeros(n,n);
Q = zeros(n, n, m);
delta = 10000; % Termination criterion
tol = 1e-4;
v_old = val;
v_new = 100*ones(n,n);
policy = zeros(n,n);
iter = 1;
% Run value iteration with synchronous updates to Q matrix
while delta >= tol
    disp(['At iteration ', num2str(iter), ' delta ', num2str(delta)])
    for x = 1:n %columns
        for y = 1:n %rows

            for act = 1:m
                Q(x, y, act) = sum(sum(squeeze(trans_probs(x, y, act, : ,:)).*(reward + gamma*v_old)));
            end
            [v_new(x, y), policy(x,y)] = max(Q(x, y, :));
        end
    end
    % Reset optimal policy and value at goal state to reward and stay still
    v_new(goal(1), goal(2)) = 0;
    policy(goal(1), goal(2)) = 0;
    delta = norm(v_new - v_old);
    iter = iter + 1;
    v_old = v_new;
end
v_new(goal(1), goal(2))=1;
str_policy = strings(n, n);
str_policy(policy == 0) = "stay";
str_policy(policy == 1) = "up";
str_policy(policy == 2) = "down";
str_policy(policy == 3) = "left";
str_policy(policy == 4) = "right";
%% Plotting
% Show heat map of value function
figure;
h = heatmap(v_new);
% Show optimal policy as a result of value iteration
u = zeros(n, n);
v = zeros(n, n);
for i = 1:m
    u(policy==i) = act_space{i}(1);
    v(policy==i) = act_space{i}(2);
end
u(goal(1), goal(2)) = 0;
u(goal(1), goal(2)) = 0;
figure;
quiver(yy, -xx, v, -u)
%% Trajectory Generation
% Generating policy and trajectory as a result of trajectory
% Including storm disturbances in the sampled trajectory
% Plot a trajectory heatmap
start = [9, 19];
start = start + 1;
point_count = zeros(n, n);
for i = 1:20
    disp(['Trajectory iteration ', num2str(i)])
    figure;
    hold on;
    curr_point = start;
    while sum(curr_point == goal)~=2
        %curr_point
        point_count(curr_point(1), curr_point(2)) = point_count(curr_point(1), curr_point(2))+1;
        next_point = transition(curr_point, policy(curr_point(1), curr_point(2)), act_space, p_vals(curr_point(1), curr_point(2)));
        plot([curr_point(2), next_point(2)], [-curr_point(1), -next_point(1)]);
        curr_point = next_point;
    end
    point_count(curr_point(1), curr_point(2)) = point_count(curr_point(1), curr_point(2))+1;
    disp('Trajectory done')
end
scatter([goal(2), start(2)], [-goal(1), -start(1)]);
xlim([0,n]);
ylim([-n,0]);
figure;
heatmap(point_count)
