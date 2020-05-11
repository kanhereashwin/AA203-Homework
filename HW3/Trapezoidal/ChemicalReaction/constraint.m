% Function providing equality and inequality constraints
% ceq(var) = 0 and c(var) \le 0 

function [c,ceq] = constraint(var)

    global N;
    global T;

    global x0;
    global y0;

    % Put here constraint inequalities
    c = []; %No inequality constraints

    % Note that var = [x;y;u]
    x = var(1:N+1); y = var(N+2:2*N+2); u = var(2*N+3:3*N+3);

    [xdot, ydot] = fDyn(x, y, u);
    % Computing dynamical constraints via the trapezoidal rule
    h = 1.0*T/(1.0*N);
    for i = 1:N
        % Provide here dynamical constraints via the trapeziodal formula
        ceq(i) = x(i+1) - x(i) - h*(xdot(i) + xdot(i+1))/2; % constraint on next x
        ceq(i+N) = y(i+1) - y(i) - h*(ydot(i) + ydot(i+1))/2; %Constraint on next y
    end

    % Put here initial conditions
    ceq(1+2*N) = x(1) - x0;
    ceq(2+2*N) = y(1) - y0;
end