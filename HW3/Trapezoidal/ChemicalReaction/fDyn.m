% Dynamics of the problem

function [xDyn,yDyn] = fDyn(x,y,u)

    % Put here the dynamics
    xDyn = -x.*u +y.*u.^2;
    yDyn = x.*u - 3*y.*u.^2;
end