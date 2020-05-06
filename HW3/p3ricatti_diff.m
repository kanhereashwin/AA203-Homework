function dpdt = p3ricatti_diff(t, p)
%% Problem parameters
A = [0, 1; 1, 0];
B = [0; 1];
Q_cost = 3*eye(2);
R_cost = 1;
%% Constructing matrix from vector values
P = [p(1), p(3); p(2), p(4)];
dPdt = -(Q_cost - P*(B*B')*P/R_cost + P*A +A'*P);
dpdt = [dPdt(1); dPdt(2); dPdt(3); dPdt(4)];
end