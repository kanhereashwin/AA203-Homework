function dpdt = ricatti_diff(t, p)
%% Problem parameters
A = [0, 1; 0, -1];
B = [0; 1];
Q = [1, 0; 0, 0];
r = 3;
%% Constructing matrix from vector values
P = [p(1), p(3); p(2), p(4)];
dPdt = -(Q - P*B*B'*P/r + P*A +A'*P);
dpdt = [dPdt(1); dPdt(2); dPdt(3); dPdt(4)];
end