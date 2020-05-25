function state_dot = prob5_cl(t, state)

a = -1;
b = 3;
am = 4;
bm = 4;
gamma = 2;

r = 4*sin(3*t);

kr_star = bm/b;
ky_star = (a - am)/b;

y = state(1);
e = state(2);
delta_r= state(3);
delta_y = state(4);

kr = delta_r + kr_star;
ky = delta_y + ky_star;
state_dot = zeros(4, 1);
state_dot(1) = -a*y + b* (kr*r + ky*y);
state_dot(2) = -am*e + b*(delta_r*r + delta_y*y);
state_dot(3) = -sign(b)*gamma*e*r;
state_dot(4) = -sign(b)*gamma*e*y;

end

