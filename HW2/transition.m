function new_pt = transition(old_pt, act, act_space, p_val)
x = old_pt(1);
y = old_pt(2);
sample = rand;
act_indices = [1, 2, 3, 4];
n = 20;
rand_act = setdiff(act_indices, act);
if sample <= 1 - 3*p_val/4
    x_next = x + act_space{act}(1);
    y_next = y + act_space{act}(2);
else
    rand_sample = rand;
    if rand_sample <= 1/3
        move = rand_act(1);  
    elseif rand_sample > 1/3 && rand_sample <= 2/3
        move = rand_act(2);
    else
        move = rand_act(3);
    end
    x_next = x + act_space{move}(1);
    y_next = y + act_space{move}(2);
end
if x_next < 1 || x_next > n 
    x_next = x;
elseif y_next < 1 || y_next > n
    y_next = y;            
end
new_pt = [x_next, y_next];
end