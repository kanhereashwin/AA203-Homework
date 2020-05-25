clc;
clear;
close all;
A = [1 1 0;
    0 0.9 1
    0 0.2 0];
B = [0 ; 1; 0];
xlb = [-5; -5; -5];
xub = [5; 5; 5];
ulb = -0.5;
uub = 0.5;

system = LTISystem('A', A, 'B', B);
system.x.min = xlb;
system.x.max = xub;
system.u.min = ulb;
system.u.max = uub;
InvSet = system.invariantSet();
InvSet.plot()