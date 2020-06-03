from model import dynamics, cost
import numpy as np

dynfun = dynamics(stochastic=True)
# dynfun = dynamics(stochastic=True) # uncomment for stochastic dynamics

costfun = cost()


T = 100 # episode length
N = 100 # number of episodes
gamma = 0.95 # discount factor

# Riccati recursion
def Riccati(A,B,Q,R):

    # TODO implement infinite horizon riccati recursion
    n = np.shape(A)[0]
    m = np.shape(B)[1]
    V_old = np.zeros([n, n])
    V_new = np.zeros_like(V_old)
    L = np.zeros([m, n])
    delta = 1000
    eps = 1e-4
    V_old = Q
    while delta > eps:
        L = np.linalg.inv(R + B.T @ V_old @ B) @ (B.T @ V_old @ A)
        temp = A - B @ L
        V_new = Q + (L.T @ R @ L) + (temp.T @ V_old @ temp)
        delta = np.linalg.norm(V_new - V_old)
        V_old = np.copy(V_new)
    P = V_new
    return L,P


A = dynfun.A
B = dynfun.B
Q = costfun.Q
R = costfun.R

L,P = Riccati(A,B,Q,R)

total_costs = []

for n in range(N):
    costs = []
    
    x = dynfun.reset()
    for t in range(T):
        
        # policy 
        u = (-L @ x)
        
        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
    
        # dynamics step
        x = dynfun.step(u)
        
    total_costs.append(sum(costs))
    
print(np.mean(total_costs))