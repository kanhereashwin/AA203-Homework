from model import dynamics, cost
import numpy as np
from matplotlib import pyplot as plt
from part_a import Riccati


stochastic_dynamics = False # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()


T = 100
N = 10000 


L_star, _ = Riccati(dynfun.A, dynfun.B, costfun.Q, costfun.R)
L_err_plot = np.empty([N])

alpha = 1.0e-13
gamma = 0.95 # discount factor
W = np.zeros([2, 4])

total_costs = []
W = np.zeros([2, 4])
Sigma = 0.1*np.eye(2)
for n_itr in range(N):
    costs = []
    states = []
    controls = []
    
    x = dynfun.reset()
    states.append(x)
    for t in range(T):

        # TODO compute action
        u = np.random.multivariate_normal(W @ x, Sigma)


        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
        
        # dynamics step
        xp = dynfun.step(u)
        states.append(xp)
        controls.append(u)


        x = xp.copy()

    # TODO update policy

    for t in range(T):
        #print('t is ', t)
        #print('The shape of the vector being added to the costs is ', np.shape(costs[t:]))
        G = 0
        G = gamma**(-t)*np.sum(costs[t:])
        x_t= np.reshape(states[t], [-1, 1])
        u_t = np.reshape(controls[t], [-1, 1])
        sigma_inv = np.linalg.inv(Sigma)
        log_grad = sigma_inv @ u_t @ x_t.T - sigma_inv @ (W @ x_t) @ x_t.T 
        W = W - alpha*G*log_grad
        #input()

    L_err_plot[n_itr] = np.linalg.norm(L_star - W)
    
    print('Cost at iteration ', n_itr, 'is ', sum(costs))
    print('Error at iteration ', n_itr, 'is ', np.mean(L_err_plot[n_itr]))

    if np.isnan(sum(costs)):
        break

    total_costs.append(sum(costs))


plt.figure()
plt.plot(np.arange(0, N), L_err_plot)
plt.title('Error in control gain (L) estimate, compared to optimal (L*)')
plt.xlabel('Episode number')
plt.figure()
plt.plot(total_costs)
plt.title('Cost incurred in each episode vs episode number')
plt.ylabel('Cost')
plt.xlabel('Episode number')
plt.show()

