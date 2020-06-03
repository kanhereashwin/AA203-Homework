from model import dynamics, cost
from part_a import Riccati
import numpy as np
from matplotlib import pyplot as plt


stochastic_dynamics = False # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

T = 100
N = 100
gamma = 0.95 # discount factor

total_costs = []
L_star, _ = Riccati(dynfun.A, dynfun.B, costfun.Q, costfun.R)
L_err_plot = np.empty([N])

H = np.random.randn(6, 6)
H = 0.5*(H + H.T)
theta = np.reshape(H, [36, 1])
U = -np.linalg.inv(H[4:, 4:]) @ H[4:, :4]
L = -U

for n_itr in range(N):
    costs = []
    
    P = 1000*np.eye(36)
    



    x = dynfun.reset()
    for t in range(T):
        
        # TODO compute action
        epsilon = np.random.randn(2)
        u = -L @ x + epsilon
        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
        
        # dynamics step
        xp = dynfun.step(np.reshape(u, 2))

        # TODO recursive least squares policy self evaluation step
        x_cost = np.reshape(x, [-1, 1])
        u_cost = np.reshape(u, [-1, 1])
        xu_aug = np.vstack((x_cost, u_cost))
        z_cost = np.reshape(np.outer(xu_aug, xu_aug), [-1, 1])

        xp_cost = np.reshape(xp, [-1, 1])
        up_cost = np.reshape(U @ xp, [-1, 1])
        xup_aug = np.vstack((xp_cost, up_cost))
        zp_cost = np.reshape(np.outer(xup_aug, xup_aug), [-1, 1])

        phi = z_cost - gamma*zp_cost

        e = c - phi.T @ theta
        theta = theta + (P @ phi @ e)/(1 + phi.T @ P @ phi)
        P = P - (P @ phi @ phi.T @ P)/(1 + phi.T @ P @ phi)
        H = np.reshape(theta, [6, 6])
        H = 0.5*(H + H.T)
        theta = np.reshape(H, [36, 1])
        x = xp.copy()
    
    # TODO policy improvement step

    U = -np.linalg.inv(H[4:, 4:]) @ H[4:, :4]
    L = -U

    L_err_plot[n_itr] = np.linalg.norm(L_star - L)
    
    total_costs.append(sum(costs))
    print('Cost at iteration ', n_itr, 'is ', sum(costs))
    print('Error at iteration ', n_itr, 'is ', np.mean(L_err_plot[n_itr]))


print(np.mean(total_costs))


plt.figure()
plt.plot(np.arange(0, N), L_err_plot)
plt.title('Error in control gain (L) estimate, compared to optimal (L*)')
plt.xlabel('Episode number')
plt.ylim([0, 10])
plt.figure()
plt.plot(total_costs)
plt.title('Cost incurred in each episode vs episode number')
plt.ylabel('Cost')
plt.xlabel('Episode number')
plt.show()
