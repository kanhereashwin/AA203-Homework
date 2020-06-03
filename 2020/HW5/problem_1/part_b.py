from model import dynamics, cost
from part_a import Riccati
import numpy as np
from matplotlib import pyplot as plt

def Riccati_few_steps(A,B,Q,R):

    # TODO implement infinite horizon riccati recursion
    n = np.shape(A)[0]
    m = np.shape(B)[1]
    V_old = np.zeros([n, n])
    V_new = np.zeros_like(V_old)
    L = np.zeros([m, n])
    delta = 1000
    eps = 1e-4
    V_old = Q
    iter = 0
    while delta > eps and iter < 50:
        L = np.linalg.inv(R + B.T @ V_old @ B) @ (B.T @ V_old @ A)
        temp = A - B @ L
        V_new = Q + (L.T @ R @ L) + (temp.T @ V_old @ temp)
        delta = np.linalg.norm(V_new - V_old)
        V_old = np.copy(V_new)
        iter += 1
    P = V_new
    return L,P

def model_estimate(P_old_dyn, x, u, xp, A_est, B_est):
    z = np.vstack((np.reshape(x, [-1, 1]), np.reshape(u, [-1, 1])))
    F_old = np.transpose(np.hstack((A_est, B_est)))
    P_dyn = P_old_dyn - (P_old_dyn @ z @ z.T @ P_old_dyn)/(1 + z.T @ P_old_dyn @ z)
    F_new = F_old + (P_old_dyn @ z) @ (xp - z.T @ F_old)/(1 + z.T @ P_old_dyn @ z)

    F_new = np.transpose(F_new)

    A_est_new = F_new[:, :4]
    B_est_new = F_new[:, 4:6]
    return [A_est_new, B_est_new, P_dyn]


def cost_estimate(P_old_cost, x, u, c, Q_est, R_est):
    # TODO: Fix this 
    x_cost = np.reshape(x, [-1, 1])
    u_cost = np.reshape(u, [-1, 1])
    z_cost = np.vstack((x[0]*x_cost, x[1]*x_cost, x[2]*x_cost, x[3]*x_cost, u[0]*u_cost, u[1]*u_cost))

    C_old = np.transpose(np.hstack((np.reshape(Q_est, [1, -1]), np.reshape(R_est, [1, -1]))))

    P_cost = P_old_cost - (P_old_cost @ z_cost @ z_cost.T @ P_old_cost)/(1 + z_cost.T @ P_old_cost @ z_cost)
    C_new = C_old + (P_old_cost @ z_cost) @ (c - z_cost.T @ C_old)/(1 + z_cost.T @ P_old_cost @ z_cost)
    C_new = np.transpose(C_new)
    Q_est_new = np.reshape(C_new[0, :16], [4, 4])
    R_est_new = np.reshape(C_new[0, 16:], [2, 2])
    return [Q_est_new, R_est_new, P_cost]


stochastic_dynamics = False # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

T = 100 # episode length
N = 100 # number of episodes
gamma = 0.95 # discount factor

total_costs = []


L_star, _ = Riccati(dynfun.A, dynfun.B, costfun.Q, costfun.R)
L_err_plot = np.empty([N, T])

for n_itr in range(N):
    # print('Running iteration ', n)
    costs = []
    x = dynfun.reset()
    n = 4
    m = 2
    A_est = np.random.randn(n,n)
    B_est = np.random.randn(n,m)
    Q_est = np.eye(n)
    R_est = np.eye(m)
    P_dyn = np.eye(6)
    P_cost = np.eye(20)
    for t in range(T):
        
        # TODO compute policy
        L, _ = Riccati_few_steps(A_est, B_est, Q_est, R_est)
        # TODO compute action
        u = (-L @ x)
        
        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
        
        # dynamics step
        xp = dynfun.step(u)
        
        # TODO implement recursive least squares update
        [A_est, B_est, P_dyn] = model_estimate(P_dyn, x, u, xp, A_est, B_est)
        [Q_est, R_est, P_cost] = cost_estimate(P_cost, x, u, c, Q_est, R_est)
        
        #[A_est, B_est, Q_est, R_est, P_dyn, P_cost] = estimate_model(P_old_dyn, P_old_cost, x, u, c, xp, A_est, B_est, Q_est, R_est)

        L_err_plot[n_itr, t] = np.linalg.norm(L_star - L)


        x = xp.copy()
        
    total_costs.append(sum(costs))
    print('Cost at iteration ', n_itr, 'is ', sum(costs))
    print('Error at iteration ', n_itr, 'is ', np.mean(L_err_plot[n_itr, :]))

print(np.mean(total_costs))

plt.figure()
plt.plot(np.arange(0, T), np.mean(L_err_plot, axis=0))
plt.title('Error in control gain (L) estimate, compared to optimal (L*)')
plt.xlabel('Time (s)')
plt.ylim([0, 10])
plt.figure()
plt.plot(total_costs)
plt.title('Cost incurred in each episode vs episode number')
plt.ylabel('Cost')
plt.xlabel('Episode number')
plt.show()
