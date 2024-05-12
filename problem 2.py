import numpy as np
import matplotlib.pyplot as plt

# data list
hmc_file = open('hmc-observations.csv', "r")
y = []
for l in hmc_file:
    y.append(float(l))
y = np.array(y)
n = len(y)

Sigma = np.diag([5, 0.5])
Sigma_inv = np.diag([0.2, 2])

# Hyperparameters
N = 10000
sig = 2
#

def H(th, v):
    K = 0.5*(np.linalg.norm(v)**2)
    s = 0
    for i in range(n):
        s += (y[i] - (np.linalg.norm(th)**2))**2
    thT_Siginv_th = Sigma_inv[0,0]*(th[0]**2) + Sigma_inv[1,1]*(th[1]**2)
    U = 0.5*(thT_Siginv_th + (s/(sig**2)))
    # print("K, U:", K ,U)
    return K + U


def leapfrog(th, v, L, eps):
    def grad_U(th):
        coeff = (2/(sig**2))*sum(y) - ((2*n*(np.linalg.norm(th)**2))/(sig**2))
        grad_0 = coeff - Sigma_inv[0,0]
        grad_1 = coeff - Sigma_inv[1,1]
        return -np.array([grad_0*th[0], grad_1*th[1]])
    theta_T = th
    v_T = v - 0.5*eps*grad_U(theta_T)
    for m in range(1,L+1):
        theta_T = theta_T + eps*v_T
        if m != L:
            v_T = v_T - eps*grad_U(theta_T)
    v_T = v_T - 0.5*eps*grad_U(theta_T)
    v_T = - v_T
    return theta_T, v_T

def hmc(L, eps):
    theta = np.zeros((N, 2))
    theta[0] = np.random.multivariate_normal(np.zeros(2), Sigma)
    acceptance_rate = 0
    for i in range(1,N):
        print("Current i:", i)
        #print("theta", theta)
        V = np.random.multivariate_normal(np.zeros(2), np.eye(2))
        theta_star, V_star = leapfrog(theta[i-1], V, L, eps)
        # print("leapfrog", theta_star, V_star)
        ratio = np.exp(H(theta[i-1], V) - H(theta_star, V_star))
        print("ratio", ratio)
        alpha = np.min([1, ratio])
        U = np.random.rand()
        if U <= alpha:
            theta[i] = theta_star
            acceptance_rate += 1
        else:
            theta[i] = theta[i-1]
    return theta, acceptance_rate/N

def mh(zeta):
    theta = np.zeros((N, 2))
    theta[0] = np.random.multivariate_normal(np.zeros(2), Sigma)
    acceptance_rate = 0
    for i in range(1,N):
        print("Current i:", i)
        #print("theta", theta)
        theta_star = np.random.multivariate_normal(theta[i-1], zeta*np.eye(2))
        # MH is like HMC with V = [0, 0]
        ratio = np.exp(H(theta[i-1], np.zeros(2)) - H(theta_star, np.zeros(2)))
        alpha = np.min([1, ratio])
        U = np.random.rand()
        if U <= alpha:
            theta[i] = theta_star
            acceptance_rate += 1
        else:
            theta[i] = theta[i-1]
    return theta, acceptance_rate/N

theta_hmc, acceptace_rate_hmc = hmc(L=25, eps=0.25)
# theta_mh, acceptace_rate_mh = mh(zeta=1)

# plt.figure()
# plt.hist2d(theta_hmc[:,0], theta_hmc[:,1], (100, 100))
# plt.show()

# plt.figure()
# plt.hist2d(theta_mh[:,0], theta_mh[:,1], (100, 100))
# plt.show()

print("\nHMC")
print("theta=", theta_hmc[N-1])
print("acceptance_rate=", acceptace_rate_hmc)

# print("\nMH")
# print("theta=", theta_mh[N-1])
# print("acceptance_rate=", acceptace_rate_mh)
