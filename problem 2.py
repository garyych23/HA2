import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
import matplotlib.pyplot as plt

np.random.seed(9)

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
    return K + U


def leapfrog(th, v, L, eps):
    def grad_U(th):
        coeff = 2*((sum(y)/(sig**2)) - ((n*(np.linalg.norm(th)**2))/(sig**2)))
        grad_0 = coeff - Sigma_inv[0,0]
        grad_1 = coeff - Sigma_inv[1,1]
        return -np.array([grad_0*th[0], grad_1*th[1]])
    theta_T = th.copy()
    v_T = (v - 0.5*eps*grad_U(theta_T)).copy()
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
        V = np.random.multivariate_normal(np.zeros(2), np.eye(2))
        theta_star, V_star = leapfrog(theta[i-1], V, L, eps)
        ratio = np.exp(H(theta[i-1], V) - H(theta_star, V_star))
        alpha = np.min([1, ratio])
        U = np.random.rand()
        if U <= alpha:
            theta[i] = theta_star
            acceptance_rate += 1
        else:
            theta[i] = theta[i-1]
    return theta, acceptance_rate/(N-1)

def mh(zeta):
    theta = np.zeros((N, 2))
    theta[0] = np.random.multivariate_normal(np.zeros(2), Sigma)
    acceptance_rate = 0
    for i in range(1,N):
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
    return theta, acceptance_rate/(N-1)

# Question i

print("hmc for L= 100 and eps= 0.12")
theta_hmc, acceptance_rate_hmc = hmc(L=100, eps=0.12)

# plt.figure()
# plt.hist2d(theta_hmc[:,0], theta_hmc[:,1], (100, 100))
# plt.show()

# print("\nHMC")
# print("theta=", theta_hmc[N-1])
# print("acceptance_rate=", acceptance_rate_hmc)

# # plot autocorrelation function
# fig = tsaplots.plot_acf(theta_hmc[:,0])
# plt.show()
# fig = tsaplots.plot_acf(theta_hmc[:,1])
# plt.show()

print("mh for zeta= 0.1")
theta_mh, acceptance_rate_mh = mh(zeta= 0.1)

# plt.figure()
# plt.hist2d(theta_mh[:,0], theta_mh[:,1], (100, 100))
# plt.show()

# print("\nMH")
# print("theta=", theta_mh[N-1])
# print("acceptance_rate=", acceptance_rate_mh)

# # plot autocorrelation function
# fig = tsaplots.plot_acf(theta_mh[:,0])
# plt.show()
# fig = tsaplots.plot_acf(theta_mh[:,1])
# plt.show()

# Question ii
# varying epsilon
print("\nVarying epsilon")
eps_list = [0.06, 0.12, 0.24]
fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i in range(3):
    if i==1:
        ax[i].set_title(f'$\\epsilon$ = {eps_list[i]}')
        ax[i].hist2d(theta_hmc[:,0], theta_hmc[:,1], (100, 100))
    else :
        print("hmc for L= 100 and eps=", eps_list[i])
        th_hmc, _ = hmc(L=100, eps=eps_list[i])
        ax[i].set_title(f'$\\epsilon$ = {eps_list[i]}')
        ax[i].hist2d(th_hmc[:,0], th_hmc[:,1], (100, 100))
plt.show()

# varying L
print("\nVarying L")
L_list = [10, 100, 1000]
fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i in range(3):
    if i==1:
        ax[i].set_title(f'$L$ = {L_list[i]}')
        ax[i].hist2d(theta_hmc[:,0], theta_hmc[:,1], (100, 100))
    else :
        print("hmc for L=", L_list[i], "and eps= 0.12")
        th_hmc, _ = hmc(L=L_list[i], eps=0.12)
        ax[i].set_title(f'$L$ = {L_list[i]}')
        ax[i].hist2d(th_hmc[:,0], th_hmc[:,1], (100, 100))
plt.show()

# varying zeta
print("\nVarying zeta")
zeta_list = [0.001, 0.1, 10]
fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i in range(3):
    if i==1:
        ax[i].set_title(f'$\\zeta$ = {zeta_list[i]}')
        ax[i].hist2d(theta_mh[:,0], theta_mh[:,1], (100, 100))
    else :
        print("mh for zeta=", zeta_list[i])
        th_mh, _ = mh(zeta=zeta_list[i])
        ax[i].set_title(f'$\\zeta$ = {zeta_list[i]}')
        ax[i].hist2d(th_mh[:,0], th_mh[:,1], (100, 100))
plt.show()