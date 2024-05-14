import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from statsmodels.graphics import tsaplots
import matplotlib.pyplot as plt

np.random.seed(9)

####### Problem 1 ########

# data list
tau_file = open('coal-mine.csv', "r")
tau = []
for l in tau_file:
    tau.append(float(l))
tau = np.array(tau)
n = len(tau)

# Question b
print("Question b")
N = 10000 # MC size
burn_in = 1000 # burn_in

def mcmc(d, nu, rho):
    def countau(t1,t2):
        count = 0
        for i in range(len(tau)):
            if t1 <= tau[i] < t2:
                count += 1
        return float(count)

    # Initialization : k = 0
    theta = np.random.gamma(2, 1/nu) # scale = 1/rate
    lambda_list = np.random.gamma(2, 1/theta, d)
    t_list = np.linspace(1851, 1963, d+1)
    accept_rate_t = np.zeros(d-1)

    # For plotting (Question c)
    theta_list = np.zeros((N+1, 1))
    lambda_matrix = np.zeros((N+1, d))
    t_matrix = np.zeros((N+1, d+1))

    theta_list[0] = theta
    lambda_matrix[0,:] = lambda_list
    t_matrix[0,:] = t_list

    for k in range(1,N+1):
        # Gibbs : teta, lambda
        theta = np.random.gamma(2*d+2, 1/(np.sum(lambda_list) + nu))
        theta_list[k] = theta # for plotting (Question c)
        for i in range(d):
            n_i = countau(t_list[i], t_list[i+1])
            lambda_list[i] = np.random.gamma(n_i + 2, 1/(t_list[i+1] - t_list[i] + theta))
        lambda_matrix[k,:] = lambda_list 
        # Metropolis-Hastings with Random walk proposal: t
        diff = np.diff(t_list)
        for i in range(1,d): # first and last time points are fixed
            R = rho*(t_list[i+1] - t_list[i-1])
            eps = np.random.rand()*2*R - R
            t_star = t_list[i] + eps
            diff = np.diff(t_list)
            if  np.any(diff <= 0): # check if the time points are in striclty increasing order
                ratio = 0 # if not, we put the ratio equal to 0, to reject the proposal
            else:
                ratio = ((t_list[i+1]-t_star)*(t_star-t_list[i-1]))/((t_list[i+1]-t_list[i])*(t_list[i]-t_list[i-1]))\
                * np.exp((lambda_list[i]-lambda_list[i-1])*(t_star-t_list[i]))\
                * lambda_list[i-1]**(countau(t_list[i-1],t_star)-countau(t_list[i-1],t_list[i]))\
                * lambda_list[i]**(countau(t_star,t_list[i+1])-countau(t_list[i],t_list[i+1]))
            alpha = np.min([1,ratio])
            U = np.random.rand()
            if U <= alpha:
                t_list[i] = t_star
                if k>burn_in:
                    accept_rate_t[i-1] += 1
        t_matrix[k,:] = t_list 
        
    accept_rate_t = accept_rate_t/(N-burn_in+1)
    theta_list = theta_list[burn_in+1:]
    lambda_matrix = lambda_matrix[burn_in+1:,:]
    t_matrix = t_matrix[burn_in+1:,:]
    return theta_list, lambda_matrix, t_matrix, accept_rate_t

# Question c
d_list = [2, 3, 4, 5]
nu = 1
rho = 1/10

fig,ax = plt.subplots(nrows=2, ncols=2,figsize=(8, 8))
for i in range(2):
    for j in range(2):
        ax[i][j].set_title(f'd = {d_list[2*i+j]}')
        ax[i][j].plot(mcmc(d_list[2*i+j], nu, rho)[0], label = '$\\theta$')
        ax[i][j].legend()
        ax[i][j].grid()
plt.show()

fig,ax = plt.subplots(nrows=2, ncols=2,figsize=(8, 8))
for i in range(2):
    for j in range(2):
        ax[i][j].set_title(f'd = {d_list[2*i+j]}')
        ax[i][j].hist(mcmc(d_list[2*i+j], nu, rho)[0],100,linewidth=1, label = '$\\theta$')
        ax[i][j].legend()
        ax[i][j].grid()
plt.show()

fig,ax = plt.subplots(nrows=2, ncols=2,figsize=(8, 8))
for i in range(2):
    for j in range(2):
        ax[i][j].set_title(f'd = {d_list[2*i+j]}')
        for k in range(d_list[2*i+j]):
            ax[i][j].plot(mcmc(d_list[2*i+j], nu, rho)[1][:,k], label=f'$\\lambda_{k+1}$')
        ax[i][j].legend()
        ax[i][j].grid()
plt.show()

fig,ax = plt.subplots(nrows=2, ncols=2,figsize=(8, 8))
for i in range(2):
    for j in range(2):
        ax[i][j].set_title(f'd = {d_list[2*i+j]}')
        for k in range(d_list[2*i+j]):
            ax[i][j].hist(mcmc(d_list[2*i+j], nu, rho)[1][:,k], 100, linewidth=1, label=f'$\\lambda_{k+1}$')
        ax[i][j].legend()
        ax[i][j].grid()
plt.show()

fig,ax = plt.subplots(nrows=2, ncols=2,figsize=(8, 8))
for i in range(2):
    for j in range(2):
        ax[i][j].set_title(f'd = {d_list[2*i+j]}')
        for k in range(1,d_list[2*i+j]):
            ax[i][j].plot(mcmc(d_list[2*i+j], nu, rho)[2][:,k], label=f'$t_{k+1}$')
        ax[i][j].legend()
        ax[i][j].grid()
plt.show()

fig,ax = plt.subplots(nrows=2, ncols=2,figsize=(8, 8))
for i in range(2):
    for j in range(2):
        ax[i][j].set_title(f'd = {d_list[2*i+j]}')
        for k in range(1,d_list[2*i+j]):
            ax[i][j].hist(mcmc(d_list[2*i+j], nu, rho)[2][:,k],100, linewidth=1, label=f'$t_{k+1}$')
        ax[i][j].legend()
        ax[i][j].grid()
plt.show()

# Question d
nu_list = [1/10, 1, 5, 10]
d = 3
rho = 1/10

fig,ax = plt.subplots(nrows=2, ncols=2,figsize=(8, 8))
for i in range(2):
    for j in range(2):
        ax[i][j].set_title(f'$\\nu$ = {nu_list[2*i+j]}')
        ax[i][j].hist(mcmc(d, nu_list[2*i+j], rho)[0],100,linewidth=1,label = '$\\theta$')
        ax[i][j].legend()
        ax[i][j].grid()
plt.show()

fig,ax = plt.subplots(nrows=2, ncols=2,figsize=(8, 8))
for i in range(2):
    for j in range(2):
        ax[i][j].set_title(f'$\\nu$ = {nu_list[2*i+j]}')
        for k in range(3):
            ax[i][j].hist(mcmc(d, nu_list[2*i+j], rho)[1][:,k], 100, linewidth=1, label=f'$\\lambda_{k+1}$')
        ax[i][j].legend()
        ax[i][j].grid()
plt.show()

fig,ax = plt.subplots(nrows=2, ncols=2,figsize=(8, 8))
for i in range(2):
    for j in range(2):
        ax[i][j].set_title(f'$\\nu$ = {nu_list[2*i+j]}')
        for k in range(1,3):
            ax[i][j].hist(mcmc(d, nu_list[2*i+j], rho)[2][:,k],100, linewidth=1, label=f'$t_{k+1}$')
        ax[i][j].legend()
        ax[i][j].grid()
plt.show()

# Question e
rho_list = [1/100, 1/10, 1, 10]
d = 3
nu = 1

fig,ax = plt.subplots(nrows=2, ncols=2,figsize=(8, 8))
for i in range(2):
    for j in range(2):
        ax[i][j].set_title(f'$\\rho$ = {rho_list[2*i+j]}')
        ax[i][j].hist(mcmc(d, nu, rho_list[2*i+j])[0],100,linewidth=1,label = '$\\theta$')
        ax[i][j].legend()
        ax[i][j].grid()
plt.show()

fig,ax = plt.subplots(nrows=2, ncols=2,figsize=(8, 8))
for i in range(2):
    for j in range(2):
        ax[i][j].set_title(f'$\\rho$ = {rho_list[2*i+j]}')
        for k in range(3):
            ax[i][j].hist(mcmc(d, nu, rho_list[2*i+j])[1][:,k], 100, linewidth=1, label=f'$\\lambda_{k+1}$')
        ax[i][j].legend()
        ax[i][j].grid()
plt.show()

fig,ax = plt.subplots(nrows=2, ncols=2,figsize=(8, 8))
for i in range(2):
    for j in range(2):
        ax[i][j].set_title(f'$\\rho$ = {rho_list[2*i+j]}')
        for k in range(1,3):
            ax[i][j].hist(mcmc(d, nu, rho_list[2*i+j])[2][:,k],100, linewidth=1, label=f'$t_{k+1}$')
        ax[i][j].legend()
        ax[i][j].grid()
plt.show()

for i in range(2):
    for j in range(2):
        print('for rho ='+ str(rho_list[2*i+j])+'Acceptance rate =',mcmc(d, nu, rho_list[2*i+j])[3])
        

####### Problem 2 ########

# data list
hmc_file = open('hmc-observations.csv', "r")
y = []
for l in hmc_file:
    y.append(float(l))
y = np.array(y)
n = len(y)

# Parameters
Sigma = np.diag([5, 0.5])
Sigma_inv = np.diag([0.2, 2])

# Hyperparameters
N = 10000
sig = 2

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

plt.figure()
plt.hist2d(theta_hmc[:,0], theta_hmc[:,1], (100, 100))
plt.show()

print("\nHMC")
print("acceptance_rate=", acceptance_rate_hmc)

# plot autocorrelation function
fig = tsaplots.plot_acf(theta_hmc[:,0])
plt.show()
fig = tsaplots.plot_acf(theta_hmc[:,1])
plt.show()

print("mh for zeta= 0.1")
theta_mh, acceptance_rate_mh = mh(zeta= 0.1)

plt.figure()
plt.hist2d(theta_mh[:,0], theta_mh[:,1], (100, 100))
plt.show()

print("\nMH")
print("acceptance_rate=", acceptance_rate_mh)

# plot autocorrelation function
fig = tsaplots.plot_acf(theta_mh[:,0])
plt.show()
fig = tsaplots.plot_acf(theta_mh[:,1])
plt.show()

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