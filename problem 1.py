import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

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

# # Question c
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
        
