import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from mpl_toolkits.mplot3d import Axes3D

# data list
tau_file = open('coal-mine.csv', "r")
tau = []
for l in tau_file:
    tau.append(float(l))
tau = np.array(tau)
n = len(tau)
# Question b

# Hyperparameters
N = 1000 # MC size
d = 2 # Number of break points - 1
nu = 1/5 # Hyperparameter of teta
rho = 1/2

def countau(tau,t1,t2):
    count = 0
    for i in range(len(tau)):
        if t1 <= tau[i] < t2:
            count += 1
    return float(count)

# Initialization : k = 0
theta = np.random.gamma(2*d+2, nu)
lambda_list = np.random.gamma(2, theta, d)
t_list = np.linspace(1851, 1963, d+1) 
for k in range(1,N+1):
    print("Current k=", k)
    # Gibbs : teta, lambda
    theta = np.random.gamma(2*d+2, np.sum(lambda_list) + nu)
    for i in range(d):
        n_i = countau(tau, t_list[i], t_list[i+1])
        lambda_list[i] = np.random.gamma(n_i + 2, t_list[i+1] - t_list[i] + theta)
    # Metropolis-Hastings with Random walk proposal: t
    diff = np.diff(t_list)
    for i in range(1,d): # first and last time points are fixed
        R = rho*(t_list[i+1] - t_list[i-1])
        eps = np.random.rand()*2*R - R
        t_star = t_list[i] + eps
        print("t_star:", t_star)
        diff = np.diff(t_list)
        if  np.any(diff <= 0):
            ratio = 0
        else:
            ratio = ((t_list[i+1]-t_star)*(t_star-t_list[i-1]))/((t_list[i+1]-t_list[i])*(t_list[i]-t_list[i-1]))\
            * np.exp((lambda_list[i]-lambda_list[i-1])*(t_star-t_list[i]))\
            * lambda_list[i-1]**(countau(tau,t_list[i-1],t_star)-countau(tau,t_list[i-1],t_list[i]))\
            * lambda_list[i]**(countau(tau,t_star,t_list[i+1])-countau(tau,t_list[i],t_list[i+1]))
        alpha = np.min([1,ratio])
        U = np.random.rand()
        if U <= alpha:
            t_list[i] = t_star
        

# Question d
d=2

# Posteriors
def prop_post_theta(theta, lambda_list, nu):
    return theta**(2*d+1)*np.exp(-theta*(np.sum(lambda_list) + nu))
    
def prop_post_lambda(lambda_list, theta, t, nu):
    diff = np.diff(t)
    if np.any(diff <= 0):
        return 0
    ans = 1
    for i in range(d):
        ans *= lambda_list[i]**(countau(tau, t[i], t[i+1]) + 1)
    for i in range(d):
        ans *= np.exp(-lambda_list[i]*(t[i+1]-t[i]+theta))
    return ans*(nu**2)*np.exp(-nu*theta)

def prop_post_t(lambda_list, theta, t_value, nu):
    t = np.array([1851, t_value, 1963])
    ans = 1
    for i in range(d):
        ans *= t[i+1]-t[i]
    for i in range(d):
        ans *= np.exp(-lambda_list[i]*(t[i+1]-t[i]))
    for i in range(d):
        ans *= lambda_list[i]**countau(tau, t[i], t[i+1])
    return ans*(nu**2)*np.exp(-nu*theta)


# Plotting
theta_space = np.linspace(0.5, 15, 50)
lambda_1_space = np.linspace(0.5, 15, 50)
lambda_2_space = np.linspace(0.5, 15, 50)
t_space = np.linspace(1852, 1962, 50)

theta_ex = 5
lambda_ex = 5
t_ex = np.array([1851,1900,1963])

nu_list = np.array([1/5,1/3,1/2,1])

# post theta plot
fig, ax = plt.subplots(2, 2)
for i in range(2):
    for j in range(2):
        ax[i,j].plot(theta_space, prop_post_theta(theta_space, lambda_ex*np.ones(d), nu_list[2*i+j]))
        ax[i,j].set_title("Posterior of theta plot with nu = " + str(nu_list[2*i+j]))
plt.show()

# post lambda plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
lambda_1_space, lambda_2_space = np.meshgrid(lambda_1_space, lambda_2_space)
post_lambda = prop_post_lambda([lambda_1_space, lambda_2_space], theta_ex, t_ex, nu_list[3])
ax.plot_surface(lambda_1_space, lambda_2_space, post_lambda, cmap='viridis')
ax.set_xlabel('Lambda 1')
ax.set_ylabel('Lambda 2')
plt.show()

# post t plot
fig, ax = plt.subplots(2, 2)
for i in range(2):
    for j in range(2):
        ax[i,j].plot(t_space, np.array([prop_post_t(lambda_ex*np.ones(d), theta_ex, t, nu_list[2*i+j]) for t in t_space]))
    
plt.show()