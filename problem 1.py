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
# We suppose d=1

# Posteriors
def prop_post_theta(theta, lambda1):
    return theta*np.exp(-theta*(lambda1 + nu))
    

def post_lambda(lambda1, theta, t):
    ans = 1
    for i in range(d):
        ans *= lambda1**(countau(tau, t[i], t[i+1]) + 1)
    for i in range(d):
        ans *= np.exp(-lambda1(t[i+1]-t[i]+theta))
    return ans

def post_t(t, lambda1):
    ans = 1
    for i in range(d):
        ans *= t[i+1]-t[i]
    for i in range(d):
        ans *= np.exp(-lambda1*(t[i+1]-t[i]))
    for i in range(d):
        ans *= lambda1**countau(tau, t[i], t[i+1])
    return ans

# Plotting
theta_space = np.linspace(0.5, 15, 50)
lambda_1_space = np.linspace(0.5, 15, 50)
t_space = np.linspace(1851, 1963, 50)

theta_example = 5
lambda_example = 5
t_example = (1963-1851)/2

# post theta plot
plt.figure(1)
plt.plot(theta_space, [post_theta(theta_, lambda_example) for theta_ in theta_space])
plt.title("Posterior of theta plot")
plt.ylabel("posterior")
plt.xlabel("theta")
plt.show()

# post lambda plot
plt.figure(1)
plt.plot(theta_space, [post_lambda(theta, lambda_example) for theta_ in theta_space])
plt.title("Posterior of theta plot")
plt.ylabel("posterior")
plt.xlabel("theta")
plt.show()

# post theta

# post t plot
plt.figure(1)
plt.plot(theta_space, [post_theta(theta_, lambda_example) for theta_ in theta_space])
plt.title("Posterior of theta plot")
plt.ylabel("posterior")
plt.xlabel("theta")
plt.show()