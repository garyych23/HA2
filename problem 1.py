import numpy as np

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
theta = np.random.gamma(2, nu)
lambda_list = np.random.gamma(2, theta, d)
t_list = np.linspace(1851, 1963, d+1) 
for k in range(1,N+1):
    print("Current k=", k)
    # Gibbs : teta, lambda
    theta = np.random.gamma(2, np.sum(lambda_list) + nu)
    for i in range(d):
        n_i = countau(tau, t_list[i], t_list[i+1])
        lambda_list[i] = np.random.gamma(n_i + 2, t_list[i+1] - t_list[i] + theta)
    # Metropolis-Hastings with Random walk proposal: t
    diff = np.diff(t_list)
    i=1
    while (not np.any(diff <= 0)) and (i < d): # 
        #for i in range(1,d): # first and last time points are fixed
        R = rho*(t_list[i+1] - t_list[i-1])
        eps = np.random.rand()*2*R - R
        t_star = t_list[i] + eps
        print("t_star:", t_star)
        ratio = ((t_list[i+1]-t_star)*(t_star-t_list[i-1]))/((t_list[i+1]-t_list[i])*(t_list[i]-t_list[i-1]))\
        * np.exp((lambda_list[i]-lambda_list[i-1])*(t_star-t_list[i]))\
        * lambda_list[i-1]**(countau(tau,t_list[i-1],t_star)-countau(tau,t_list[i-1],t_list[i]))\
        * lambda_list[i]**(countau(tau,t_star,t_list[i+1])-countau(tau,t_list[i],t_list[i+1]))
        alpha = np.min([1,ratio])
        U = np.random.rand()
        if U <= alpha:
            t_list[i] = t_star
        i += 1
        
        