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

# Hyperparameters
N = 1000 # MC size
d = 3 # Number of break points - 1
nu = 1 # Hyperparameter of teta
rho = 1/2

def countau(tau,t1,t2):
    count = 0
    for i in range(len(tau)):
        if t1 <= tau[i] < t2:
            count += 1
    return float(count)

# Initialization : k = 0
theta = np.random.gamma(2*d+2, 1/nu) # scale = 1/rate
lambda_list = np.random.gamma(2, 1/theta, d)
t_list = np.linspace(1851, 1963, d+1)
accept_rate_t = np.zeros((d-1, 1))

# For plotting (Question c)
theta_list = np.zeros((N+1, 1))
lambda_matrix = np.zeros((N+1, d))
t_matrix = np.zeros((N+1, d+1))

theta_list[0] = theta
lambda_matrix[0,:] = lambda_list
t_matrix[0,:] = t_list
#

for k in range(1,N+1):
    print("Current k=", k)
    # Gibbs : teta, lambda
    theta = np.random.gamma(2*d+2, 1/(np.sum(lambda_list) + nu))
    theta_list[k] = theta # for plotting (Question c)
    for i in range(d):
        n_i = countau(tau, t_list[i], t_list[i+1])
        lambda_list[i] = np.random.gamma(n_i + 2, 1/(t_list[i+1] - t_list[i] + theta))
    lambda_matrix[k,:] = lambda_list # for plotting (Question c)
    # Metropolis-Hastings with Random walk proposal: t
    diff = np.diff(t_list)
    for i in range(1,d): # first and last time points are fixed
        R = rho*(t_list[i+1] - t_list[i-1])
        eps = np.random.rand()*2*R - R
        t_star = t_list[i] + eps
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
            accept_rate_t[i-1] += 1
    t_matrix[k,:] = t_list # for plotting (Question c)
accept_rate_t = accept_rate_t/N
    
# Question c
print("\nQuestion c")

fig = plt.figure()
ax = fig.subplots()
ax.plot(theta_list)
ax.set_title("$\\theta$ plot for d = " + str(d))
ax.set_xlabel("$k$")
ax.set_ylabel("$\\theta$")
ax.grid()
plt.show()

fig = plt.figure()
ax = fig.subplots()
for i in range(d):
    ax.plot(lambda_matrix[:,i], label=f"$\\lambda_{i+1}$")
ax.set_title("$\\lambda$ plot for d = " + str(d))
ax.set_xlabel("$k$")
ax.set_ylabel("$\\lambda_i$")
ax.grid()
ax.legend()
plt.show()

fig = plt.figure()
ax = fig.subplots()
for i in range(1,d):
    ax.plot(t_matrix[:,i], label=f"$t_{i+1}$")
ax.set_title(f"$t$ plot for d = " + str(d))
ax.set_xlabel("$k$")
ax.set_ylabel(f"$t_i$")
ax.grid()
ax.legend()
plt.show()

fig = plt.figure()
ax = fig.subplots()
ax.hist(theta_list, 100)
ax.set_title("$\\theta$ hist for d = " + str(d))
ax.grid()
plt.show()

fig = plt.figure()
ax = fig.subplots()
for i in range(d):
    ax.hist(lambda_matrix[:,i], 100, label=f"$\\lambda_{i+1}$")
ax.set_title("$\\lambda$ hist for d = " + str(d))
ax.grid()
ax.legend()
plt.show()

fig = plt.figure()
ax = fig.subplots()
for i in range(1,d):
    ax.hist(t_matrix[:,i], 50, label=f"$t_{i+1}$")
ax.set_title(f"$t$ hist for d = " + str(d))
ax.grid()
ax.legend()
plt.show()

# Question d
# test with several values of nu
print("\nQuestion d")
print("For nu =", nu)
print("theta =", theta)
print("lambda =", lambda_list)
print("breakpoints =", t_list[1:d])

# Question e
# test with several values of rho
print("\nQuestion e")
print("For rho =", rho)
print("theta =", theta)
print("lambda =", lambda_list)
print("breakpoints =", t_list[1:d])
print("Acceptance rate of t = ", accept_rate_t)