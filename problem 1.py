import numpy as np

tau_file = open('coal-mine.csv', "r")
n = 191
tau = []
for l in tau_file:
    tau.append(float(l))

# Question b

# Hyperparameters
M = 100 # MC size
d = 2 # Number of break points - 1
nu = 1 # Hyperparameter of teta
rho = 1

# Initialization : k = 0
teta = np.random.gamma(2, nu)
lambdas = np.ones(d)
t = np.linspace(tau[0], tau[-1], d+1)
t[0], t[-1] = 1851, 1963 # t[0] = t_1
for k in range(1,M):
    # Gibbs : teta, lambda
    teta = np.random.gamma(2, sum(lambdas) + nu)

    # lambda computation
    for i in range(d):
        # n_i computation
        n_i = 0
        for j in range(n):
            if t[i] <= tau[j] and tau[j] < t[i+1]:
                n_i += 1
        lambdas[i] = np.random.gamma(n_i + 2, t[i+1] - t[i] + teta)


    # Metropolis-Hastings : t
    # Random walk proposal
    R = rho*(t[i+1] - t[i-1])
    eps = np.random.rand()*2*R - R
    t_star = t[i] + eps
    # Independent proposal
    # eps = np.random.beta(rho, rho)
    # t_star = t[i-1] + eps*(t[i+1] - t[i-1])

    alpha = 0
    U = np.random.rand()
    if U <= alpha:
        t[i] = t_star
    # else : t[i] = t[i]