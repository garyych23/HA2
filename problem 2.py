import numpy as np

# data list
tau_file = open('hmc-observations.csv', "r")
hmc = []
for l in tau_file:
    hmc.append(float(l))
y = np.array(hmc)
n = len(y)

Sigma = np.diag([5, 0.5])
Sigma_inv = np.diag([0.2, 2])

# Hyperparameters
N = 10
sig = 2
L = 25
eps = 0.25
#

def H(th, v):
    K = np.linalg.norm(v)**2
    s = 0
    for i in range(n):
        s += (y[i] - (np.linalg.norm(th)**2))**2
    thT_Siginv_th = Sigma_inv[0,0]*(th[0]**2) + Sigma_inv[1,1]*(th[1]**2)
    U = 0.5*(thT_Siginv_th + (s/(sig**2)))
    # print("K, U:", K ,U)
    return K + U


def leapfrog(th, v):
    expression = (2/(sig**2))*sum(y) - ((2*n*(np.linalg.norm(th)**2))/(sig**2))
    # print("expression", expression)
    grad_0 = expression - Sigma_inv[0,0]
    grad_1 = expression - Sigma_inv[1,1]
    grad = np.array([grad_0*th[0], grad_1*th[1]])
    theta_T = th
    v_T = v - 0.5*eps*grad
    # print("grad", grad)
    for m in range(L):
        theta_T = theta_T + eps*v_T
        if m != (L-1):
            v_T = v_T - eps*grad
        print(m==(L-1))
    v_T = v_T - 0.5*eps*grad
    v_T = - v_T
    return theta_T, v_T

theta = np.random.multivariate_normal(np.zeros(2), Sigma)
for i in range(N):
    print("Current i:", i)
    V = np.random.multivariate_normal(np.zeros(2), np.eye(2))
    theta_star, V_star = leapfrog(theta, V)
    # print("leapfrog", theta_star, V_star)
    ratio = np.exp(H(theta, V) - H(theta_star, V_star))
    # print("ratio", ratio)
    alpha = min(1, ratio)
    U = np.random.rand()
    if U <= alpha:
        theta = theta_star
print("theta", theta)