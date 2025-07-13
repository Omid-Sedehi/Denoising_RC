import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def Lorenz(t, state, sigma, beta, rho):
    x, y, z = state
    dxdt = sigma *(y-x)
    dydt = x*(rho-z)-y
    dzdt = x*y - beta*z
    return [dxdt, dydt, dzdt]

sigmas = np.linspace(0, 20, 1000)
beta = 8/3
rho = 28
max_z = []

for sigma in sigmas:
    sol = solve_ivp(Lorenz, [0, 100], [1, 1, 1], args=(sigma, beta, rho), dense_output=True)
    z_values = sol.sol(np.linspace(50,100, 10000))[0]
    max_z.append(np.max(z_values))

plt.style.use('./Styles/AIPStyles.mplstyle')
plt.figure(figsize=(5,3.5))
plt.scatter(sigmas, max_z, c='b', marker=',', s=1)
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$\mathrm{Max}(x(t))$")
plt.xlim([0, 20])
plt.show()
