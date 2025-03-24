import numpy as np
from matplotlib import pyplot as plt

def f(theta, alpha, beta):
    return 1 + 0.5 * (alpha + beta) * np.cos(theta * np.pi / 180) ** 2 + 0.5 * (alpha - beta) * np.cos(theta * np.pi / 180) ** 4

alpha_0 = 1 / 6
beta_0 = 1/ 12

# theta = np.linspace(0, 360, 1000)
theta = np.linspace(90, 270, 1000)
plt.plot(theta, f(theta, alpha_0, beta_0), label=r"$\alpha=\alpha_0, \beta=\beta_0$")
plt.plot(theta, f(theta, 2*alpha_0, beta_0), label=r"$\alpha=2\alpha_0, \beta=\beta_0$")
plt.plot(theta, f(theta, alpha_0, 5*beta_0), label=r"$\alpha=\alpha_0, \beta=5\beta_0$")
plt.xlabel(r"$\theta$ / °")
plt.ylabel(r"$f(\theta)$")
plt.legend()
plt.grid()
plt.show()