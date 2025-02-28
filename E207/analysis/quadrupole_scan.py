import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit # type: ignore
from uncertainties import ufloat

data = np.loadtxt("quadrupole_scan_Q2.txt", skiprows=1)
currents = data[:, 0]
currents_err = np.array(len(currents) * [0.1])  # constant error
quad_strenghts = data[:, 1]
quad_strenghts_err = np.array(len(quad_strenghts) * [0.1])  # constant error
sigma_x = data[:, 2]
sigma_x_err = data[:, 3]
sigma_z = data[:, 4]
sigma_z_err = data[:, 5]


def func(x, a, b, c):
    return a * x**2 + b * x + c


popt_x, pcov_x = curve_fit(func, quad_strenghts, sigma_x**2)
perr_x = np.sqrt(np.diag(pcov_x))

plt.plot(
    quad_strenghts,
    sigma_x**2,
    "k+",
    label="data",
)
plt.plot(
    np.linspace(quad_strenghts[0], quad_strenghts[-1], 100),
    func(np.linspace(quad_strenghts[0], quad_strenghts[-1], 100), *popt_x),
    "r-",
    label=r"fit",
)
plt.xlabel(r"$k$ in $\frac{1}{\text{m}^2}$")
plt.ylabel(r"$\sigma_x^2$ in mm²")
plt.grid()
plt.legend()
plt.show()


popt_z, pcov_z = curve_fit(func, quad_strenghts, sigma_z**2)
perr_z = np.sqrt(np.diag(pcov_z))

plt.plot(
    quad_strenghts,
    sigma_z**2,
    "k+",
    label="data",
)
plt.plot(
    np.linspace(quad_strenghts[0], quad_strenghts[-1], 100),
    func(np.linspace(quad_strenghts[0], quad_strenghts[-1], 100), *popt_z),
    "r-",
    label=r"fit",
)
plt.xlabel(r"$k$ in $\frac{1}{\text{m}^2}$")
plt.ylabel(r"$\sigma_z^2$ in mm²")
plt.grid()
plt.legend()
plt.show()
