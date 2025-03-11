import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit # type: ignore
from uncertainties import ufloat, unumpy

data = np.loadtxt("quadrupole_scan_Q1.txt", skiprows=1)
currents = unumpy.uarray(data[:, 0], 0.001)
quad_strenghts = unumpy.uarray(data[:, 1], 0.1)
sigma_x = unumpy.uarray(data[:, 2], data[:, 3])
sigma_z = unumpy.uarray(data[:, 4], data[:, 5])

# Currents = unumpy.uarray(data[:, 0], 0.001)
# Quad_strenghts = unumpy.uarray(data[:, 1], 0.1)
# Sigma_x = unumpy.uarray(data[:, 2], data[:, 3])
# Sigma_z = unumpy.uarray(data[:, 4], data[:, 5])

# omit the first 3 data points
# currents = Currents[3:]
# quad_strenghts = Quad_strenghts[3:]
# sigma_x = Sigma_x[3:]
# sigma_z = Sigma_z[3:]

# omit the last 4 data points
# currents = Currents[:-4]
# quad_strenghts = Quad_strenghts[:-4]
# sigma_x = Sigma_x[:-4]
# sigma_z = Sigma_z[:-4]


def func(x, a, b, c):
    return a * x**2 + b * x + c


popt_x, pcov_x = curve_fit(func, unumpy.nominal_values(quad_strenghts), unumpy.nominal_values(sigma_x**2), sigma=unumpy.std_devs(sigma_x**2))
perr_x = np.sqrt(np.diag(pcov_x))

res_x = func(unumpy.nominal_values(quad_strenghts), *popt_x) - unumpy.nominal_values(sigma_x**2)
chi2_x = np.sum(res_x**2 / unumpy.std_devs(sigma_x**2)**2)
chi2_x_ndf = chi2_x / (len(res_x) - len(popt_x))
print(chi2_x_ndf)

plt.rcParams["figure.figsize"] = (10,8)

plt.errorbar(
    unumpy.nominal_values(quad_strenghts), 
    unumpy.nominal_values(sigma_x**2), 
    unumpy.std_devs(sigma_x**2), 
    unumpy.std_devs(quad_strenghts), 
    # unumpy.nominal_values(Quad_strenghts), 
    # unumpy.nominal_values(Sigma_x**2), 
    # unumpy.std_devs(Sigma_x**2), 
    # unumpy.std_devs(Quad_strenghts), 
    fmt="None", 
    ecolor="k", 
    capsize=2, 
    label="data",
)
plt.plot(
    np.linspace(unumpy.nominal_values(quad_strenghts)[0], unumpy.nominal_values(quad_strenghts)[-1], 500),
    func(np.linspace(unumpy.nominal_values(quad_strenghts)[0], unumpy.nominal_values(quad_strenghts)[-1], 500), *popt_x),
    "r-",
    label=r"fit: $\sigma_x^2(k) = (%.3g \pm %.2g)\,\text{mm}^2\text{m}^4 \cdot k^2 + (%.3g \pm %.2g)\,\text{mm}^2\text{m}^2 \cdot k + (%.3g \pm %.2g)\,\text{mm}^2$" 
    % (popt_x[0], perr_x[0], popt_x[1], perr_x[1], popt_x[2], perr_x[2]),
)
plt.xlabel(r"$k$ in $\frac{1}{\text{m}^2}$")
plt.ylabel(r"$\sigma_x^2$ in mm²")
plt.grid()
plt.legend()
plt.show()


popt_z, pcov_z = curve_fit(func, unumpy.nominal_values(quad_strenghts), unumpy.nominal_values(sigma_z**2), sigma=unumpy.std_devs(sigma_z**2))
perr_z = np.sqrt(np.diag(pcov_z))

res_z = func(unumpy.nominal_values(quad_strenghts), *popt_z) - unumpy.nominal_values(sigma_z**2)
chi2_z = np.sum(res_z**2 / unumpy.std_devs(sigma_z**2)**2)
chi2_z_ndf = chi2_z / (len(res_z) - len(popt_z))
print(chi2_z_ndf)

plt.errorbar(
    unumpy.nominal_values(quad_strenghts), 
    unumpy.nominal_values(sigma_z**2), 
    unumpy.std_devs(sigma_z**2), 
    unumpy.std_devs(quad_strenghts), 
    # unumpy.nominal_values(Quad_strenghts), 
    # unumpy.nominal_values(Sigma_z**2), 
    # unumpy.std_devs(Sigma_z**2), 
    # unumpy.std_devs(Quad_strenghts), 
    fmt="None", 
    ecolor="k", 
    capsize=2, 
    label="data",
)
plt.plot(
    np.linspace(unumpy.nominal_values(quad_strenghts)[0], unumpy.nominal_values(quad_strenghts)[-1], 500),
    func(np.linspace(unumpy.nominal_values(quad_strenghts)[0], unumpy.nominal_values(quad_strenghts)[-1], 500), *popt_z),
    "r-",
    label=r"fit: $\sigma_z^2(k) = (%.3g \pm %.2g)\,\text{mm}^2\text{m}^4 \cdot k^2 + (%.3g \pm %.2g)\,\text{mm}^2\text{m}^2 \cdot k + (%.3g \pm %.1g)\,\text{mm}^2$" 
    % (popt_z[0], perr_z[0], popt_z[1], perr_z[1], popt_z[2], perr_z[2]),
)
plt.xlabel(r"$k$ in $\frac{1}{\text{m}^2}$")
plt.ylabel(r"$\sigma_z^2$ in mm²")
plt.grid()
plt.legend()
plt.show()