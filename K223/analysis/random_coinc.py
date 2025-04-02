import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit # type: ignore
from uncertainties import ufloat, unumpy

df = pd.read_table("random_coinc.dat")
data = df.to_numpy()
angles = np.array(data[:, 0], dtype=float)
time = unumpy.uarray(data[:, 1], data[:, 10]) / 1000
UC = np.array(data[:, 8], dtype=int)  # universal coincidences
UC_rate = unumpy.uarray(UC, np.sqrt(UC)) / time

def func(theta, a):
    return a

popt, pcov = curve_fit(func, angles, unumpy.nominal_values(UC_rate), sigma=unumpy.std_devs(UC_rate))
perr = np.sqrt(np.diag(pcov))

UC_rate_0 = ufloat(popt[0], perr[0])
print(UC_rate_0)

res = func(angles, *popt) - unumpy.nominal_values(UC_rate)
chi2 = np.sum(res**2 / unumpy.std_devs(UC_rate)**2)
chi2_ndf = chi2 / (len(res) - len(popt))
print(chi2_ndf)

plt.errorbar(angles, unumpy.nominal_values(UC_rate), unumpy.std_devs(UC_rate), fmt="None", ecolor="k", capsize=2, label="data")
plt.plot(angles, np.array([popt[0]] * len(angles)), "r-", label=r"fit, $\chi^2/\text{ndf} = %.2f$" %chi2_ndf)
plt.xlabel(r"$\theta$ / °")
plt.ylabel(r"count rate / $\text{s}^{-1}$")
plt.legend()
plt.grid()
plt.show()

