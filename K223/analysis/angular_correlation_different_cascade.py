import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit # type: ignore
from scipy.special import eval_legendre # type: ignore
from uncertainties import ufloat, unumpy

df = pd.read_table("Counter_Data.dat")
data = df.to_numpy()
angles = np.array(data[:, 0], dtype=float)
times = np.array(data[:, 1], dtype=float) / 1000
UC = np.array(data[:, 8], dtype=int)  # universal coincidences

time_sum = []
UC_sum = []

for angle in np.unique(angles):
    indices = np.where(angles == angle)[0]
    # print(len(indices))  # should be number of runs = 11
    time_sum.append(np.sum(times[indices]))
    UC_sum.append(np.sum(UC[indices]))
    
angles = np.unique(angles)
time_sum = np.array(time_sum)
UC_rate = unumpy.uarray(UC_sum, np.sqrt(UC_sum)) / time_sum


# de-adjustment correction
r = 5
epsilon = ufloat(0.0726, 0.0030)
phi = ufloat(131.6, 2.5)
K = (r**2 - 2 * epsilon * r * unumpy.cos((phi - angles) * np.pi / 180)) / (r**2 - 2 * epsilon * r)

# random coincidence correction
rand_coinc_rate = ufloat(0.175, 0.009)

UC_rate = UC_rate * K - rand_coinc_rate


theta = np.linspace(angles[0], angles[-1], 1000)


B_list = [1, -3, -15/13, 1/8]
C_list = [0, 4, 16/13, 1/24]
cascades = ["0(1)1(1)0", "0(2)0(2)0", "2(2)2(2)0", "4(2)2(2)0"]

for i in range(len(cascades)):
    def func(theta, A):
        B = B_list[i]
        C = C_list[i]
        return A * (1 + B * np.cos(theta * np.pi / 180) ** 2 + C * np.cos(theta * np.pi / 180) ** 4)

    popt, pcov = curve_fit(func, angles, unumpy.nominal_values(UC_rate), sigma=unumpy.std_devs(UC_rate))
    perr = np.sqrt(np.diag(pcov))

    res = func(angles, *popt) - unumpy.nominal_values(UC_rate)
    chi2 = np.sum(res**2 / unumpy.std_devs(UC_rate)**2)
    chi2_ndf = chi2 / (len(res) - len(popt))

    plt.plot(theta, func(theta, *popt), label=r"%s, $\chi^2/\text{ndf} = %.2f$" %(cascades[i], chi2_ndf))

plt.errorbar(angles, unumpy.nominal_values(UC_rate), unumpy.std_devs(UC_rate), fmt="None", ecolor="k", capsize=2, label="data")
plt.xlabel(r"$\theta$ / °")
plt.ylabel(r"coincidence rate / $\text{s}^{-1}$")
plt.legend(loc = "lower center")
plt.grid()
plt.show()