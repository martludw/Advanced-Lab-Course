import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit # type: ignore
from uncertainties import ufloat, unumpy

df = pd.read_table("Counter_Data.dat")
data = df.to_numpy()
angles = np.array(data[:, 0], dtype=float)
times = np.array(data[:, 1], dtype=float) / 1000
counts = np.array(data[:, 6], dtype=int)  # 6: SCA1, 7: SCA2, 8: UC, 9: FC

count_sum = []
time_sum = []

for angle in np.unique(angles):
    indices = np.where(angles == angle)[0]
    count_sum.append(np.sum(counts[indices]))
    time_sum.append(np.sum(times[indices]))
    
angles = np.unique(angles)
time_sum = np.array(time_sum)
count_rate = np.array(count_sum) / time_sum
count_rate_err = np.sqrt(count_sum) / time_sum

# print(angles)

r = 5

def func(theta, N0, epsilon, phi):
    return N0 / (r**2 - 2 * epsilon * r * np.cos((phi - theta) * np.pi / 180))

popt, pcov = curve_fit(func, angles, count_rate, sigma=count_rate_err, p0=(count_rate[0], 0, 180))
perr = np.sqrt(np.diag(pcov))

res = func(angles, *popt) - count_rate
chi2 = np.sum(res**2 / count_rate_err**2)
chi2_ndf = chi2 / (len(res) - len(popt))
# print(chi2_ndf)

N0 = ufloat(popt[0], perr[0])
epsilon = ufloat(popt[1], perr[1])
phi = ufloat(popt[2], perr[2])
print("N0 =", N0)
print("epsilon =", epsilon)
print("phi =", phi)

theta = np.linspace(angles[0], angles[-1], 1000)

plt.errorbar(angles, count_rate, count_rate_err, fmt="None", ecolor="k", capsize=2, label="data")
plt.plot(theta, func(theta, *popt), "r-", label=r"fit, $\chi^2/\text{ndf} = %.2f$" %chi2_ndf)
plt.xlabel(r"$\theta$ / °")
plt.ylabel(r"rate / $\text{s}^{-1}$")
plt.legend()
plt.grid()
plt.show()


plt.plot(theta, (r**2 - 2 * popt[1] * r * np.cos((popt[2] - theta) * np.pi / 180)) / (r**2 - 2 * popt[1] * r))
plt.xlabel(r"$\theta$ / °")
plt.ylabel(r"$K(\theta)$")
plt.grid()
plt.show()
