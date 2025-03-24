import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit # type: ignore

df = pd.read_table("Counter_Data.dat")
data = df.to_numpy()
angles = np.array(data[:, 0], dtype=float)
C1 = np.array(data[:, 6], dtype=int)  # SCA1
C2 = np.array(data[:, 7], dtype=int)  # SCA2
C3 = np.array(data[:, 8], dtype=int)  # universal coincidences
C4 = np.array(data[:, 9], dtype=int)  # fast coincidencs

C1_sum = []
C2_sum = []
C3_sum = []
C4_sum = []

for angle in np.unique(angles):
    indices = np.where(angles == angle)[0]
    print(len(indices))  # should be number of runs = 11
    C1_sum.append(np.sum(C1[indices]))
    C2_sum.append(np.sum(C2[indices]))
    C3_sum.append(np.sum(C3[indices]))
    C4_sum.append(np.sum(C4[indices]))
    
angles = np.unique(angles)
C1_sum = np.array(C1_sum)
C2_sum = np.array(C2_sum)
C3_sum = np.array(C3_sum)
C4_sum = np.array(C4_sum)
C1_sum_err = np.sqrt(C1_sum)
C2_sum_err = np.sqrt(C2_sum)
C3_sum_err = np.sqrt(C3_sum)
C4_sum_err = np.sqrt(C4_sum)

print(angles)

def func(theta, A, B, C):
    return A * (1 + B * np.cos(theta * np.pi / 180) ** 2 + C * np.cos(theta * np.pi / 180) ** 4)

popt, pcov = curve_fit(func, angles, C3_sum, sigma=C3_sum_err)
perr = np.sqrt(np.diag(pcov))

res = func(angles, *popt) - C3_sum
chi2 = np.sum(res**2 / C3_sum_err**2)
chi2_ndf = chi2 / (len(res) - len(popt))
print(chi2_ndf)

theta = np.linspace(angles[0], angles[-1], 1000)

plt.errorbar(angles, C3_sum, C3_sum_err, fmt="None", ecolor="k", capsize=2, label="data")
plt.plot(theta, func(theta, *popt), "r-", label="Fit")
plt.xlabel(r"$\theta$ / °")
plt.ylabel("coincidences")
# plt.ylabel(r"coincidence rate / $\text{s}^{-1}$")
plt.legend()
plt.show()