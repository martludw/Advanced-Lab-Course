import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erf # type: ignore
from scipy.optimize import curve_fit # type: ignore
from uncertainties import ufloat

data = np.loadtxt("prompt_curve.txt", skiprows=1)
delay1 = data[:, 0]
delay2 = data[:, 1]
delay = delay2 - delay1
counts = data[:, 2]
counts_err = np.sqrt(counts)

def func(t, A, A0, t0, w, sigma):
    return 0.5 * A * (1 + erf((t - (t0 - w / 2)) / sigma) * erf(((t0 + w / 2) - t) / sigma)) + A0

popt, pcov = curve_fit(func, delay, counts, sigma=counts_err)
perr = np.sqrt(np.diag(pcov))

A = ufloat(popt[0], perr[0])
A0 = ufloat(popt[1], perr[1])
t0 = ufloat(popt[2], perr[2])
w = ufloat(popt[3], perr[3])
sigma = ufloat(popt[4], perr[4])

res = func(delay, *popt) - counts
chi2 = np.sum(res**2 / counts_err**2)
chi2_ndf = chi2 / (len(res) - len(popt))
print(chi2_ndf)

t = np.linspace(np.min(delay), np.max(delay), 1000)

print("A = ", A)
print("A0 = ", A0)
print("t0 = ", t0)
print("w = ", w)
print("sigma = ", sigma)

plt.errorbar(delay, counts, counts_err, fmt="None", ecolor="k", capsize=2, label="data")
plt.plot(t, func(t, *popt), "r-", label=r"fit, $\chi^2/\text{ndf}=%.2f$" %chi2_ndf)
plt.xlabel(r"$t$ / ns")
plt.ylabel(r"Counts")
plt.legend()
plt.grid()
plt.show()