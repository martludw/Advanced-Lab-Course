import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit # type: ignore
from uncertainties import ufloat, unumpy

data = np.loadtxt("beam_alignment_C3_x.txt", skiprows=1)
currents = data[:, 0]
currents_err = np.array(len(currents) * [0.0005])  # constant error
offsets1 = unumpy.uarray(data[:, 1], data[:, 2])
offsets2 = unumpy.uarray(data[:, 3], data[:, 4])

differences = offsets1 - offsets2


def func(x, a, b):
    return a * x + b


popt, pcov = curve_fit(func, currents, unumpy.nominal_values(differences), sigma=unumpy.std_devs(differences))
perr = np.sqrt(np.diag(pcov))

res = func(currents, *popt) - unumpy.nominal_values(differences)
chi2 = np.sum(res**2 / unumpy.std_devs(differences)**2)
chi2_ndf = chi2 / (len(res) - len(popt))
print(chi2_ndf)

a = ufloat(popt[0], perr[0])
b = ufloat(popt[1], perr[1])
root = -b/a
print(root)

plt.plot(
    currents,
    func(currents, *popt),
    "r-",
    label=r"fit: $\Delta x(I)=(%.4g \pm %.2g)\, \frac{\text{mm}}{\text{A}} \cdot I + (%.4g \pm %.2g)\, \text{mm}$"
    % (popt[0], perr[0], popt[1], perr[1]),
)
plt.errorbar(
    currents,
    unumpy.nominal_values(differences),
    unumpy.std_devs(differences),
    currents_err,
    fmt="None",
    ecolor="k",
    capsize=2,
    label="data",
)
plt.xlabel(r"$I$ in A")
plt.ylabel(r"$\Delta x$ in mm")
plt.grid()
plt.legend()
plt.show()
