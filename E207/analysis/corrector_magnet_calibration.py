import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit  # type: ignore
from uncertainties import ufloat

data = np.loadtxt("corrector_magnet_calibration_C0_x.txt", skiprows=1)
angles = data[:, 0]
angles_err = np.array(len(angles) * [0.1])  # constant error
offsets = data[:, 1]
# offsets_err = data[:, 2]
offsets_err = np.array(len(angles) * [0.05])  # for C0_x
# offsets_err = np.array(len(angles) * [0.05])  # for C0_z
# offsets_err = np.array(len(angles) * [0.1])  # for C1_x
# offsets_err = np.array(len(angles) * [0.2])  # for C1_z


def func(x, a, b):
    return a * x + b


popt, pcov = curve_fit(func, angles, offsets, sigma=offsets_err)
perr = np.sqrt(np.diag(pcov))

res = func(angles, *popt) - offsets
chi2 = np.sum(res**2 / offsets_err**2)
chi2_ndf = chi2 / (len(res) - len(popt))
print(chi2_ndf)

plt.plot(
    angles,
    func(angles, *popt),
    "r-",
    label=r"fit: $x(\alpha)=(%.3f \pm %.3f)\, \frac{\text{mm}}{\text{mrad}} \cdot \alpha + (%.3f \pm %.3f)\, \text{mm}$"
    % (popt[0], perr[0], popt[1], perr[1]),
)
plt.errorbar(
    angles,
    offsets,
    offsets_err,
    angles_err,
    fmt="None",
    ecolor="k",
    capsize=2,
    label="data",
)
plt.xlabel(r"$\alpha_x$ in mrad")
plt.ylabel(r"$x$ in mm")
plt.grid()
# plot legend lables in customized order
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[i] for i in [1, 0]], [labels[i] for i in [1, 0]])
# plt.text(
#     0.025,
#     0.76,
#     r"$\chi^2/\text{ndf}=%.2f$" % chi2_ndf,
#     transform=plt.gca().transAxes,
#     bbox=dict(facecolor="white", alpha=0.5),
# )
plt.show()
