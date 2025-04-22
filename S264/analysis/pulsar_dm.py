import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit # type: ignore
from uncertainties import ufloat, unumpy
import seaborn as sns # type: ignore

T = 156.4
nu_max = 1430.39
nu_band = 12.3

data = np.loadtxt("./../data/0355_1_8.ascii", skiprows=1)
time = data[:,0] * T / 256

def gauss(t, a, b, mu, sigma):
    return a * np.exp(-(t-mu)**2 / (2*sigma**2)) + b


# colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
colors = sns.color_palette("Set2")
fig, axes = plt.subplots(4, 2, sharex=True, figsize=(10, 10))

for k in range(1, 9):
    nu = nu_max - (2*k - 1) / 2 * nu_band
    
    mu0 = time[np.argmax(data[:, k])]
    sigma0 = T / 20
    b0 = np.mean(data[:, k])
    a0 = np.max(data[:, k]) - b0
    
    popt, pcov = curve_fit(gauss, time, data[:, k], p0=(a0, b0, mu0, sigma0))
    perr = np.sqrt(np.diag(pcov))
    
    # print(a0, b0, mu0, sigma0)
    a = ufloat(popt[0], perr[0])
    b = ufloat(popt[1], perr[1])
    mu = ufloat(popt[2], perr[2])
    sigma = ufloat(popt[3], perr[3])
    print(nu, a, b, mu, sigma)
    
    tmin = popt[2] - 3 * popt[3]
    tmax = popt[2] + 3 * popt[3]
    mask = (time > tmin) * (time < tmax)
    
    # ax.shape = (4, 2)
    ax = axes[(k-1) % 4, (k-1) // 4]
    ax.plot(time, data[:, k], color=colors[k-1], label=r"band %d, $\nu = %.2f \pm %.2f$" %(k, nu, nu_band / 2))
    ax.plot(time[mask], gauss(time[mask], *popt), "k-")
    ax.grid()
    ax.legend()
fig.text(0.5, 0.04, r"$t$ / ms", ha="center", fontsize=14)
fig.text(0.04, 0.5, "intensity (arbitrary units)", va="center", rotation="vertical", fontsize=14)
plt.show()


mu1 = ufloat(58.3, 0.4)
mu8 = ufloat(75.2, 0.4)
nu1 = ufloat(1424.24, 6.15)
nu8 = ufloat(1338.14, 6.15)

DM = 1 / 4150 * (mu8 - mu1) / 1000 / (nu8 ** (-2) - nu1 ** (-2))
print(DM)

n_e = 0.03
d = DM / n_e
print(d)