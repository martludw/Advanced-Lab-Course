import numpy as np
from matplotlib import pyplot as plt
from uncertainties import unumpy

spectrum_ungated = np.loadtxt("CFD2_spectrum_ungated.txt", skiprows=1)
spectrum_gated = np.loadtxt("CFD2_spectrum_gated.txt", skiprows=1)

channels = spectrum_ungated[:, 0]
counts_ungated = spectrum_ungated[:, 1]
counts_gated = spectrum_gated[:, 1]

def rebin(spectrum, bin_size):
    n_bins = len(spectrum) // bin_size  # use only full bins
    rebinned = np.sum(spectrum[:n_bins * bin_size].reshape(n_bins, bin_size), axis=1)  # sum counts in each bin
    return rebinned

bin_size = 50
counts_ungated_rebinned = rebin(counts_ungated, bin_size)
counts_gated_rebinned = rebin(counts_gated, bin_size)
channels_rebinned = np.arange(len(counts_gated_rebinned)) * bin_size + bin_size // 2

with np.errstate(divide='ignore', invalid='ignore'):  # handle division by zero gracefully
    sensitivity = counts_gated_rebinned / counts_ungated_rebinned
    sensitivity[~np.isfinite(sensitivity)] = 0  # replace NaN and inf with 0

sensitivity_err = np.zeros_like(sensitivity)
mask = counts_ungated_rebinned > 0  # only compute errors where full-spectrum counts are nonzero
A = counts_gated_rebinned[mask]
B = counts_ungated_rebinned[mask]
sensitivity_err[mask] = sensitivity[mask] * np.sqrt(np.where(A > 0, 1/A, 0) + 1/B)


fig, (ax1, ax2) = plt.subplots(2, figsize=(8,6), height_ratios=[3, 1])

ax1.plot(channels_rebinned, counts_ungated_rebinned, label="ungated")
ax1.plot(channels_rebinned, counts_gated_rebinned, label="gated")
ax1.set_ylabel(f"Counts / %d Channels" %bin_size)
ax1.grid()
ax1.legend()

ax2.sharex(ax1)
# ax2.plot(channels_rebinned, sensitivity)
ax2.errorbar(channels_rebinned, sensitivity, sensitivity_err, color="k", linewidth=.5, ecolor="k", elinewidth=2)
ax2.set_ylabel(r"$\eta_\text{CFD}$")
ax2.set_xlabel("Channel")
ax2.grid()

plt.setp(ax1.get_xticklabels(), visible=False)
plt.show()