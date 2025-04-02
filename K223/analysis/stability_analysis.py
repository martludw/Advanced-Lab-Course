import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit # type: ignore
from uncertainties import ufloat, unumpy
import datetime

df = pd.read_table("Counter_Data.dat")
data = df.to_numpy()
angles = np.array(data[:, 0], dtype=float)
times = np.array(data[:, 1], dtype=float) / 1000
counts = np.array(data[:, 9], dtype=int)  # 6: SCA1, 7: SCA2, 8: UC, 9: FC

time_tags = data[:, 4]
time_format = "%Y-%m-%d_%H:%M:%S"
timestamps = [datetime.datetime.strptime(tag, time_format) for tag in time_tags]

# calculate elapsed time in minutes from the first measurement
start_time = timestamps[0]
elapsed_minutes = np.array([(t - start_time).total_seconds() / 60 for t in timestamps])
# print(elapsed_minutes)


def func(minutes, a, b):
    return a * minutes + b

perform_fit = True
write_table = True
r0, s, r0_err, s_err = [], [], [], []


fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(10, 10))

for i, angle in enumerate(np.unique(angles)):
    indices = np.where(angles == angle)[0]
    minutes = elapsed_minutes[indices]
    count_rates = counts[indices] / times[indices]
    count_rates_err = np.sqrt(counts[indices]) / times[indices]
    
    ax = axes[i // 3, i % 3]
    ax.errorbar(minutes, count_rates, count_rates_err, fmt="None", ecolor="k", capsize=2)
    ax.set_title(r"$\theta = %.1f $°" %angle)
    ax.grid()
    
    if perform_fit:
        popt, pcov = curve_fit(func, minutes, count_rates, sigma=count_rates_err)
        perr = np.sqrt(np.diag(pcov))
        s.append(popt[0])
        s_err.append(perr[0])
        r0.append(popt[1])
        r0_err.append(perr[1])

        res = func(minutes, *popt) - count_rates
        chi2 = np.sum(res**2 / count_rates_err**2)
        chi2_ndf = chi2 / (len(res) - len(popt))
        
        ax.plot(minutes, func(minutes, *popt), "r-", label=r"fit, $\chi^2/\text{ndf} = %.2f$" % chi2_ndf)
        ax.legend()
        # ax.legend(loc = "lower left")
    
    if i < 6:
        plt.setp(ax.get_xticklabels(), visible=False)
    if i % 3 != 0:
        plt.setp(ax.get_yticklabels(), visible=False)    

fig.text(0.5, 0.04, "time / min", ha="center", fontsize=14)
fig.text(0.04, 0.5, r"rate / $\text{s}^{-1}$", va="center", rotation="vertical", fontsize=14)
# plt.savefig("../figs/stability_C1.pdf")
plt.show()


### write fit params into a latex table ###

# functions to format parameters and their uncertainties for siunitx
def format_r0(val, err, fmt="{:.1f}"):
    return r"\num{" + fmt.format(val) + r" \pm " + fmt.format(err) + r"}"

def format_s(val, err, fmt="{:.2g}"):
    return r"\num{" + fmt.format(val) + r" \pm " + fmt.format(err) + r"}"

if write_table: 
    table_lines = []
    table_lines.append(r"$\theta \,/\, \si{\degree}$ & $r_0 \,/\, \si{\per\s}$ & $s \,/\, \si{\per\s\per\minute}$ \\ \hline")

    for ang, a_r0, a_r0_err, a_s, a_s_err in zip(np.unique(angles), r0, r0_err, s, s_err):
        angle_str = f"%.1f" % ang
        r0_str = format_r0(a_r0, a_r0_err)
        s_str = format_s(a_s, a_s_err)
        
        row = " & ".join([angle_str, r0_str, s_str]) + r" \\"
        table_lines.append(row)

    latex_table = "\n".join(table_lines)
    with open("stability_C4_table.txt", "w") as f:
        f.write(latex_table)

