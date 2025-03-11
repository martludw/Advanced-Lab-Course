import numpy as np
from uncertainties import ufloat, unumpy

data = np.loadtxt("beam_alignment_C3_x.txt", skiprows=1)
currents = data[:, 0]
currents_err = np.array(len(currents) * [0.0005])  # constant error
offsets1 = unumpy.uarray(data[:, 1], data[:, 2])
offsets2 = unumpy.uarray(data[:, 3], data[:, 4])
differences = offsets1 - offsets2

with open("C3_x_latex.txt", "w") as f:
    for i in range(len(currents)):
        line = (f"\\num{{{currents[i]} \\pm {currents_err[i]}}} & "
                f"\\num{{{unumpy.nominal_values(offsets1)[i]} \\pm {unumpy.std_devs(offsets1)[i]}}} & "
                f"\\num{{{unumpy.nominal_values(offsets2)[i]} \\pm {unumpy.std_devs(offsets2)[i]}}} & "
                f"\\num{{{unumpy.nominal_values(differences)[i]:.2f} \\pm {unumpy.std_devs(differences)[i]:.2f}}} \\\\")
        f.write(line + "\n")