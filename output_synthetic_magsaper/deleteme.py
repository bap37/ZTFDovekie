import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dfP = pd.read_csv("synth_PS1_shift_0.000.txt", sep=r"\s+")

dfT = pd.read_csv("TEST-synth_CSP_shift_0.000.txt", sep=r"\s+")
dfT = pd.merge(dfT, dfP, on=['standard'], how='outer')

dfN = pd.read_csv("synth_CSP_shift_0.000.txt", sep=r"\s+")
dfN = pd.merge(dfN, dfP, on=['standard'], how='outer')

yax = "CSP-r"
ZP = -0.011

plt.figure()
for filt in ['CSP-g', 'CSP-r', 'CSP-i', 'CSP-B', 'CSP-V', 'CSP-m', 'CSP-o', 'CSP-n']:
    plt.scatter( (dfN['PS1-g']-0.006) - (dfN['PS1-i']-0.003), dfT[yax] - dfN[yax] - ZP, label=f"{filt}")
    print(np.std(dfT[yax] - dfN[yax]))

plt.xlabel("SNANA PS1 g-i")
plt.ylabel(f"Neutral - SNANA {yax}")
plt.legend()
plt.ylim([-0.01, 0.01])
plt.savefig("bla.pdf", bbox_inches="tight")