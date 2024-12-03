import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


dfT = pd.read_csv("TEST-synth_CFA3S_shift_0.000.txt", sep=r"\s+")

dfN = pd.read_csv("../output_synthetic_fragilistic/synth_CFA3S_shift_0.000.txt", sep=r"\s+")
#dfN = pd.read_csv("synth_CFA3S_shift_0.000.txt", sep=r"\s+")

dfM = pd.merge(dfT, dfN, on=["standard"])

print(dfM)

yax = "CFA4P1-r"

#plt.figure()
for yax in ['CFA3S-B', 'CFA3S-V', 'CFA3S-R', 'CFA3S-I']:
    print(np.mean(dfM[yax+'_y'] - dfM[yax+'_x']))

