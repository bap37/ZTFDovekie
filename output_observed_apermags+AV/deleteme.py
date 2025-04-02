import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#df = pd.read_pickle("/project2/rkessler/SURVEYS/PS1MD/USERS/dscolnic/Excalibur_dillon/surveydfsaper/save_irsa_newPS1.pkl")

#df2 = pd.read_pickle("/project2/rkessler/SURVEYS/PS1MD/USERS/dscolnic/Excalibur_dillon/surveydfsaper/save_irsa_oldPS1.pkl")

df = pd.read_csv("PS1SN_observed.csv")

#survey,PS1SN-g,PS1SN-r,PS1SN-i,PS1SN-z,PS1-g,PS1-r,PS1-i,PS1-z,RA,DEC,PS1SN-g_AV,PS1SN-r_AV,PS1SN-i_AV,PS1SN-z_AV,PS1-g_AV,PS1-r_AV,PS1-i_AV,PS1-z_AV

plt.figure()

plt.scatter(df['PS1SN-g'], df['PS1SN-g'] - df['PS1-g'], label="g")
plt.scatter(df['PS1SN-g'], df['PS1SN-r'] - df['PS1-r'], label="r")
plt.scatter(df['PS1SN-g'], df['PS1SN-i'] - df['PS1-i'], label="i")
plt.scatter(df['PS1SN-g'], df['PS1SN-z'] - df['PS1-z'], label="z")

print('g band difference',np.median(df['PS1SN-g'] - df['PS1-g']))
print('r band difference',np.median(df['PS1SN-r'] - df['PS1-r']))
print('i band difference',np.median(df['PS1SN-i'] - df['PS1-i']))
print('z band difference',np.median(df['PS1SN-z'] - df['PS1-z']))

plt.axhline(0, c='k')

plt.xlabel("PSF g")
plt.ylabel("PSF - AP band")
plt.legend()

plt.savefig("bla.pdf")
