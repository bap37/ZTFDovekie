import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dfN = pd.read_csv("/project2/rkessler/SURVEYS/PS1MD/USERS/dscolnic/Excalibur_dillon/output_synthetic_magsaper/synth_PS1_shift_0.000.txt.gz", sep=" ")

#dfN = pd.read_csv("synth_PS1_shift_0.000.txt", sep=" ")
dfT = pd.read_csv("synth_PS1_shift_0.000.txt", sep=" ")

plt.figure()
plt.scatter(dfT['PS1-g'], dfT['PS1-g'] + dfN['PS1s_RS14_PS1_tonry-g'], label='g')
plt.scatter(dfT['PS1-g'], dfT['PS1-r'] + dfN['PS1s_RS14_PS1_tonry-r'], label='r')
plt.scatter(dfT['PS1-g'], dfT['PS1-i'] + dfN['PS1s_RS14_PS1_tonry-i'], label='i')
plt.scatter(dfT['PS1-g'], dfT['PS1-z'] + dfN['PS1s_RS14_PS1_tonry-z'], label='z')
plt.xlim([0,10])
plt.ylim([-0.025,0.025])
plt.legend()

plt.savefig("bla.pdf")


print(np.mean(dfT['PS1-g'] + dfN['PS1s_RS14_PS1_tonry-g']))
print(np.mean(dfT['PS1-r'] + dfN['PS1s_RS14_PS1_tonry-r']))
print(np.mean(dfT['PS1-i'] + dfN['PS1s_RS14_PS1_tonry-i']))
print(np.mean(dfT['PS1-z'] + dfN['PS1s_RS14_PS1_tonry-z']))
