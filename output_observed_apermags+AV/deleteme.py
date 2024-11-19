import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("CSP_observed.csv")
df = df.loc[df['GAIA_CSP-B'] > -900]

fig, ax = plt.subplots()

#filter = 'CSP-V'



for filt in ['B', 'V', 'r', 'i', 'g', 'm', 'n', 'o']:
    ax.scatter(df['PS1-g'] - df['PS1-i'], df[f'CSP-V'] - df[f'GAIA_CSP-{filt}'], label=f"{filt}")
    print(np.std(df[f'CSP-V'] - df[f'GAIA_CSP-{filt}']), filt)
#ax2.set_xlabel("GAIA DES g-i")

plt.legend()
plt.savefig("bla.pdf")
