import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("SDSS_observed.csv")
df = df.loc[df['GAIA_SDSS-g'] > -900]

fig, ax = plt.subplots()

ax.scatter(df['PS1-g'] - df['PS1-i'], df['PS1-g'] - df['SDSS-g'], c="orange", label="reference=PS1")
ax.set_xlabel("PS1 g-i")

ax.scatter(df['PS1-g'] - df['PS1-i'], df['GAIA_SDSS-g'] - df['SDSS-g'], c="blue", label="reference=GAIA")
#ax2.set_xlabel("GAIA DES g-i")

plt.legend()
plt.savefig("bla.pdf")
