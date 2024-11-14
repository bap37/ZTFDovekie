import pandas as pd
import matplotlib.pyplot as plt

oldfile = 'effMEGACAM-g.dat'
newfile = 'newcam_g.dat'

df = pd.read_csv(oldfile, sep=r'\s+', comment="#", names=['wave', 'trans'])
dfn = pd.read_csv(newfile, sep=r'\s+', comment="#", names=['wave', 'trans'])

plt.figure()
plt.scatter(df.wave, df.trans, label="og")
plt.scatter(dfn.wave, dfn.trans, label="downloaded")
plt.legend()

plt.savefig("bla.pdf")
