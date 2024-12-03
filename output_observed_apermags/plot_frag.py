import pandas as pd
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("DES_FRAG.csv")

d = pd.read_pickle("../newcatalog/Y6A1_FGCM_V3_3_1_PSF_ALL_STARS_small_July2020.pkl")
print(len(d))

d['radiff']  = d['RA']  - 150
d['decdiff'] = d['DEC'] - 2.3

print(d.loc[(abs(d.radiff) < 0.5) & abs(d.decdiff < 1)])

d.loc[(abs(d.radiff) < 0.5) & abs(d.decdiff < 1)].to_csv('../newcatalog/Y6A1_FGCM_V3_3_1_PSF_ALL_STARS_small_July2020-INSNFIELDS.csv')

quit()

plt.figure()
plt.scatter(df.RA.values, df.DEC.values, c='tab:blue')
plt.scatter(d['RA'], d['DEC'])

plt.xlabel("RA")
plt.ylabel("DEC")
plt.savefig("bla.pdf")
