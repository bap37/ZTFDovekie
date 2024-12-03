import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("DES_observed.csv")
df = df.loc[df['GAIA_DES-r'] > -900]

print(len(df))

mags = np.arange(14,20,1)

for filt in ['GAIA_DES-g', 'GAIA_DES-r', 'GAIA_DES-i', 'GAIA_DES-z']:

    for n in range(len(mags)):
        try:
            dft = df.loc[(df[filt] >= mags[n]) & (df[filt] < mags[n+1])]
            dft = dft.dropna()
            scatter = np.std(dft[filt.replace('GAIA_', '')] - dft[filt].values)
            print(f"For filter {filt}, between {mags[n]} and {mags[n+1]} mags, the scatter is {scatter} with {len(dft)} stars")
        except IndexError:
            continue
