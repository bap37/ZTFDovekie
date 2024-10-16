import numpy as np
import pandas as pd

df = pd.read_csv("MASTER_FILTERS.DAT", sep=r"\s+")

print(df)

df = df.replace(99.99, 0)

df['WAVELEN'] = df['WAVELEN']*10

df.to_csv("g.dat", columns=['WAVELEN', 'G'], sep=" ", header=None, index=False)
df.to_csv("b.dat", columns=['WAVELEN', 'BP'], sep=" ", header=None, index=False)
df.to_csv("r.dat", columns=['WAVELEN', 'RP'], sep=" ", header=None, index=False)
