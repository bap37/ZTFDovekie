import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dfN = pd.read_csv('CFA3S_observed.csv', sep=r'\s+')
dfO = pd.read_csv('FRAG_CFA3S.csv', sep=r'\s+')

df = pd.merge(dfN, dfO, on=['standard'])

print(list(df))

plt.figure()
plt.scatter(df['CFA3S-B'] + df['CFA3_4shooter_native-B'], df['CFA3S-I'] + df['CFA3_4shooter_native-I'])
plt.savefig("bla.pdf") 
