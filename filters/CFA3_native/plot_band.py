import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

files = ["SNLS3_4shooter2_V.dat", "SNLS3_4shooter2_B.dat", "SNLS3_4shooter2_R.dat", "SNLS3_4shooter2_I.dat"]

lazy = ["V", "B", "R", "I"]

#dic = {"U":3562.1, "B":4355.83, "V":5409.74, "r":6242.35, "i":7674.08} #KCAM vals
dic = {"U":3562.1, "B":4355.83, "V":5409.74, "R":6242.35, "I":7674.08} #assuming the same for CFA3S 

for n,filt in enumerate(files):
    df = pd.read_csv(filt, sep=r"\s+", names=["wav", "trans"])

    waveeff = (dic[lazy[n]])

    print(filt, waveeff)

    df['trans'] = df.trans*(waveeff/df.wav)**-1

    df.to_csv(filt+"_weighted", index=False, float_format='%g', sep=" ", header=False)
