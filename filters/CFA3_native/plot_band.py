import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

files = ["SNLS3_Keplercam_V_modtran.dat", "SNLS3_Keplercam_B_modtran.dat", "SNLS3_Keplercam_U_modtran.dat", "SNLS3_Keplercam_r_modtran.dat", "SNLS3_Keplercam_i_modtran.dat"]

lazy = ["V", "B", "U", "r", "i"]

dic = {"U":3562.1, "B":4355.83, "V":5409.74, "r":6242.35, "i":7674.08}

for n,filt in enumerate(files):
    df = pd.read_csv(filt, sep=r"\s+", names=["wav", "trans"])

    waveeff = dic[lazy[n]]

    print(filt, waveeff)

    df['trans'] = df.trans*(waveeff/df.wav)

    df.to_csv(filt+"_weighted", index=False, float_format='%g', sep=" ", header=False)