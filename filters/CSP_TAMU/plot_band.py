import pandas as pd
import matplotlib.pyplot as plt

files = ['g_tel_ccd_atm_ext_1.2.dat','r_tel_ccd_atm_ext_1.2.dat', 'i_tel_ccd_atm_ext_1.2.dat', 'B_tel_ccd_atm_ext_1.2.dat', "u_tel_ccd_atm_ext_1.2.dat"]
files = ['V_LC9844_tel_ccd_atm_ext_1.2.dat']

dic = {"B":4392.19, "g":4733.56, "r":6184.75, "u":3671.61, "i":7585.26, "V":5358.25}

for filt in files:
    df = pd.read_csv(filt, sep=r"\s+", names=['wav', 'trans'])

    #plt.figure()
    #plt.scatter(df.wav, df.trans, label="og")
    #plt.scatter(df.wav, df.trans*(5358.25/df.wav), label='weighted')
    #plt.legend()

    #plt.savefig("bla.pdf")

    waveeff = dic[filt[0]]
    X = 1
    df['trans'] = df.trans*((waveeff/df.wav)**X)

    df.to_csv(filt+"_mod_X_"+str(X), index=False, float_format='%g', sep=" ", header=False)