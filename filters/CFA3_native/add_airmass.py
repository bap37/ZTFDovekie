import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

airmass = pd.read_csv("modtran_atm_stubbs_Jan21.dat", sep=r"\s+", header=None, names=['wl', 'trans', '?', 'trans_again?'])

#print(airmass)

filters = [f'SNLS3_4shooter2_{bp}' for bp in 'BVRI']

interp = interp1d(airmass.wl*10,airmass.trans)

atmo=1

for f in filters:
    df = pd.read_csv(f'{f}.dat', sep=r'\s+', comment="#", names=['wavelength', 'trans'])

    aim = (interp(df.wavelength.values))

    df['trans'] *= aim**atmo

    df.to_csv(f"{f}_atmo_{atmo}.dat", index=False, float_format='%g', header=False, sep=" ")
