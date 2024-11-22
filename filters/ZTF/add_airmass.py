import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

airmass = pd.read_csv("Buton.dat", sep=r"\s+")

#print(airmass)

filters = ['ztfg1.dat', 'ztfg2.dat', 'ztfr1.dat', 'ztfr2.dat', 'ztfi1.dat', 'ztfi2.dat']

interp = interp1d(airmass.Wavelength,airmass.Extinction)

atmo=1

for f in filters:
    df = pd.read_csv(f, sep=r'\s+', comment="#", names=['wavelength', 'trans'])
    
    aim = (interp(df.wavelength.values))

    df['trans'] *= (1-aim*atmo)

    df.to_csv(f"{f}_atmo_{atmo}", index=False, float_format='%g', header=False, sep=" ")
