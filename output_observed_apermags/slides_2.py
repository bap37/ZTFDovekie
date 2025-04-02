import glob
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import skyproj 
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams["font.family"] = "Helvetica"


import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic


fig = plt.figure(1, figsize=(16, 12))
ax = fig.add_subplot(111)
#sp = skyproj.McBrydeSkyproj(ax=ax, c="tab:orange")
sp = skyproj.EqualEarthSkyproj(ax=ax, c="tab:orange")

df = pd.read_csv('ALL_CALSPEC_RADEC.tsv', sep=r'\s+')

for n,q in df.iterrows():
    c = SkyCoord(ra=q.RA, dec=q.Dec, unit=(u.hourangle, u.deg))

    sp.plot(c.ra, c.dec, marker=".", lw=0, c='k')
sp.draw_milky_way(label='Milky Way')
plt.savefig("CALSPEC.png", transparent=True)
