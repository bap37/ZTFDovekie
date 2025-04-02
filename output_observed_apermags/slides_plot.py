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

files = glob.glob("*.csv")


fig = plt.figure(1, figsize=(16, 12))
ax = fig.add_subplot(111)
#sp = skyproj.McBrydeSkyproj(ax=ax, c="tab:orange")
sp = skyproj.EqualEarthSkyproj(ax=ax, c="tab:orange")

for f in files:
    df = pd.read_csv(f)
    label = f.split("_")[0]
    if ".csv" in label:
        continue
    elif "ZTF" in label:
        continue
    sp.plot(df.RA, df.DEC, marker=".", lw=0, zorder=1, alpha=0.5, c='k')

#plt.text(y=-10, x=50, s='DES', c="k", fontsize=12, zorder=100)
#plt.text(y=-8, x=-40, s='SDSS', c="k", fontsize=12, zorder=100)
#plt.text(y=45, x=-80, s="PS1", c="k", fontsize=12, zorder=100)
#plt.text(y=45, x=-130, s="SNLS", c="k", fontsize=12, zorder=100)
#plt.text(y=30, x=130, s="CFA3K", c="k", fontsize=12, zorder=100)
#plt.text(y=-30, x=160, s="CFA3S", c="k", fontsize=12, zorder=100)
#plt.text(y=0, x=0, s="CSP", c="k", fontsize=12, zorder=100)
#plt.legend()

sp.draw_milky_way(label='Milky Way')
#plt.legend()
    
plt.savefig("Surveys.png", transparent=True)
plt.close()

fig = plt.figure(1, figsize=(16, 12))
ax = fig.add_subplot(111)
#sp = skyproj.McBrydeSkyproj(ax=ax, c="tab:orange")
sp = skyproj.EqualEarthSkyproj(ax=ax, c="tab:orange")

df = pd.read_csv('ALL_CALSPEC_RADEC.tsv', sep=r'\s+')

sp.plot(df.RA, df.Dec, marker=".", lw=0, c='k')
sp.draw_milky_way(label='Milky Way')
plt.savefig("CALSPEC.png", transparent=True)
