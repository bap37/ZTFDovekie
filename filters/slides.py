import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd 
import glob
from scipy.interpolate import CubicSpline

def DISCRETE_CMAP(CMAP, bins):
    cmap = plt.get_cmap(CMAP, bins)
    colours = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    return colours

import colorcet as cc


files = glob.glob('../filter_characterisation/*PS1.csv')

files = [i for i in files if ('DES' in i)]

#this is the combined one

prior = 500

fig, ax = plt.subplots(1,1, figsize=(20,12))

for n, f in enumerate(files):

    
    gaiafile = f.replace('PS1','GAIA')
    if gaiafile == '../filter_characterisation/GAIASN_slope_deviations_GAIA.csv':
        gaiafile = '../filter_characterisation/PS1SN_slope_deviations_GAIA.csv'
    df = pd.read_csv(gaiafile) #Gaia
    df2 = pd.read_csv(f) #PS1
    label = f.split('/')[-1].split('_')[0]
    #print(label)
    
    colours = DISCRETE_CMAP('cet_CET_CBL1', len(list(df))-1)
    
    filters = list(df)[1:-1]
    #print(filters)
    
    for _,filt in enumerate(filters):
        if filt in ['m', 'n', 'o']: continue 
        if filt == 'g': 
            PS1weight = 1
            Gweight = 0
        else:
            PS1weight = 0.5
            Gweight = 0.5

        minimum = 0
        y = (Gweight*np.abs(df[filt])**2 + PS1weight*np.abs(df2[filt])**2) + df.shifts**2/(prior**2) - minimum 
        minimum = np.amin((Gweight*np.abs(df[filt])**2 + PS1weight*np.abs(df2[filt])**2)) 
        z = np.polyfit(df.shifts, y, 3)
        p = np.poly1d(z)
        x = np.linspace(-50,50,1000)
        minimum = np.amin(p(x))
        ax.plot(x, p(x) - minimum, c='k', zorder=2, label=filt, lw=_+1)

    ax.set_ylim(0.,8)
    ax.set_xlim(-50,50)
    ax.set_xticks(np.arange(-50,50,5), labels=np.arange(-50,50,5), rotation=90, fontsize=20, c="k")
    ax.set_yticks(np.arange(0,9,1), labels=np.arange(0,9,1), fontsize=20, c="k")
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.tick_params(axis="both", which="both", bottom=False, colors="k")    

plt.savefig("Just_DES_slide.png", bbox_inches="tight", transparent=True)
