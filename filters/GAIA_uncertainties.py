import pandas as pd 
import glob
import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

prior = 500

def DISCRETE_CMAP(CMAP, bins):
    cmap = plt.get_cmap(CMAP, bins)
    colours = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    return colours

files = glob.glob('../filter_characterisation/*GAIA*')
files = [i for i in files if not ('ZTF' in i)]

fig, axs = plt.subplots(3,3, figsize=(12,12))

for n,file in enumerate(files):
    
    df = pd.read_csv(file)
    label = file.split('/')[-1].split('_')[0]
    #print(label)
    
    colours = DISCRETE_CMAP('cet_CET_CBL1', len(list(df))-1)
    
    filters = list(df)[1:-1]
    #print(filters)
    
    
    xnew = np.linspace(-50,50, 300)
    ax = axs.flat[n]
    for _,filt in enumerate(filters):
        if filt in ['m', 'n', 'o', 'g']: continue 
        minimum =  np.amin(df[filt]**2)
        y = (np.abs(df[filt])**2 + df.shifts**2/(prior**2)) 

        #ax.plot(df.shifts, y - minimum, c=colours[_], zorder=2, label=filt) 

        z = np.polyfit(df.shifts, y, 3)
        p = np.poly1d(z)
        x = np.linspace(-50,50,1000)
        minimum = np.amin(p(x))
        ax.plot(x, p(x) - minimum, c=colours[_], zorder=2, label=filt, lw=3)
        
    ax.legend(frameon=False)
    ax.set_ylim(0.,8)
    ax.set_yticks(np.arange(0,8,1), labels=np.arange(0,8,1),  c="dimgrey")
    ax.set_xlim(-50,50)
    ax.text(x=20, y=3.5, s=label, c="dimgrey", fontsize=15, zorder=5)
    ax.set_xticks(np.arange(-50,50,5), labels=np.arange(-50,50,5), rotation=90, fontsize=10, c="dimgrey")
    
    #ax.axhline(1, ls=":", c="dimgrey", zorder=2)
    
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.tick_params(axis="both", which="both", bottom=False, colors="dimgrey")
    ax.set_ylabel(r"$\Delta \chi^2$", c="dimgrey", size=13)
    ax.set_xlabel(r"Shift ($\AA$)", c="dimgrey")
    ax.grid(zorder=1, alpha=.5)
    
    
plt.savefig("ONE_SIGMA_GAIA.pdf", bbox_inches="tight")
