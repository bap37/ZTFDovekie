import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
import matplotlib

def DISCRETE_CMAP(CMAP, bins):
    cmap = plt.get_cmap(CMAP, bins)
    colours = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    return colours

og_files = ['CFA3_native/SNLS3_Keplercam_B_modtran.dat', 
            'CFA3_native/SNLS3_Keplercam_V_modtran.dat', 
            'CFA3_native/SNLS3_Keplercam_r_modtran.dat',
            'CFA3_native/SNLS3_Keplercam_i_modtran.dat',
            'CFA3_native/SNLS3_4shooter2_B.dat',
            'CFA3_native/SNLS3_4shooter2_V.dat',
            'CFA3_native/SNLS3_4shooter2_R.dat',
            'CFA3_native/SNLS3_4shooter2_I.dat',
]

new_files = ['CFA3_native/KC_B_modtran.dat', 
             'CFA3_native/KC_V_modtran.dat', 
             'CFA3_native/KC_r_modtran.dat', 
             'CFA3_native/KC_i_modtran.dat', 
             'CFA3_native/4sh_B_modtran.dat',
             'CFA3_native/4sh_V_modtran.dat',
             'CFA3_native/4sh_R_modtran.dat',
             'CFA3_native/4sh_I_modtran.dat',
]

labels = ['CFA3K-B', 'CFA3K-V', 'CFA3K-r', 'CFA3K-i',
          'CFA3S-B', 'CFA3S-V', 'CFA3S-R', 'CFA3S-I']

colours = DISCRETE_CMAP('cet_CET_CBL1', 4)

fig, axs = plt.subplots(3,3, figsize=(10,10))

for n, ax in enumerate(axs.flat):
    try:
        old_filt = og_files[n]
        new_filt = new_files[n]

    except IndexError:
        ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        ax.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)
        ax.set_yticks([])
        ax.set_xticks([])
        continue

    oldf = pd.read_csv(old_filt, names=['wave', 'trans'], sep=r'\s+', comment="#")
    newf = pd.read_csv(new_filt, names=['wave', 'trans'], sep=r'\s+', comment="#")

    ax.plot(oldf.wave, oldf.trans/np.amax(oldf.trans), c='#b19d59', lw=3, ls='--', zorder=4)
    ax.plot(newf.wave, newf.trans/np.amax(newf.trans), c=colours[1], lw=3, zorder=3)

    ax.plot(oldf.wave, oldf.trans, alpha=0, c="white", label=labels[n], zorder=1)

    ax.legend(frameon=False, fontsize=14, loc='center')

    #aesthetics
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)
    ax.set_yticks([])
    ax.set_ylabel("Efficiency", c="dimgrey", fontsize=12)
    ax.set_xlabel(r"Wavelength ($\AA$)", c="dimgrey", fontsize=12)
    minval = min(np.amin(newf.wave), np.amin(oldf.wave))
    maxval = max(np.amax(newf.wave), np.amax(oldf.wave))
    ax.set_xticks(np.arange(minval, maxval, 1000), labels=np.around((np.arange(minval, maxval, 1000)),2), c="dimgrey")

ax = axs.flat[-1]
ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
ax.plot(-1,-1, c="#b19d59", lw=3, ls='--', label="Old Filter")
ax.plot(-1,-1, c=colours[1], lw=3, label="New Filter")
ax.legend(frameon=False, fontsize='x-large', labelcolor=["#b19d59", colours[1]], loc="center")
ax.set_xlim([-0.1, 0.1])
ax.set_ylim([-0.1, 0.1])
ax.set_yticks([])
ax.set_xticks([])

plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.savefig("Filter_changes_CFA3.pdf", bbox_inches="tight")
