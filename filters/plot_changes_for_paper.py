import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
import matplotlib

def DISCRETE_CMAP(CMAP, bins):
    cmap = plt.get_cmap(CMAP, bins)
    colours = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    return colours

og_files = ['CFA3_native/SNLS3_4shooter2_V.dat', 'CFA3_native/SNLS3_Keplercam_V_modtran.dat', 
            'CSP_TAMU/B_tel_ccd_atm_ext_1.2.dat',
            'CSP_TAMU/V_LC9844_tel_ccd_atm_ext_1.2.dat',
            'SDSS_Doi2010_CCDAVG/g.dat',
            'SNLS3-Megacam/effMEGACAM-g.dat',
            'SNLS3-Megacam/effMEGACAM-r.dat',
            'SNLS3-Megacam/effMEGACAM-i.dat',
            'SNLS3-Megacam/effMEGACAM-z.dat',
]

new_files = ['CFA3_native/SNLS3_4shooter2_V.dat+-20', 'CFA3_native/SNLS3_Keplercam_V_modtran.dat+-30', 
             'CSP_TAMU/B_tel_ccd_atm_ext_1.2.dat_mod_inv',
             'CSP_TAMU/V_weighted.dat',
             'SDSS_Doi2010_CCDAVG/g.dat+15',
             'SNLS3-Megacam/effMEGACAM-g.dat+30',
             'SNLS3-Megacam/effMEGACAM-r.dat+30',
             'SNLS3-Megacam/effMEGACAM-i.dat+30',
             'SNLS3-Megacam/effMEGACAM-z.dat+30',

]

labels = ['CfA3S-V', "CfA3K-V", "CSP-B", "CSP-V", "SDSS-g", "SNLS-g", "SNLS-r", "SNLS-i", "SNLS-z"]

colours = DISCRETE_CMAP('cet_CET_CBL1', 4)

fig, axs = plt.subplots(2,5, figsize=(12,6))

for n, ax in enumerate(axs.flat):
    try:
        old_filt = og_files[n]
        new_filt = new_files[n]

    except IndexError:
        continue

    oldf = pd.read_csv(old_filt, names=['wave', 'trans'], sep=r'\s+', comment="#")
    newf = pd.read_csv(new_filt, names=['wave', 'trans'], sep=r'\s+', comment="#")

    ax.plot(oldf.wave, oldf.trans, c='#b19d59', lw=3, ls='--', zorder=4)
    ax.plot(newf.wave, newf.trans, c=colours[1], lw=3, zorder=3)

    ax.plot(oldf.wave, oldf.trans, alpha=0, c="white", label=labels[n], zorder=1)

    ax.legend(frameon=False, fontsize=15, loc='best')

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

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig("Filter_changes.pdf", bbox_inches="tight")