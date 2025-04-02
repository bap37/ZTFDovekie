import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
import matplotlib

def DISCRETE_CMAP(CMAP, bins):
    cmap = plt.get_cmap(CMAP, bins)
    colours = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    return colours

og_files = ['CFA3_native/SNLS3_Keplercam_V_modtran.dat', 
            'CFA3_native/SNLS3_4shooter2_B.dat',
            'CSP_TAMU/B_tel_ccd_atm_ext_1.2.dat',
            'CSP_TAMU/V_LC9844_tel_ccd_atm_ext_1.2.dat',
            'SDSS_Doi2010_CCDAVG/g.dat',
            'SNLS3-Megacam/effMEGACAM-g.dat',
            'SNLS3-Megacam/effMEGACAM-r.dat',
            'SNLS3-Megacam/effMEGACAM-i.dat',
            'SNLS3-Megacam/effMEGACAM-z.dat',
            'PS1_CFA4/cfa4_B_p1_modtran.dat',
            'PS1_CFA4/cfa4_V_p1_modtran.dat',
            'PS1_CFA4/cfa4_r_p1_modtran.dat',
            'PS1_CFA4/cfa4_i_p1_modtran.dat',
            'PS1_CFA4/cfa4_B_p2_modtran.dat',
            'PS1_CFA4/cfa4_V_p2_modtran.dat',
            'PS1_CFA4/cfa4_r_p2_modtran.dat',
            'PS1_CFA4/cfa4_i_p2_modtran.dat',
]

new_files = ['CFA3_native/SNLS3_Keplercam_V_modtran.dat+-30', 
             'CFA3_native/SNLS3_4shooter2_B.dat+70',
             'CSP_TAMU/B_tel_ccd_atm_ext_1.2.dat+70',
             'CSP_TAMU/V_LC9844_tel_ccd_atm_ext_1.2.dat+-50',
             'SDSS_Doi2010_CCDAVG/g.dat+15',
             'SNLS3-Megacam/effMEGACAM-g.dat+30',
             'SNLS3-Megacam/effMEGACAM-r.dat+30',
             'SNLS3-Megacam/effMEGACAM-i.dat+30',
             'SNLS3-Megacam/effMEGACAM-z.dat+30',
             'PS1_CFA4/cfa4_B_p1_modtran.dat_weighted',
             'PS1_CFA4/cfa4_V_p1_modtran.dat_weighted',
             'PS1_CFA4/cfa4_r_p1_modtran.dat_weighted',
             'PS1_CFA4/cfa4_i_p1_modtran.dat_weighted',
             'PS1_CFA4/cfa4_B_p2_modtran.dat_weighted+-20',
             'PS1_CFA4/cfa4_V_p2_modtran.dat_weighted+20',
             'PS1_CFA4/cfa4_r_p2_modtran.dat_weighted+20',
             'PS1_CFA4/cfa4_i_p2_modtran.dat_weighted+20',
]

labels = ["CfA3K-V", "CfA3S-B", "CSP-B", "CSP-V", "SDSS-g", "SNLS-g", "SNLS-r", "SNLS-i", "SNLS-z", 
          'CfA4P1-B', 'CfA4P1-V', 'CfA4P1-r', 'CfA4P1-i', 'CfA4P2-B', 'CfA4P2-V', 'CfA4P2-r', 'CfA4P2-i',]

colours = DISCRETE_CMAP('cet_CET_CBL1', 4)

fig, axs = plt.subplots(3,6, figsize=(12,10))

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
plt.savefig("Filter_changes.pdf", bbox_inches="tight")
