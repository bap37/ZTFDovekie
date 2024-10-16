import pandas as pd 
import numpy as np

from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"  # Select Data Release 2
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Reselect Data Release 3, default

import astropy.units as u
from astropy.coordinates import SkyCoord

#######################

def GAIA_query(RA,DEC, radius=0.6*0.000277778):
    coord = SkyCoord(ra=RA, dec=DEC, unit=(u.degree, u.degree), frame='icrs')
    j = Gaia.cone_search_async(coord, radius=u.Quantity(radius, u.deg))
    #INFO: Query finished. [astroquery.utils.tap.core]
    r = j.get_results()
    r.pprint()
    
    mag_g, mag_rp, mag_bp = -999, -999, -999
    
    if len(r) == 0:
        return mag_g, mag_rp, mag_bp
    
    if r['phot_proc_mode'].value[0] == 0:
        mag_g = r['phot_g_mean_mag'].value[0]
        mag_rp = r['phot_rp_mean_mag'].value[0]
        mag_bp = r['phot_bp_mean_mag'].value[0]
    
    return mag_g, mag_rp, mag_bp

###################

def GAIA_merge(df):
    GAIA_g = []
    GAIA_bp = []
    GAIA_rp = []

    for index, row in df.iterrows():
        RA = row.RA
        DEC = row.DEC
        g, rp, bp = GAIA_query(RA,DEC)
        GAIA_g.append(g)
        GAIA_rp.append(rp)
        GAIA_bp.append(bp)

    df['GAIA-g'] = np.asarray(GAIA_g)
    df['GAIA-r'] = np.asarray(GAIA_rp)
    df['GAIA-b'] = np.asarray(GAIA_bp)
    return df

listsurveys = ['output_observed_apermags+AV/PS1SN_observed.csv']

for filename in listsurveys:
    df = pd.read_csv(filename)
    df.to_csv(f'{filename}.bak', header=True, index=False, float_format='%g')

    GAIA_merge(df)
    df.to_csv(f'{filename}', header=True, index=False, float_format='%g')
