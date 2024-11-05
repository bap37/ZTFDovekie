import pandas as pd 
import numpy as np
from gaiaxpy import calibrate
import astropy.constants as const
from scipy.integrate import simpson
from gaiaxpy import plot_spectra
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, 'scripts/')
from queryhelpers2 import *
import argparse

from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"  # Select Data Release 2
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Reselect Data Release 3, default

import astropy.units as u
from astropy.coordinates import SkyCoord

jsonload = 'DOVEKIE_DEFS.yml' #where all the important but unwieldy dictionaries live
config = load_config(jsonload)

#######################

def GAIA_query(RA,DEC, radius=2*0.000277778):
    coord = SkyCoord(ra=RA, dec=DEC, unit=(u.degree, u.degree), frame='icrs')
    j = Gaia.cone_search_async(coord, radius=u.Quantity(radius, u.deg))
    #INFO: Query finished. [astroquery.utils.tap.core]
    r = j.get_results()
    r.pprint()
    if len(r) < 1:
        return -999
    SOURCE_ID = r['SOURCE_ID'].value[0]

    query = f"SELECT TOP 1 \
    source_id, ra, dec, pmra, pmdec, parallax \
    FROM gaiadr3.gaia_source \
    WHERE gaiadr3.gaia_source.source_id = {SOURCE_ID} \
    AND has_xp_continuous = 'True'"

    job     = Gaia.launch_job_async(query)
    results = job.get_results()

    r.pprint()

    print(f'Table size (rows): {len(results)}')

    if len(results) != 0:
        return SOURCE_ID
        
    return -999

def search_spectrum(SOURCE_ID, filters, filtpath, sampling):
    calibrated_spectra, sampling = calibrate([str(SOURCE_ID)], sampling=sampling, save_file=False) #output_file='my_output_name'

    # Do not show the legend as there's only one source in the data
    plt.figure()
    plot_spectra(calibrated_spectra, sampling=sampling, legend=False, 
                 output_path="plots/GAIA/", output_file=f"{str(SOURCE_ID)}", format="pdf")
    plt.close()

    band_weights, zps = prep_filts(sampling, filters, filtpath)

    oldfluxunit = u.W / (u.nm * u.m ** 2)
    newfluxunit = u.erg / (u.cm ** 2 * u.s * u.AA)
    #(W m-2 Hz-1)
    oldfluxes = calibrated_spectra.flux.values[0]*oldfluxunit*100 #factor of 100 is because Gaia expects nanometers and we normally use angstroms; this is a conversion factor 
    newfluxes = oldfluxes.to(newfluxunit).value

    seds = get_model_mag(newfluxes,band_weights, zps)

    print(seds)

    return seds 

###################


def get_args():
  parser = argparse.ArgumentParser()

  msg = "HELP menu for config options"

  msg = "Default -9, which will only output the list of available surveys. Integer number corresponding to the survey in the code printout."
  parser.add_argument("--SURVEY", help=msg, type=int, default=-9)
  args = parser.parse_args()
  return args


parallel = '1'
if __name__ == '__main__':


  args = get_args()
  if args.SURVEY == -9:
    for ind, surv,kcorpath,kcor,shiftfilts,obsfilts in zip(
        range(len(config['survs'])),config['survs'],config['kcorpaths'],config['kcors'],config['shiftfiltss'],config['obsfiltss']):
      print(ind,surv)
    print('please call with integer arg like so:\npython loopsyntheticmags.py X')
    sys.exit()
  else:
    index = args.SURVEY

  for ind, surv,filtpath,kcor,filters,obsfilts in zip(
        range(len(config['survs'])),config['survs'],config['filtpaths'],config['kcors'],config['filttranss'],config['obsfiltss']):
    print(ind,surv)
    if ind != float(index): continue

    #load in data, df, all that good stuff 
    df = pd.read_csv(f'output_observed_apermags+AV/{surv}_observed.csv')
    for tempfilt in obsfilts:
        df[f'GAIA_{surv}-{tempfilt}'] = -999

    sampling = np.arange(336, 1021, 2)
    #load in filters etc here
    band_weights, zps = prep_filts(sampling, filters, filtpath)

    #generate empty bands and stuff and things.

    for n, row in df.iterrows():
        RA = row.RA
        DEC = row.DEC
        SOURCE_ID = GAIA_query(RA,DEC, radius=0.6*0.000277778)
        if SOURCE_ID != -999:
            #import pdb;pdb.set_trace()
            integrated_vals = search_spectrum(SOURCE_ID, filters, filtpath, sampling=sampling)
            wrapout = [f'GAIA_{surv}-'+_ for _ in obsfilts]
            for _ in range(len(wrapout)):
                df.loc[df.index[n], wrapout[_]] = integrated_vals[_]

    # if the query search returns -9 then skip


    df.to_csv(f'output_observed_apermags+AV/{surv}_observed.csv', header=True, index=False, float_format='%g')
