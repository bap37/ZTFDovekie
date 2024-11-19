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
import re

from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"  # Select Data Release 2
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Reselect Data Release 3, default

import astropy.units as u
from astropy.coordinates import SkyCoord

jsonload = 'DOVEKIE_DEFS.yml' #where all the important but unwieldy dictionaries live
config = load_config(jsonload)

#######################


def integrate_spectrum(SOURCE_ID, filters, filtpath, sampling, surv='None', shift=0):

    #opening the relevant spectra (which has regular sampling)
    filename = f'spectra/{surv}/{SOURCE_ID}.csv'
    _ = open(filename, 'r')
    flux = re.findall(r'"([^"]*)"', _.read(), re.U) #I don't understand this insane storage type so we're grabbing the first entry in quotes which is the flux

    flux = flux[0]
    flux = flux.replace("(", '').replace(")", '').split(",") #and removing random crap and turning it into a list
    flux = np.array(flux, dtype=float) * 100 #100 is the conversion factor for GAIA spectra (which use nm instead of AA)

    band_weights, zps = prep_filts(sampling, filters, filtpath, shift=shift)

    oldfluxunit = u.W / (u.nm * u.m ** 2)
    newfluxunit = u.erg / (u.cm ** 2 * u.s * u.AA)
    #(W m-2 Hz-1)
    oldfluxes = flux*oldfluxunit #factor of 100 is because Gaia expects nanometers and we normally use angstroms; this is a conversion factor 
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

  msg = "Filter shift to implement. Input is an float, default is 0"
  parser.add_argument("--SHIFT", help=msg, type=float, default=0)

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

  shift = args.SHIFT/10 #converts to nm automatically 
  for ind, surv,filtpath,kcor,filters,obsfilts in zip(
        range(len(config['survs'])),config['survs'],config['filtpaths'],config['kcors'],config['filttranss'],config['obsfiltss']):
    print(ind,surv)
    if ind != float(index): continue

    #load in data, df, all that good stuff 
    df = pd.read_csv(f'output_observed_apermags+AV/{surv}_observed.csv')

    sampling = np.arange(336, 1021, 2)
    #load in filters etc here
    band_weights, zps = prep_filts(sampling, filters, filtpath, shift=shift)

    #generate empty bands and stuff and things.

    for n, row in df.iterrows():
        #check if source_id is real
        #if not, do nothing
        SOURCE_ID = row.SOURCEID
        if SOURCE_ID != -999: #handle -999 and nans
            #if it is, load spectra
            integrated_vals = integrate_spectrum(SOURCE_ID, filters, filtpath, sampling=sampling, surv=surv, shift=shift)
            #print(integrated_vals)
            wrapout = [f'GAIA_{surv}-'+_ for _ in obsfilts]
            for _ in range(len(wrapout)):
                df.loc[df.index[n], wrapout[_]] = integrated_vals[_]

    print("Done!")
    df.to_csv(f'output_observed_apermags+AV/{surv}_observed.csv', header=True, index=False, float_format='%g')
