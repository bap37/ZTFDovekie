import pandas as pd 
import numpy as np
from gaiaxpy import calibrate
import astropy.constants as const
from scipy.integrate import simpson
from gaiaxpy import plot_spectra
import matplotlib.pyplot as plt
import yaml, os


import astropy.units as u
from astropy.coordinates import SkyCoord

def load_config(config_path):
    with open(config_path, "r") as cfgfile:
        config = yaml.load(cfgfile, Loader=yaml.FullLoader)
    return config

def prep_config(config):
    survmap = config['survmap'] 
    survmap4shift = config['survmap4shift']
    survfiltmap = config['survfiltmap']
    obssurvmap = config['obssurvmap']
    revobssurvmap = config['revobssurvmap']
    revobssurvmapforsnana = config['revobssurvmapforsnana']
    survcolormin = config['survcolormin']
    survcolormax = config['survcolormax']
    synth_gi_range = config['synth_gi_range']
    obsfilts = config['obsfilts']
    snanafilts = config['snanafilts']
    snanafiltsr = config['snanafiltsr']
    relativeweights = config['relativeweights']
    errfloors = config['errfloors']
    whitedwarf_obs_loc = config['whitedwarf_obs_loc']
    return survmap, survmap4shift, survfiltmap, obssurvmap, revobssurvmap, revobssurvmapforsnana, survcolormin, survcolormax, synth_gi_range, obsfilts, snanafilts, snanafiltsr, relativeweights, errfloors,  config['target_acceptance'] , config['n_burnin'], whitedwarf_obs_loc


jsonload = 'DOVEKIE_DEFS.yml' #where all the important but unwieldy dictionaries live
config = load_config(jsonload)
survmap, survmap4shift, survfiltmap, obssurvmap, revobssurvmap, revobssurvmapforsnana, survcolormin, survcolormax, synth_gi_range, obsfilts, snanafilts, snanafiltsr, relativeweights, errfloors,target_acceptance , n_burnin, bboyd_loc = prep_config(config)

def f_lam(l):
    f = (const.c.to('AA/s').value / 1e23) * ((l) ** -2) * 10 ** (-48.6 / 2.5) * 1e23
    return f

def prep_filts(sampling, filters, filtpath, isgaia=True):

    wav = sampling #sampling 
    band_weights=[]
    zps=[]
    for i,filt in enumerate(filters):
        R = np.loadtxt(os.path.join(filtpath,filt))
        T = np.zeros(len(wav))

        if isgaia:
            R[:,0] *= 0.1
            R[:,1] *= 10
        lam = R[:, 0]
        R[:,1]*=lam

        min_id=np.argmin((wav-np.min(R[:,0]))**2)
        max_id=np.argmin((wav-np.max(R[:,0]))**2)
    
        #T[min_id:max_id]= np.interp(wav[min_id:max_id], R[:, 0], R[:, 1])
        T = np.interp(wav, R[:, 0], R[:, 1], left=0, right=0)

        dlambda = np.diff(wav)
        dlambda = np.r_[dlambda, dlambda[-1]]

        num =  T * dlambda
        denom = np.sum(num)
        band_weight = num / denom
        band_weights.append(band_weight)

        zp_sed = f_lam(lam)

        int1 = simpson( zp_sed * R[:, 1], lam)
        int2 = simpson( R[:, 1], lam)
        zp = 2.5 * np.log10(int1 / int2)
        zps.append(zp)

    band_weights = np.array(band_weights)
    zps = np.array(zps)
    return band_weights, zps

def get_model_mag(flux_grid, band_weights, zps):
    model_flux = band_weights @ flux_grid

    model_mag = -2.5 * np.log10(model_flux)+zps 
    return model_mag
