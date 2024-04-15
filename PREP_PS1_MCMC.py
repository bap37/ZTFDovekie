import pandas as pd
import numpy as np
import sys
sys.path.insert(1, 'scripts/')
import write_obs_output as wo

system = 'PS1'

from astropy.table import Table
t = Table.read('newcatalog/PS1_in_oldPS1.fits')

xx = (t['oldPS1_g']>10) &(t['oldPS1_r']>10) &(t['oldPS1_i']>10) & (t['oldPS1_z']>10) & (t['gMeanPSFMag'] > 10)

colnames = ['%s_AB_g'%system,'%s_AB_r'%system,'%s_AB_i'%system,'%s_AB_z'%system,
            'PS1_g','PS1_r','PS1_i','PS1_z','RA','DEC']
collists = [t['gMeanPSFMag'][xx],t['rMeanPSFMag'][xx],t['iMeanPSFMag'][xx],t['zMeanPSFMag'][xx],
            t['gMeanApMag'][xx],t['rMeanApMag'][xx],t['iMeanApMag'][xx],t['zMeanApMag'][xx],
            t['raMean'][xx],t['decMean'][xx]]
outfile = 'output_observed_apermags/'+system+'_AB_observed.csv'
wo.write(system+'_AB',colnames,collists,outfile)

