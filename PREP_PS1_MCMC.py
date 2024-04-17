import pandas as pd
import numpy as np
import sys
sys.path.insert(1, 'scripts/')
import write_obs_output as wo

system = 'PS1SN'

from astropy.table import Table
t = Table.read('newcatalog/PS1_in_oldPS1.fits')

xx = (t['oldPS1_g']>10) &(t['oldPS1_r']>10) &(t['oldPS1_i']>10) & (t['oldPS1_z']>10) & (t['gMeanPSFMag'] > 10)

colnames = ['%s-g'%system,'%s-r'%system,'%s-i'%system,'%s-z'%system,
            'PS1-g','PS1-r','PS1-i','PS1-z','RA','DEC']
collists = [t['gMeanPSFMag'][xx],t['rMeanPSFMag'][xx],t['iMeanPSFMag'][xx],t['zMeanPSFMag'][xx],
            t['gMeanApMag'][xx],t['rMeanApMag'][xx],t['iMeanApMag'][xx],t['zMeanApMag'][xx],
            t['raMean'][xx],t['decMean'][xx]]
outfile = 'output_observed_apermags/'+system+'_observed.csv'
wo.write(system,colnames,collists,outfile)

