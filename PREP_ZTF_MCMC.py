import pandas as pd
import numpy as np
import sys
sys.path.insert(1, 'scripts/')
import write_obs_output as wo

system = 'ZTF'

#5YR,ZTF_g,ZTF_r,ZTF_i,PS1_g,PS1_r,PS1_i,PS1_z,RA,DEC

from astropy.table import Table
t = Table.read('newcatalog/PS1_in_ZTF.fits')

print(t.columns)

xx = (t['ZTF_ZTF_g']>10) &(t['ZTF_ZTF_r']>10) &(t['ZTF_ZTF_i']>10) & (t['gMeanPSFMag'] > 10)

colnames = ['%s-g'%system,'%s-r'%system,'%s-i'%system,
            'PS1-g','PS1-r','PS1-i','PS1-z','RA','DEC']
collists = [t['ZTF_ZTF_g'][xx],t['ZTF_ZTF_r'][xx],t['ZTF_ZTF_i'][xx],
            t['gMeanApMag'][xx],t['rMeanApMag'][xx],t['iMeanApMag'][xx],t['zMeanApMag'][xx],
            t['raMean'][xx],t['decMean'][xx]]
outfile = 'output_observed_apermags/'+system+'_observed.csv'
wo.write(system,colnames,collists,outfile)

