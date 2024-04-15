import pandas as pd
import numpy as np
import sys
sys.path.insert(1, 'scripts/')
import write_obs_output as wo

system = 'DES5YR'

from astropy.table import Table
t = Table.read('newcatalog/PS1_in_DES5YR.fits')
print(t.columns)
xx = (t['DES_Y6A1_G'].astype(float) > 10) & (t['gMeanPSFMag'].astype(float) > 10)



colnames = ['%s_AB_g'%system,'%s_AB_r'%system,'%s_AB_i'%system,'%s_AB_z'%system,
            'PS1_g','PS1_r','PS1_i','PS1_z','RA','DEC']
collists = [t['DES_Y6A1_G'][xx],t['DES_Y6A1_R'][xx],t['DES_Y6A1_I'][xx],t['DES_Y6A1_Z'][xx],
            t['gMeanApMag'][xx],t['rMeanApMag'][xx],t['iMeanApMag'][xx],t['zMeanApMag'][xx],
            t['raMean'][xx],t['decMean'][xx]]
outfile = 'output_observed_apermags/'+system+'_AB_observed.csv'
wo.write(system+'_AB',colnames,collists,outfile)

