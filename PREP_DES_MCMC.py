import pandas as pd
import numpy as np
import sys
sys.path.insert(1, 'scripts/')
import write_obs_output as wo

system = 'DES'

from astropy.table import Table
t = Table.read('newcatalog/PS1_in_DES5YR.fits')
print(t.columns)

t = t[0:2000]

xx = (t['DES_MAG_STD_G'].astype(float) > 10) & (t['gMeanPSFMag'].astype(float) > 10)


colnames = ['%s-g'%system,'%s-r'%system,'%s-i'%system,'%s-z'%system,
            'PS1-g','PS1-r','PS1-i','PS1-z','RA','DEC']
collists = [t['DES_MAG_STD_G'][xx],t['DES_MAG_STD_R'][xx],t['DES_MAG_STD_I'][xx],t['DES_MAG_STD_Z'][xx],
            t['gMeanApMag'][xx],t['rMeanApMag'][xx],t['iMeanApMag'][xx],t['zMeanApMag'][xx],
            t['raMean'][xx],t['decMean'][xx]]
outfile = 'output_observed_apermags/'+system+'_observed.csv'
wo.write(system,colnames,collists,outfile)

