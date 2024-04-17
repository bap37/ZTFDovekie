import pandas as pd
import numpy as np
import write_obs_output as wo

system = 'SDSS'

from astropy.table import Table
t = Table.read('newcatalog/PS1_in_SDSS.fits')
print(t.columns)
xx = (t['SDSS_g_mmu']>10) & (t['gMeanApMag'] > 10)



colnames = ['%s-g'%system,'%s-r'%system,'%s-i'%system,'%s-z'%system,
            'PS1-g','PS1-r','PS1-i','PS1-z','RA','DEC']
collists = [t['SDSS_g_mmu'][xx],t['SDSS_r_mmu'][xx],t['SDSS_i_mmu'][xx],t['SDSS_z_mmu'][xx],
            t['gMeanApMag'][xx],t['rMeanApMag'][xx],t['iMeanApMag'][xx],t['zMeanApMag'][xx],
            t['raMean'][xx],t['decMean'][xx]]
outfile = 'output_observed_apermags/'+system+'_observed.csv'
wo.write(system,colnames,collists,outfile)

