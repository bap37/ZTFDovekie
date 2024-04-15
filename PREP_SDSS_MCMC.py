import pandas as pd
import numpy as np
import write_obs_output as wo

system = 'SDSS'

from astropy.table import Table
t = Table.read('newcatalog/PS1_in_SDSS.fits')
print(t.columns)
xx = (t['SDSS_g_mmu']>10) & (t['gMeanApMag'] > 10)



colnames = ['%s_AB_g'%system,'%s_AB_r'%system,'%s_AB_i'%system,'%s_AB_z'%system,
            'PS1_g','PS1_r','PS1_i','PS1_z','RA','DEC']
collists = [t['SDSS_g_mmu'][xx],t['SDSS_r_mmu'][xx],t['SDSS_i_mmu'][xx],t['SDSS_z_mmu'][xx],
            t['gMeanApMag'][xx],t['rMeanApMag'][xx],t['iMeanApMag'][xx],t['zMeanApMag'][xx],
            t['raMean'][xx],t['decMean'][xx]]
outfile = 'output_observed_apermags/'+system+'_AB_observed.csv'
wo.write(system+'_AB',colnames,collists,outfile)

