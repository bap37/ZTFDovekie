import pandas as pd
import numpy as np
import sys
sys.path.insert(1, 'scripts/')
import write_obs_output as wo

system = 'ZTF'

#print("Hello! Applying a cut on the two ZTF samples to limit the calibration stars to 1K.")

from astropy.table import Table
t = Table.read('newcatalog/PS1_in_ZTF_PSF.fits')

print(t.columns)

t = t[0:4000]

xx = (t['ZTF_ZTF-g']>10) &(t['ZTF_ZTF-r']>10) &(t['ZTF_ZTF-i']>10) & (t['gMeanPSFMag'] > 10)

colnames = ['ZTF-g','ZTF-r','ZTF-i', 'ZTF-G', 'ZTF-R', 'ZTF-I',
            'PS1-g','PS1-r','PS1-i','PS1-z','RA','DEC']
collists = [t['ZTF_ZTF-g'][xx],t['ZTF_ZTF-r'][xx],t['ZTF_ZTF-i'][xx], t['ZTF_ZTF-G'][xx],t['ZTF_ZTF-R'][xx],t['ZTF_ZTF-I'][xx],
            t['gMeanApMag'][xx],t['rMeanApMag'][xx],t['iMeanApMag'][xx],t['zMeanApMag'][xx],
            t['raMean'][xx],t['decMean'][xx]]

#collists = [t['ZTF_ZTF_g'][xx],t['ZTF_ZTF_r'][xx],t['ZTF_ZTF_i'][xx],
#            t['gMeanPSFMag'][xx],t['rMeanPSFMag'][xx],t['iMeanPSFMag'][xx],t['zMeanPSFMag'][xx],
#            t['raMean'][xx],t['decMean'][xx]]

outfile = 'output_observed_apermags/'+system+'_observed.csv'
#outfile = 'output_observed_apermags/'+system+'_PSF_observed.csv'

wo.write(system,colnames,collists,outfile)

print("Done with ZTF")

quit()

system = 'ZTFD'

t = Table.read('newcatalog/PS1_in_ZTFD.fits')

print(t.columns)
#t = t[0:1000]

xx = (t['ZTF_ZTF_g']>10) &(t['ZTF_ZTF_r']>10) &(t['ZTF_ZTF_i']>10) & (t['gMeanPSFMag'] > 10)

colnames = ['ZTFD-g','ZTFD-r','ZTFD-i',
            'PS1-g','PS1-r','PS1-i','PS1-z','RA','DEC']
collists = [t['ZTF_ZTF_g'][xx],t['ZTF_ZTF_r'][xx],t['ZTF_ZTF_i'][xx],
            t['gMeanApMag'][xx],t['rMeanApMag'][xx],t['iMeanApMag'][xx],t['zMeanApMag'][xx],
            t['raMean'][xx],t['decMean'][xx]]
collists = [t['ZTF_ZTF_g'][xx],t['ZTF_ZTF_r'][xx],t['ZTF_ZTF_i'][xx],
            t['gMeanPSFMag'][xx],t['rMeanPSFMag'][xx],t['iMeanPSFMag'][xx],t['zMeanPSFMag'][xx],
            t['raMean'][xx],t['decMean'][xx]]

outfile = 'output_observed_apermags/'+system+'_observed.csv'
#outfile = 'output_observed_apermags/'+system+'_PSF_observed.csv'
wo.write(system,colnames,collists,outfile)

