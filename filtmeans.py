import pandas as pd
import numpy as np


survs = [
  'PS1',
  'DES',
  'ZTF',
  'CSP',
  'SDSS',
  'Foundation',
  'SNLS'
]

filtpaths = [
  'filters/PS1s_RS14_PS1_tonry/',
  'filters/DES-SN3YR_DECam/',
  'filters/ZTF/',
  'filters/CSP/',
  'filters/SDSS_Doi2010_CCDAVG/',
  'filters/PS1s_RS14_PS1_tonry/',
  'filters/SNLS3-Megacam/',
]

filttranss = [
  ['g_filt_revised.txt','r_filt_tonry.txt','i_filt_tonry.txt','z_filt_tonry.txt'],
    ['DECam_g.dat','DECam_r.dat','DECam_i.dat','DECam_z.dat'],
    ['g_ztf+25.dat' ,'r_ztf.dat', 'i_ztf.dat'],
  ['u_tel_ccd_atm_ext_1.2.dat','g_tel_ccd_atm_ext_1.2.dat','r_tel_ccd_atm_ext_1.2.dat','i_tel_ccd_atm_ext_1.2.dat','B_tel_ccd_atm_ext_1.2.dat','V_LC9844_tel_ccd_atm_ext_1.2.dat','V_LC3009_tel_ccd_atm_ext_1.2.dat','V_LC3014_tel_ccd_atm_ext_1.2.dat','V_LC9844_tel_ccd_atm_ext_1.2.dat','Y_SWO_TAM_scan_atm.dat','Y_DUP_TAM_scan_atm.dat','Jrc1_SWO_TAM_scan_atm.dat','Jrc2_SWO_TAM_scan_atm.dat','H_SWO_TAM_scan_atm.dat','H_DUP_TAM_scan_atm.dat'],
  ['G.dat','R.dat','I.dat','Z.dat','g.dat','r.dat','i.dat','z.dat'],
  ['g_filt_revised.txt','r_filt_tonry.txt','i_filt_tonry.txt','z_filt_tonry.txt'],
   ['effMEGACAM-g.dat','effMEGACAM-r.dat','effMEGACAM-i.dat','effMEGACAM-z.dat'],
]

obsfiltss = [
  ['g','r','i','z'],
  ['g','r','i','z'],
  ['g', 'r', 'i'],
  ['u','g','r','i','B','V','o','m','n','Y','y'],
  ['G','R','I','Z','g','r','i','z'],
  ['g','r','i','z'],
  ['g','r','i','z'],
]

out = open('filter_means.csv','w')
out.write('SURVEYFILTER,MEANLAMBDA \n')
for fp,fts,ofs,surv in zip(filtpaths,filttranss,obsfiltss,survs):
    for ft,of in zip(fts,ofs):
        d = pd.read_csv(fp+'/'+ft,names=['wavelength', 'trans'],delim_whitespace=True,comment='#')
        #print(d['wavelength'],d['trans'])
        print(ft,round(np.average(d['wavelength'],weights=d['trans'])))
        out.write(surv+str(of)+','+str(round(np.average(d['wavelength'],weights=d['trans'])))+'\n')

out.close()
