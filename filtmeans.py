import pandas as pd
import numpy as np


survs = [
  'PS1',
  'DES',
  'ZTF',
  'CSP',
  'SDSS',
  'Foundation',
  'SNLS',
  'PS1SN',
  'CFA3S',
  'CFA3K',  
]

filtpaths = [
  'filters/PS1s_RS14_PS1_tonry/',
  'filters/DES-SN3YR_DECam/',
  'filters/ZTF/',
  'filters/CSP_TAMU/',
  'filters/SDSS_Doi2010_CCDAVG/',
  'filters/PS1s_RS14_PS1_tonry/',
  'filters/SNLS3-Megacam/',
  'filters/PS1s_RS14_PS1_tonry/',
  'filters/CFA3_native/',
  'filters/CFA3_native/',
]

filttranss = [
  ['g_filt_revised.txt','r_filt_tonry.txt','i_filt_tonry.txt','z_filt_tonry.txt'],
    ['DECam_g.dat','DECam_r.dat','DECam_i.dat','DECam_z.dat'],
    ['ztfg1.dat+-20' ,'ztfr1.dat', 'ztfi1.dat'],
  ['u_tel_ccd_atm_ext_1.2.dat','g_tel_ccd_atm_ext_1.2.dat','r_tel_ccd_atm_ext_1.2.dat','i_tel_ccd_atm_ext_1.2.dat','B_tel_ccd_atm_ext_1.2.dat','V_LC9844_tel_ccd_atm_ext_1.2.dat','V_LC3009_tel_ccd_atm_ext_1.2.dat','V_LC3014_tel_ccd_atm_ext_1.2.dat','V_LC9844_tel_ccd_atm_ext_1.2.dat','Y_SWO_TAM_scan_atm.dat','Y_DUP_TAM_scan_atm.dat','Jrc1_SWO_TAM_scan_atm.dat','Jrc2_SWO_TAM_scan_atm.dat','H_SWO_TAM_scan_atm.dat','H_DUP_TAM_scan_atm.dat'],
  ['G.dat','R.dat','I.dat','Z.dat','g.dat','r.dat','i.dat','z.dat'],
  ['g_filt_revised.txt','r_filt_tonry.txt','i_filt_tonry.txt','z_filt_tonry.txt'],
   ['effMEGACAM-g.dat','effMEGACAM-r.dat','effMEGACAM-i.dat','effMEGACAM-z.dat'],
  ['g_filt_revised.txt','r_filt_tonry.txt','i_filt_tonry.txt','z_filt_tonry.txt'],
  ['SNLS3_4shooter2_U.dat', 'SNLS3_4shooter2_B.dat', 'SNLS3_4shooter2_V.dat', 'SNLS3_4shooter2_R.dat', 'SNLS3_4shooter2_I.dat'], #CFA3S
    ['SNLS3_Keplercam_U_modtran.dat', 'SNLS3_Keplercam_B_modtran.dat', 'SNLS3_Keplercam_V_modtran.dat', 'SNLS3_Keplercam_r_modtran.dat', 'SNLS3_Keplercam_i_modtran.dat'] #CFA3K
]

obsfiltss = [
  ['g','r','i','z'],
  ['g','r','i','z'],
  ['g', 'r', 'i'],
  ['u','g','r','i','B','V','o','m','n','Y','y'],
  ['G','R','I','Z','g','r','i','z'],
  ['g','r','i','z'],
  ['g','r','i','z'],
  ['g','r','i','z'],
  ['B', 'V', 'R', 'I'],
  ['U','B','V','r','i'],

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
