import astropy
from astropy.io import fits
import os
import numpy as np
from glob import glob
import os
import sys
from scipy import interpolate

#Adding these fuckers one at a time
survs = [
  'PS1SN',
  'DES',
  'ZTF',
  'CSP',
  'SDSS',
  'Foundation',
  'SNLS',
  'CFA3S',
  'CFA3K',
]

kcorpaths = [
  'kcor/PS1s_RS14_PS1_tonry/',
  'kcor/DES/',
  'kcor/ZTF/',
  'kcor/CSP_TAMU/',
  'kcor/SDSS/',
  'kcor/PS1s_RS14_PS1_tonry/',
  'kcor/SNLS/',
  'kcor/CFA3_4shooter_native/',
  'kcor/CFA3_KEPLERCAM/',
]

kcors = [
  'PS1_excalinaper_+30A.input',
  'DECam_excalin.input',
  'ZTF_excalin.input',
  'CSPDR3_excalin.input',
  'SDSS_kcor.input',
  'PS1_excalinaper_+30A.input',
  'SNLS_excalin.input',
  'CFA3_4shooter_excalin.input',
  'CFA3_KEPLERCAM_excalin.input',

]


filtpaths = [
  'filters/PS1s_RS14_PS1_tonry/',
  'filters/DES-SN3YR_DECam/',
  'filters/ZTF/',
  'filters/CSP_TAMU/',
  'filters/SDSS_Doi2010_CCDAVG/',
  'filters/PS1s_RS14_PS1_tonry/',
  'filters/SNLS3-Megacam/',
  'filters/CFA3_native/',
  'filters/CFA3_native/',
]


shiftfiltss = [
  ['g','r','i','z'],
  ['g','r','i','z'],
  ['g', 'r', 'i'],
  ['u','g','r','i','B','V','o','m','n','Y','y'],
  ['G','R','I','Z','g','r','i','z'],
  ['g', 'r', 'i', 'z'],
  ['g', 'r', 'i', 'z'],
  ['B', 'V', 'R', 'I'],
  ['U','B','V','r','i'],

]


namedfiltss = [
  ['g','r','i','z'],
  ['g','r','i','z'],
  ['g', 'r', 'i'],
  ['u','g','r','i','B','V','o','m','n','Y','y'],
  ['G','R','I','Z','g','r','i','z'],
  ['g', 'r', 'i', 'z'],  
  ['B', 'V', 'R', 'I'],
  ['U','B','V','r','i'],

]


obsfiltss = [
  ['g','r','i','z'],
  ['g','r','i','z'],
  ['g', 'r', 'i'],
  ['u','g','r','i','B','V','o','m','n','Y','y'],
  ['G','R','I','Z','g','r','i','z'],
  ['g','r','i','z'],
  ['g','r','i','z'],
  ['B', 'V', 'R', 'I'],
  ['U','B','V','r','i'],

]




parallel = '1'
if __name__ == '__main__':


  print(range(len(survs)))
  print(len(survs),len(kcorpaths),len(kcors),len(shiftfiltss),len(obsfiltss))

  try:
    index = sys.argv[1]
    #parallel = str(sys.argv[2])
  except:
    for ind, surv,kcorpath,kcor,shiftfilts,obsfilts in zip(
        range(len(survs)),survs,kcorpaths,kcors,shiftfiltss,obsfiltss):
      print(ind,surv)
    print('please call with parallelization arg like so:\npython loopsyntheticmags.py 2 1')
    sys.exit()

  for ind, surv,kcorpath,kcor,shiftfilts,obsfilts in zip(
        range(len(survs)),survs,kcorpaths,kcors,shiftfiltss,obsfiltss):
    print(ind,surv)
    if ind != float(index): continue
      
    if kcorpath[-1] == '/': kcorpath=kcorpath[:-1]
    
    #for shift in np.arange(-30,40,10):
    for shift in [0]:
      #version = kcorpath.split('/')[1] #This is where the fuckery gets written out
      version = surv
      print(f'starting shift = {shift}')

      ngsl_files = glob('spectra/stis_ngsl_v2/*.fits')#[:5]
      dillon_calspec_files = glob('spectra/calspec23/*.fits')
      allfiles = ngsl_files
      allfiles.extend(dillon_calspec_files)
      speccats = [fn.split('/')[1] for fn in allfiles]

      bd=open('output_synthetic_magsaper/synth_%s_shift_%.3f.txt'%(surv,shift),'w')
      bd.write(' '.join(['survey','version','standard','standard_catagory','shift','']))
      for fff in obsfilts:
        bd.write(version+'-'+fff+' ')
      bd.write('\n')

      for ngslf,cat in zip(allfiles,speccats):
        hdul=fits.open(ngslf)
        print(ngslf)

        x=open('%s/fillme_%s.dat'%(kcorpath,surv),'w')
        f = interpolate.interp1d(hdul[1].data.WAVELENGTH,hdul[1].data.FLUX)
        w = [0]
        f = [0.0]
        w.extend(hdul[1].data.WAVELENGTH)
        f.extend(hdul[1].data.FLUX)
        xr = np.arange(2000,12000,2)
        if max(hdul[1].data.WAVELENGTH) < 12000:
          ww9 = (hdul[1].data.WAVELENGTH <9500) & (hdul[1].data.WAVELENGTH>9000)
          ww95 =(hdul[1].data.WAVELENGTH<10000) & (hdul[1].data.WAVELENGTH>9500)
          slope = (np.mean(hdul[1].data.FLUX[ww95]) - np.mean(hdul[1].data.FLUX[ww9]))/1000
          lastw = hdul[1].data.WAVELENGTH[-1]
          lastf = hdul[1].data.FLUX[-1]
          w.append(12500)
          f.append(lastf+slope*(12500-lastw))
        interp = interpolate.interp1d(w,f)
        
        for co in xr[:-10]:
          x.write(str(co)+' '+str(interp(co))+'\n')
        x.close()

        x=open('textfiles/%s.txt'%(ngslf.split('/')[-1]),'w')
        for co in range(0,len(hdul[1].data.WAVELENGTH)):
          x.write(str(hdul[1].data.WAVELENGTH[co])+' '+str(hdul[1].data.FLUX[co])+'\n')
        x.close()


        filtshiftstring = ' FILTER_LAMSHIFT '
        for filt in shiftfilts:
          #if (filt == 'E') & (surv == 'CFA1') : 
          #  filtshiftstring += ' E '+str(200.)+' '
          #else:
          filtshiftstring += ' '+filt+' '+str(shift)+' '
        try:
          print('kcor.exe %s/%s %s OUTFILE %s/.in_%s.fits%s BD17_SED %s/fillme_%s.dat > logs/kcor_%s.log'%(kcorpath,kcor,filtshiftstring,kcorpath,surv,parallel,kcorpath,surv,surv))
          os.system('kcor.exe %s/%s %s OUTFILE %s/.in_%s.fits%s BD17_SED %s/fillme_%s.dat > logs/kcor_%s.log'%(kcorpath,kcor,filtshiftstring,kcorpath,surv,parallel,kcorpath,surv,surv))
          #os.system('kcor.exe %s/%s > logs/kcor_%s.log'%(kcorpath,kcor,surv))
        except:
          print('this ^^^ spectrum failed kcor')
          continue
        
        g=open('logs/kcor_%s.log'%(surv),'r').readlines()
        vals=['99' for n in range(len(obsfilts))]
        for gg in g:
          for iii, obsf in enumerate(obsfilts):
            if (f'{surv}-{obsf}' in gg):
              if ('BD17' in gg.split()[2]):
                try:
                  vals[iii]=gg.split()[6]
                except:
                  vals[0] == 99
        print(vals, cat)
        if vals[0]!='99':
          bd.write(' '.join([surv,version,ngslf,cat,str(round(shift,3)),'']))
          bd.write(' '.join(vals)+'\n')
      bd.close()
      
    
