import astropy
from astropy.io import fits
import numpy as np
from glob import glob
import os, sys
from scipy import interpolate
sys.path.insert(1, 'scripts/')
from helpers import load_config
import argparse

jsonload = "DOVEKIE_DEFS.yml"
config = load_config(jsonload)

def get_args():
  parser = argparse.ArgumentParser()

  msg = "HELP menu for config options"

  msg = "Default -9, which will only output the list of available surveys. Integer number corresponding to the survey in the code printout."
  parser.add_argument("--SURVEY", help=msg, type=int, default=-9)

  msg = 'Default -9. If unspecified, will not shift any filters. If specified, please use something corresponding to "np.arange(minval, maxval, binsize)" where you fill out the argument appropriately. \nIn command line, proper quotations around the np.arange are very important!'
  parser.add_argument("--SHIFT", help=msg, type=str, default="-9")
  args = parser.parse_args()
  return args


parallel = '1'
if __name__ == '__main__':

  print("WARNING!")
  print("The magnitudes you will see may appear to be negative. They will be written out as positive values.")

  args = get_args()
  if args.SURVEY == -9:
    for ind, surv,kcorpath,kcor,shiftfilts,obsfilts in zip(
        range(len(config['survs'])),config['survs'],config['kcorpaths'],config['kcors'],config['shiftfiltss'],config['obsfiltss']):
      print(ind,surv)
    print('please call with integer arg like so:\npython loopsyntheticmags.py X')
    sys.exit()
  else:
    index = args.SURVEY

  if args.SHIFT != "-9":
    shifts = eval(args.SHIFT)
  else:
    shifts = [0]

  for ind, surv,kcorpath,kcor,shiftfilts,obsfilts in zip(
        range(len(config['survs'])),config['survs'],config['kcorpaths'],config['kcors'],config['shiftfiltss'],config['obsfiltss']):
    print(ind,surv)
    if ind != float(index): continue
      
    if kcorpath[-1] == '/': kcorpath=kcorpath[:-1]
    
    #for shift in np.arange(-30,40,10):
    for shift in shifts:
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
        searchsurv = surv
        for gg in g:
          for iii, obsf in enumerate(obsfilts):
            if surv == "Foundation": searchsurv = "PS1SN"
            elif "ZTF" in surv: searchsurv = "ZTF"
            if (f'{searchsurv}-{obsf}' in gg):
              if ('BD17' in gg.split()[2]):
                try:
                  vals[iii]=gg.split()[6]
                except:
                  vals[0] == 99
        print(vals, cat)
        vals = np.array(vals).astype(float); vals = -1*vals ;
        if vals[0]!='99':
          bd.write(' '.join([surv,version,ngslf,cat,str(round(shift,3)),'']))
          bd.write(' '.join(vals.astype(str))+'\n')
      bd.close()
  print("The magnitudes you saw may appear to have been negative. They should have been written out as positive values.")
    
