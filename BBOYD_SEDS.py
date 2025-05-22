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

def clean_surveys(surv):
  if surv == "Foundation": 
    surv = "PS1SN"
  elif "ASASSN" in surv:
    surv = "ASASSN"
  elif "KAIT" in surv:
    surv = "KAIT_2018"
  elif "SWIFT" in surv:
    surv = "SWIFTnat"
  return surv

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

  allfiles = {}

  # Loop through each file, split up to avoid hitting filesize limits on github
  for file_path in glob('spectra/bboyd/wd_seds_new_subset*.npz'):
        with np.load(file_path) as data:
           for key in data:
              assert(key not in allfiles)
              allfiles[key]=data[key]

  for ind, surv,kcorpath,kcor,shiftfilts,obsfilts in zip(
        range(len(config['survs'])),config['survs'],config['kcorpaths'],config['kcors'],config['shiftfiltss'],config['obsfiltss']):
    print(ind,surv)
    if ind != float(index): continue
      
    if kcorpath[-1] == '/': kcorpath=kcorpath[:-1]
    
    #for shift in np.arange(-30,40,10):
    for shift in shifts:
      version = surv
      print(f'starting shift = {shift}')

      #Here we open the bboyd seds

      bd=open('output_synthetic_magsaper/bboyd_synth_%s_shift_%.3f.txt'%(surv,shift),'w')
      bd.write(' '.join(['survey','version','standard','shift',' ']))
      for fff in obsfilts:
        bd.write(version+'-'+fff+' ')
      bd.write('\n')

      for key in allfiles:
        if key == 'wave':
          continue
        ngslf = key
        for entry in range(98):
          filename1='%s/fillme_%s.dat'%(kcorpath,surv)
          x=open(filename1,'w')
          f = interpolate.interp1d(allfiles['wave'], allfiles[key][entry,:])
          w = [0]
          f = [0.0]
          w.extend(allfiles['wave'])
          f.extend(allfiles[key][entry,:])
          xr = np.arange(2000,12000,2)
          if max(allfiles['wave']) < 12000:
            ww9 = (allfiles['wave'] <9500) & (allfiles['wave']>9000)
            ww95 =(allfiles['wave']<10000) & (allfiles['wave']>9500)
            slope = (np.mean(allfiles[key][entry,:][ww95]) - np.mean(allfiles[key][entry,:][ww9]))/1000
            lastw = allfiles['wave'][-1]
            lastf = allfiles[key][entry,:][-1]
            w.append(12500)
            f.append(lastf+slope*(12500-lastw))
          interp = interpolate.interp1d(w,f)
        
          for co in xr[:-10]:
            x.write(str(co)+' '+str(interp(co))+'\n')
          x.close()

          filename2='textfiles/%s.txt'%ngslf
          x=open(filename2,'w')
          for co in range(0,len(allfiles['wave'])):
            x.write(str(allfiles['wave'][co])+' '+str(allfiles[key][entry,:][co])+'\n')
          x.close()

          filtshiftstring = ' FILTER_LAMSHIFT '
          for filt in shiftfilts:
            filtshiftstring += ' '+filt+' '+str(shift)+' '
          try:
            print('kcor.exe %s/%s %s OUTFILE %s/.in_%s.fits%s BD17_SED %s/fillme_%s.dat > logs/kcor_%s.log'%(kcorpath,kcor,filtshiftstring,kcorpath,surv,parallel,kcorpath,surv,surv))
            os.system('kcor.exe %s/%s %s OUTFILE %s/.in_%s.fits%s BD17_SED %s/fillme_%s.dat > logs/kcor_%s.log'%(kcorpath,kcor,filtshiftstring,kcorpath,surv,parallel,kcorpath,surv,surv))
          except:
            print('this ^^^ spectrum failed kcor')
            continue
        
          g=open('logs/kcor_%s.log'%(surv),'r').readlines()
          vals=['99' for n in range(len(obsfilts))]
          searchsurv = surv
          print(surv)
          for gg in g:
            for iii, obsf in enumerate(obsfilts):
              searchsurv = clean_surveys(surv)
              if (f'{searchsurv}-{obsf}' in gg):
                if ('BD17' in gg.split()[2]):
                  try:
                    vals[iii]=gg.split()[6]
                  except:
                    vals[0] == 99
          print(vals)
          vals = np.array(vals).astype(float); vals = -1*vals ;
          if 'ASASSN' in version: version = "ASASSN";
          if vals[0]!='99':
            bd.write(' '.join([surv,version,ngslf,str(round(shift,4)),'']))
            bd.write(' '.join(vals.astype(str))+'\n')
          os.remove(filename1)
          os.remove(filename2)
  bd.close()
  print("The magnitudes you saw may appear to have been negative. They should have been written out as positive values.")
    
