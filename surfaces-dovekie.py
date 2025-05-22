import pandas as pd
import os, sys
from glob import glob
import numpy as np
import argparse
sys.path.insert(1, 'scripts/')
from helpers import *
import shutil

#/project2/rkessler/SURVEYS/PS1MD/USERS/dscolnic/PANTHEON+/kcor/fragilistic_cov_template_filler
#OG comes from around here

overridedict = {"CSP-g":   "CSP-g/A",
                "CSP-r":   "CSP-r/L",
                "CSP-i":   "CSP-i/C",
                "CSP-B":   "CSP-B/u",
                "CSP-o":   "CSP-o/v",
                "CSP-m":   "CSP-m/w",
                "CSP-n":   "CSP-n/x",
                "CFA3S-U": "CFA3S-U/a",
                "CFA3S-B": "CFA3S-B/b",
                "CFA3S-V": "CFA3S-V/c",
                "CFA3S-R": "CFA3S-R/d",
                "CFA3S-I": "CFA3S-I/e",
                "CFA3K-U": "CFA3K-U/f",
                "CFA3K-B": "CFA3K-B/h",
                "CFA3K-V": "CFA3K-V/j",
                "CFA3K-r": "CFA3K-r/k",
                "CFA3K-i": "CFA3K-i/l",
               "CFA4P1-B": "CFA41-B/D",
               "CFA4P1-V": "CFA41-V/E",
               "CFA4P1-r": "CFA41-r/F",
               "CFA4P1-i": "CFA41-i/G",
               "CFA4P2-B": "CFA42-B/P",
               "CFA4P2-V": "CFA42-V/Q",
               "CFA4P2-r": "CFA42-r/W",
               "CFA4P2-i": "CFA42-i/T",
}

NREAL = 10
jsonload = 'DOVEKIE_DEFS.yml' 
config = load_config(jsonload)

def get_args():
   parser = argparse.ArgumentParser()

   msg = "HELP menu for config options"

   msg = """Name of the post-Dovekie offsets file.  """
   parser.add_argument("--OFFSETS", help=msg, type=str)

   msg = """Name of the post-Dovekie Covariance file. """
   parser.add_argument("--COV", help=msg, type=str)

   msg = """Name of the output directory.  """
   parser.add_argument("--OUTDIR", help=msg, type=str)

   msg = "Temporary placeholder for further development. Default False."
   parser.add_argument("--FULL", help=msg, action="store_true")
   parser.set_defaults(FULL=False)
   
   args = parser.parse_args()
   return args

def prep_config(args):
   OFFSETS = args.OFFSETS
   COV = args.COV
   FULL = args.FULL
   OUTDIR = args.OUTDIR
  
   return OFFSETS, FULL, COV, OUTDIR

#These two appear to set kcor information. Commented out for the moment. 
###################
#for i,row in off.iterrows(): os.system("sed -i 's/+%s/%s/g' %s/*.input"%(row['SURVEYFILT'],row['OFFSETSTR'],version))
#for f in glob(version+'/*.input'):
#    os.system('cd %s; kcor.exe %s'%(version,f.split('/')[-1]))
#################

def create_kcor(OFF, OUTDIR):
   #replace the "&" values in these kcor with the appropriate thingo
   #example line
   #FILTER: SNLS-z effMEGACAM-z.dat 0+0.007&
   kcor_ogs = glob("templates/new_kcor_templates/*.input")
   for kcorog in kcor_ogs:
      print(f"starting {kcorog.split('/')[-1]}")
      tmptxtt = ""
      with open(kcorog, "r") as kcorfile:
         Lines = kcorfile.readlines()
         for line in Lines:
            if "&" not in line:
               tmptxtt += line
            else:
               objs = line.split()
               surveyfilt = objs[1]
               surveyfilt = surveyfilt.split("/")[0]
               #print(surveyfilt, "AAAAAA")
               if "PS1" in surveyfilt:
                  surveyfilt = surveyfilt.replace("PS1", "PS1SN")
               elif ("CFA4" in surveyfilt) & ("DES" not in kcorog):
                  surveyfilt = surveyfilt.replace("CFA4", "CFA4P")
               offset = OFF.loc[OFF.SURVEYFILT == surveyfilt].OFFSET.values
               print(surveyfilt, offset)
               if "D3YR" in line:
                  line = line.replace("D3YR", "DES")
               if len(offset) < 1 :
                  offset = ''
               else:
                  offset = offset[0]
                  if offset >= 0: offset = "+"+str(offset)
                  else: offset = str(offset)
               line = line.replace("&", offset)
               tmptxtt += line
      #OUTDIR should exist, so just write to OUTDIR/kcor
      with open(f'{OUTDIR}/kcor/{kcorog.split("/")[-1]}', "w") as writefile:
         writefile.write(tmptxtt)
   return print("Done")


def DOSALT(OUTDIR, OFF):
   os.system(f'cp -r templates/SALT_templates {OUTDIR}/SALT_{OUTDIR}')
   for i,row in OFF.iterrows(): os.system("sed -i 's/+%s/%s/g' %s/%s/MagSys/*.dat"%('EX_'+row['SURVEYFILT'],row['OFFSETSTR'], OUTDIR, 'SALT_'+OUTDIR))

   for f in glob("%s/%s/MagSys/*.dat"%(OUTDIR,'SALT_'+OUTDIR)):
      txt = open(f,'r').read()
      newtxt = ''
      print(f)
      if len(txt.split('$@'))>1:
         for snippit in txt.split('$@')[:-1]:
            if (('+' in snippit) | ('-' in snippit)) & (snippit[0] != '#') & (snippit[0] != 'S'):
               try:
                  snip = float(eval(snippit))
                  newtxt += '%.4f'%snip
               except:
                  newtxt += snippit
               else:
                  newtxt += snippit
      fout = open(f,'w')
      fout.write(newtxt)
      fout.close()
   return print("Done with whatever that was.")

def SYSTSURFACES(OUTDIR, covf):
   np.random.seed(42)

   os.system(f'cp -r templates/realisations_templates {OUTDIR}/realisations_{OUTDIR}')
   os.system(f'cp -r templates/fitopt_templates {OUTDIR}/fitopt_{OUTDIR}')

   """"
   txt = open(f'{OUTDIR}/fitopt_{OUTDIR}/ALL.fitopts','r').read()
   newtxt = ''
   for snippit in txt.split('&'):
      if snippit[:2] == 'R_':
         randf = float(snippit[2:])
         newtxt += '%.4f'%float(np.random.normal(loc=0.0,scale=randf,size=1))
      else:
         newtxt += snippit
   fout = open(f'{OUTDIR}/fitopt_{OUTDIR}/ALL.fitopts','w')
   fout.write(newtxt)
   fout.close()
   """

   #ok definitely fucked this one up 
   #os.system("sed -i 's/_OUTDIR/_%s/g' %s/ALL.fitopts"%(OUTDIR,'fitopt_'+OUTDIR))

   from scipy.linalg import eigh, cholesky
   from scipy.stats import norm

   #not sure we have labels anymore ? 
   cov = np.load(covf)
   x = norm.rvs(size=(len(cov['labels']), NREAL))
   c = cholesky(cov['cov'], lower=True)
   real = np.dot(c, x)
   
   for n in range(NREAL):
      params = real[:,n]
      labels = cov['labels']
      WRITE_ACTUAL(params, labels, OUTDIR, n, config)

   return "Done"

def WRITE_ACTUAL(params, labels, OUTDIR, n, config):
   waveshifts = config['waveshifts']
   labels = [sub[:-2] for sub in labels]

   with open(f'{OUTDIR}/realisations_{OUTDIR}/shift_{n}.dat', "w") as filew:
      for n in range(len(labels)):
         surv = labels[n].split("-")[0]
         if surv == "PS1":
            continue
         if surv == 'D3YR': continue
         if surv == "PS1SN": surv = "PS1MD"
         if "CFA4" in surv: surv = surv.replace("P", "p")
         survband = labels[n]
         if surv == "PS1MD":
            survband = survband.replace("SN", '')
#         if "CFA4" in surv: survband = survband.replace("P", "p")
         #doot overridedict
         try: 
            survband = overridedict[survband]
         except KeyError:
            pass
         if "CFA3" in surv: surv = "CFA3"
         if surv == 'ZTF': surv = "ZTF_MSIP"
#         if surv == "D3YR": surv = "DES"
         if surv == "Foundation":
            surv = "FOUNDATION" ; survband = survband.replace("Foundation", "PS1")
         buildstr = f'MAGSHIFT {surv} {survband.replace("D3YR", "DES")} {np.around(params[n], 3)}' #hacky ugly
         filew.write(buildstr+'\n')
      filew.write("\n")
      for n in range(len(labels)):
         #set up value of waveshift here
         surv = labels[n].split("-")[0]
         if surv == "PS1":
            continue
         survband = labels[n].split("-")[-1]
         survbandwrite = labels[n]
         if surv == "D3YR": continue 
         waveval = waveshifts[surv][survband]
         if surv == "PS1SN": surv = "PS1MD"
         if "CFA4" in surv: surv.replace("P", "p")
         #doot overridedict
         try: 
            survbandwrite = overridedict[survbandwrite]
         except KeyError:
            pass
         if "CFA3" in surv: surv = "CFA3"
         if "CFA4" in surv: surv = surv.replace("P", "p")
         if surv == "PS1MD":
            survbandwrite = survbandwrite.replace("SN", '')
         if surv == 'ZTF': surv = "ZTF_MSIP"
         if surv == "Foundation":
            surv = "FOUNDATION" ; survbandwrite = survbandwrite.replace("Foundation", "PS1")
         buildstr = f'WAVESHIFT {surv} {survbandwrite} {np.around(np.random.normal(0,waveval),3)}'
         filew.write(buildstr+'\n')
   return print(f"Done writing this iteration of SALTShaker Training files at {OUTDIR}")
        
if __name__ == "__main__":
   
   args = get_args()
   OFFSETS, FULL, COV, OUTDIR = prep_config(args)
   if not (os.path.exists(OFFSETS) | os.path.exists(COV)):
      print(f'You gave {OFFSETS} and {COV}, but this file does not exist. Quitting.')
      quit()
   elif not OUTDIR:
      print("Please enter a value for OUTDIR!")
      quit()
   OFF=pd.read_csv(OFFSETS,delim_whitespace=True)
   #OFF['OFFSET'] = OFF['OFFSET'] + OFF['EXTRA'] #For extra offsets ? Unclear where/why
   OFF['OFFSETSTR'] = OFF['OFFSET'].astype(str)

   OFF['OFFSETSTR'][OFF['OFFSET']>=0] = '+'+OFF['OFFSETSTR']
   try:
      os.mkdir(OUTDIR)
   except FileExistsError:
      print(OUTDIR,'directory already exists, removing it')
      shutil.rmtree(OUTDIR)
      os.mkdir(OUTDIR)
   os.mkdir(f"{OUTDIR}/kcor")
   SYSTSURFACES(OUTDIR, COV)
   create_kcor(OFF, OUTDIR)
