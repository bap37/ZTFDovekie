import pandas as pd
import os
from glob import glob
import numpy as np
import argparse

#/project2/rkessler/SURVEYS/PS1MD/USERS/dscolnic/PANTHEON+/kcor/fragilistic_cov_template_filler
#OG comes from around here

global LANDOLT_OFFSIG
LANDOLT_OFFSIG = 0.02
NREAL = 9

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
   filename = args.OFFSETS
   covfile = args.COV
   FULL = args.FULL
   OUTDIR = args.OUTDIR
  
   return OFFSETS, FULL, COV, OUTDIR

#These two appear to set kcor information. Commented out for the moment. 
###################
#for i,row in off.iterrows(): os.system("sed -i 's/+%s/%s/g' %s/*.input"%(row['SURVEYFILT'],row['OFFSETSTR'],version))
#for f in glob(version+'/*.input'):
#    os.system('cd %s; kcor.exe %s'%(version,f.split('/')[-1]))
#################

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

   #ok definitely fucked this one up 
   os.system("sed -i 's/_OUTDIR/_%s/g' %s/ALL.fitopts"%(OUTDIR,'fitopt_'+OUTDIR))

   from scipy.linalg import eigh, cholesky
   from scipy.stats import norm

   #not sure we have labels anymore ? 
   cov = np.load(covf)
   x = norm.rvs(size=(len(cov['labels']), NREAL))
   c = cholesky(cov['cov'], lower=True)
   real = np.dot(c, x)
   
   #first do one with zero offsets
   if FIRST:
      WRITE_SALTSHAKER(params, labels, OUTDIR, o)
   for n in range(NREAL):
      params = real[:,n]
      labels = cov['labels']
      WRITE_SALTSHAKER(params, labels, OUTDIR, n)

   return print("Done, ish?")

def WRITE_SALTSHAKER(params, labels, OUTDIR, n):
   os.system(f"cp {OUTDIR}/realisations_{OUTDIR}/shift_template.dat realisations_{OUTDIR}/shift_{n}.dat")
   txt = open(f'{OUTDIR}/realisations_{OUTDIR}/shift_{n}.dat').read()
   newtxt = ''
   for snippit in txt.split('&'):
      if snippit[0] == 'R':
         randf = float(snippit[1:])
         newtxt += '%.4f'%float(0)
      else:
         newtxt += snippit
   fout = open(f'{OUTDIR}/realisations_{OUTDIR}/shift_{n}.dat', 'w')
   fout.write(newtxt)
   fout.close()

   for p,l in zip(params,labels): 
      print("sed -i 's/%s/%.4f/g' %s/%s/shift_%d.dat"%('EX_'+'_'.join(l.split(' ')[:-1]),p,OUTDIR,'realisations_'+OUTDIR,n))
      os.system("sed -i 's/%s/%.4f/g' %s/%s/shift_%d.dat"%('EX_'+'_'.join(l.split(' ')[:-1]),p,OUTDIR,'realisations_'+OUTDIR,n))   

   for p,l in zip([np.random.normal(loc=0,scale=LANDOLT_OFFSIG),
                        np.random.normal(loc=0,scale=LANDOLT_OFFSIG),
                        np.random.normal(loc=0,scale=LANDOLT_OFFSIG),
                        np.random.normal(loc=0,scale=LANDOLT_OFFSIG),
                        np.random.normal(loc=0,scale=LANDOLT_OFFSIG)],
                       ['LANDOLT_U','LANDOLT_B','LANDOLT_V','LANDOLT_R','LANDOLT_I']):
      print("sed -i 's/%s/%.4f/g' %s/%s/shift_%d.dat"%(l,p,OUTDIR,'realisations_'+OUTDIR,n))
      os.system("sed -i 's/%s/%.4f/g' %s/%s/shift_%d.dat"%(l,p,OUTDIR,'realisations_'+OUTDIR,n))

   txt = open(f'{OUTDIR}/realisations_{OUTDIR}/shift_{n}.dat', 'r').read()
   newtxt = ''

   for snippit in txt.split('@'):
      if '+' in snippit:
         snip = float(eval(snippit.split(')')[0]))
         newtxt += '%.4f'%snip
      else:
         newtxt += snippit
   fout = open(f'{OUTDIR}/realisations_{OUTDIR}/shift_{n}.dat', 'w')
   fout.write(newtxt)
   fout.close()
   return print("Done writing... Something.")
        
if __name__ == "__main__":
   
   args = get_args()
   OFFSETS, FULL, COV, OUTDIR = prep_config(args)
   if not (os.path.exists(OFFSETS) | os.path.exists(COV)):
      print(f'You gave {OFFSETS} and {COV}, but this file does not exist. Quitting.')
      quit()
   OFF=pd.read_csv(OFFSETS,delim_whitespace=True)
   OFF['OFFSET'] = OFF['OFFSET'] + OFF['EXTRA']
   OFF['OFFSETSTR'] = OFF['OFFSET'].astype(str)

   OFF['OFFSETSTR'][OFF['OFFSET']>=0] = '+'+OFF['OFFSETSTR']
   try:
      os.mkdir(OUTDIR)
   except:
      print(OUTDIR,'directory already exists')

   #Perhaps key in a small program to zero-out certain filters
