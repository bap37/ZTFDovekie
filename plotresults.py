import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import pandas as pd
from chainconsumer import ChainConsumer
import dovekie as ll
import sys, argparse, os
from functools import partial
import corner
sys.path.insert(1, 'scripts/')
import helpers
from glob import glob


def get_args():
   parser = argparse.ArgumentParser()

   msg = "HELP menu for config options"

   msg = """Name of file where the chains are stored. 
          \nTypically a string of the surveys and a date.   
   """
   parser.add_argument("--FILENAME", help=msg, type=str)

   msg = "Whether or not do do the full suite of diagnostics. Default False."
   parser.add_argument("--FULL", help=msg, action="store_true")
   parser.set_defaults(FULL=False)
   
   args = parser.parse_args()
   return args

def prep_config(args):
   filename = args.FILENAME
   FULL = args.FULL
   return filename, FULL

global surveys_for_chisq, fixsurveynames, surveydata, obsdfs, DEBUG
fixsurveynames = [] ; DEBUG = False

if __name__ == "__main__":
   
   args = get_args()
   filename, FULL = prep_config(args)
   if not os.path.exists(filename):
      print(f'You gave {filename}, but this file does not exist. Quitting.')
      quit()

   version = (filename.split('.')[1])

   #initialise
   labels = helpers.create_labels(filename)
   samples = np.load(filename,allow_pickle=True)['samples']
   flat_samples = samples.reshape(-1, samples.shape[-1])
   surveys_for_chisq = np.load(filename,allow_pickle=True)['surveys_for_chisq']
   #calls mcmc.py to do some quick loading of data

   #starts the plotting 
   helpers.create_chains(labels, samples) #10 dimensions by default
   c = helpers.create_cov(labels, flat_samples, version)
   postoffsets = helpers.create_postoffsets_summary(c)
   helpers.create_latex("postoffsets.dat", "postoffsets-latex.tex")

   surveydata = ll.get_all_shifts(surveys_for_chisq, reference_surveys=['PS1'])
   obsdfs = ll.get_all_obsdfs(surveys_for_chisq, False, False)

   tablefile = "postproces.dat"
   tableout = open(tablefile,'w')
   tableout.write('COLORSURV COLORFILT1 COLORFILT2 OFFSETFILT1 OFFSETSURV OFFSETFILT2 SPECLIB OFFSET NDATA D_SLOPE S_SLOPE SIGMA SHIFT\n')

   #full_likelihood_data= partial(ll.full_likelihood,surveys_for_chisq, fixsurveynames,surveydata,obsdfs, whitedwarf_seds=None,whitedwarf_obs=None)
   #full_likelihood_data(params=postoffsets,subscript='postprocess',doplot=True, tableout=tableout, reference_surveys=['PS1'])
   #print("Forced reference survey to PS1")
   tableout.close()

   print("Reached end of testing")

   if FULL:
      print("bloop")
      helpers.create_corr(c, labels)
      helpers.create_corner(labels, flat_samples)
#Below are the shadow-touched lands. Do not go there. Only madness lies.
