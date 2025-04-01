![alt text](https://github.com/bap37/ZTFDovekie/blob/main/Dov_w_shoesjpg?raw=true)



Original code comes from https://arxiv.org/abs/2112.03864

This is the cleaned, upgraded version. Less WIP. 

--------------------------

dovekie.py collects the relevant data and calculates the colour/magnitude slope for each band. It will also run a NUTS mcmc process to determine the best offsets, if asked. 

`python dovekie.py` will _not_ run the mcmc. Instead, it will generate a set of colour/magnitude plots for each filter combination and apply the initial magnitude offsets calculated from the PS1 aperture magntiudes. When finished, it will generate a file called `preoffsetsaper.dat`, which contains a lot of very helpful diagnostic information, such as the number of stars in each survey, the synthetic colour/magnitude slopes, the data colour/magnitude slopes, and the significance of the diference between them. 

mcmc.py has some additional options:
DEBUG - a debug key that prints additional information if needs be.

IRSA - This will run an IRSA dust query on each of the surveys that you have specified in `surveys_for_chisq`. This should **only** be run after you have generated new observed data to use, at is a bit time consuming, and is disabled by default.

FAKES  - if you want to test the validity of the mcmc with fake stars, this will switch mcmc.py to look in an alternative directory, `output_fake_apermags`.

MCMC - runs the MCMC process, and outputs the chains every 100 steps. This takes time, so best to do in a screen or something. 

A lot of information that dovekie.py reads in is stored in DOVEKIE_DEFS.yaml. This file contains input information (which surveys to fit) and output information; e.g. where to store the mcmc chains. 

-------------------------------------

REGENERATE SYNTHETIC MAGNITUDES:
1. run `python loopsyntheticmags_commandlineaper.py` without any commands. This will give a list of available surveys to process, with an associated integer.
2. choose the survey integer and run `python loopsyntheticmags_commandlineaper.py INT` where INT is your desired number.
3. If you desire a shift be applied to the filters, add the `--SHIFT "np.arange(minval,maxval,binsize)"` key in the command line, which will apply shifts based on your input.



