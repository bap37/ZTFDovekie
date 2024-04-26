Original code comes from https://arxiv.org/abs/2112.03864

This is the cleaned, somewhat upgraded version. Still very WIP.

--------------------------

mcmc.py - The Magic:
The beating heart of the project is mcmc.py, which both runs the mcmc to determine offsets _and_ calculates the colour/magnitude slope. It works off of several command line inputs, which are listed below:

without any command line keys:

`python mcmc.py` will _not_ run the mcmc. Instead, it will generate a set of colour/magnitude plots for each filter combination and apply the initial magnitude offsets calculated from the PS1 aperture magntiudes. When finished, it will generate a file called `preoffsetsaper.dat`, which contains a lot of very helpful diagnostic information, such as the number of stars in each survey, the synthetic colour/magnitude slopes, the data colour/magnitude slopes, and the significance of the diference between them. 

mcmc.py has some additional options:
DEBUG - a debug key that prints additional information if needs be.

REDO - This will run an IRSA dust query on each of the surveys that you have specified in `surveys_for_chisq`. This should **only** be run after you have generated new observed data to use, at is a bit time consuming, and is disabled by default.

FAKES - CURRENTLY NON-FUNCTIONAL - if you want to test the validity of the mcmc with fake stars, this will switch mcmc.py to look in an alternative directory, `output_fake_apermags`.

MCMC - runs the MCMC process, and outputs the chains every 100 steps. This takes time, so best to do in a screen or something. 


Unfortunately, changing the surveys that will be fit requires tinkering with the code directly at the moment. This will be changed in an upcoming release to use an input file. 

-------------------------------------

REGENERATE SYNTHETIC MAGNITUDES:
1. run `python loopsyntheticmags_commandlineaper.py` without any commands. This will give a list of available surveys to process, with an associated integer.
2. choose the survey integer and run `python loopsyntheticmags_commandlineaper.py INT` where INT is your desired number.
3. If you desire a shift be applied to the filters, add the `--SHIFT "np.arange(minval,maxval,binsize)"` key in the command line, which will apply shifts based on your input.

Some general notes: 
synth_PS1_shift... and synth_PS1SN_shift... are identical, and synth_PS1_shift **should not be touched**. 


