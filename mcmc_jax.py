import pandas as pd
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import os, sys
from scipy.optimize import curve_fit
from glob import glob
sys.path.insert(1, 'scripts/')
from helpers import *
import time, pickle
import emcee
from astroquery.irsa_dust import IrsaDust
import astropy.coordinates as coord
import astropy.units as u
from scipy import interpolate
import argparse
from functools import partial
import jax
from jax import numpy as jnp
from jax import random as random
from jaxnuts.sampler import NUTS
from collections import namedtuple
from os import path

#search for "TODO"
isshift = False
global DEBUG
DEBUG = False

jsonload = 'DOVEKIE_DEFS.yml' #where all the important but unwieldy dictionaries live
config = load_config(jsonload)
survmap, survmap4shift, survfiltmap, obssurvmap, revobssurvmap, revobssurvmapforsnana, survcolormin, survcolormax, synth_gi_range, obsfilts, snanafilts, snanafiltsr, relativeweights, errfloors ,target_acceptance , n_burnin= prep_config(config)


obscolors_by_survey = {'PS1':['PS1-g','PS1-i']} #dodgy, feel like this should be tonry

filter_means = pd.read_csv('filter_means.csv') 

filter_means = filter_means.set_index(['SURVEYFILTER']).to_dict()['MEANLAMBDA ']


chi2result=namedtuple('chi2result',['chi2','datax','datay','synthxs','synthys',
     'data_popt','data_pcov','cats','synthpopts',
     'synthpcovs','modelcolors','modelress','sigmasynth','sigmadata',
     'obslongfilt1','obslongfilt2','obslongfilta','obslongfiltb',
     'surv1','colorfilta','colorfiltb','yfilt1','surv2','yfilt2','shift'])

def get_all_shifts(surveys): #acquires all the surveys and collates them. 
    surveydfs = {}
    for survey in surveys:
        files = glob('output_synthetic_magsaper/synth_%s_shift_*.000.txt'%survmap4shift[survey]) #TODO - better determination of whether or not there are lambda shifts and what to do if there are
        print(files)
        if len(files) > 1:
            print("Picking up shifts!")
            isshift = True
        dfl = []
        for f in files:
            try:
                tdf = pd.read_csv(f,sep=" ") #formerly delim_whitespace      
                for x in list(tdf): #Converts the mags into the weird negative space that the code expects.
                    if "-" in x: tdf[x] = -1*tdf[x] ;
                if 'PS1_' in f:
                    tdf = tdf[-1*tdf['PS1-g']+tdf['PS1-i']>.25]
                    tdf = tdf[-1*tdf['PS1-g']+tdf['PS1-i']<1.6]
                dfl.append(tdf)
            except:
                print('WARNING: Could not read in ',f) 
        df = pd.concat(dfl, axis=0, ignore_index=True) ; df = df.sort_values(by=['standard','shift'])

        if len(df) < 2:
            print("You have an empty dataframe!")
            quit()

        surveydfs[survey] = df
    #First for loop ends here.
    for survey in surveys:
        if survey != 'PS1':
            surveydfs[survey] = pd.merge(surveydfs[survey],surveydfs['PS1'],left_on='standard',right_on='standard',suffixes=('','_b'))
    return surveydfs
    
def get_all_obsdfs(surveys, redo=False, fakes=False): 
    surveydfs = {}
    surveydfs_wext = {}
    for survey in surveys:
        survname = obssurvmap[survey]
        if survey == 'PS1': continue
        realdirname = 'output_observed_apermags'
        if fakes: realdirname = realdirname.replace("observed", "fake")
        if redo:
            print(f"Starting IRSA query for {survey}. If nothing is printing that's probably fine.")
            obsdf = pd.read_csv(f'{realdirname}/{survname}_observed.csv')
            surveydfs[survey] = obsdf
            surveydfs_wext[survey] = get_extinction(surveydfs[survey])
            obsdf = surveydfs_wext[survey]
            print(f"Finished performing IRSA query for {survey}")
            obsdf = obsdf[(obsdf['PS1-g']-obsdf['PS1-g_AV']-obsdf['PS1-i']+obsdf['PS1-i_AV'])<1.]
            obsdf = obsdf[(obsdf['PS1-g']-obsdf['PS1-g_AV']-obsdf['PS1-i']+obsdf['PS1-i_AV'])>.25]
            obsdf = obsdf[obsdf['PS1-g']-obsdf['PS1-g_AV']>14.3] #from dan via eddy schlafly
            obsdf = obsdf[obsdf['PS1-r']-obsdf['PS1-r_AV']>14.4]
            obsdf = obsdf[obsdf['PS1-i']-obsdf['PS1-i_AV']>14.6]
            obsdf = obsdf[obsdf['PS1-z']-obsdf['PS1-z_AV']>14.1]
            obsdf = obsdf[(obsdf['PS1-g']-obsdf['PS1-g_AV'])-
                          (obsdf['PS1-i']-obsdf['PS1-i_AV']) < survcolormax[survey]]
            obsdf = obsdf[(obsdf['PS1-g']-obsdf['PS1-g_AV'])-
                          (obsdf['PS1-i']-obsdf['PS1-i_AV']) > survcolormin[survey]]
            surveydfs_wext[survey] = obsdf
            obsdf.to_csv(f'{realdirname}+AV/{survname}_observed.csv', header=True, index=False, float_format='%g')
        else:
            try:
                obsdf = pd.read_csv(f'{realdirname}+AV/{survname}_observed.csv')
            except FileNotFoundError:
                print(f'For whatever reason, {realdirname}+AV/{survname} does not exist.')
                quit()

            surveydfs[survey] = obsdf
            if "PS1-g_AV" not in list(obsdf):
                print(f"output_observed_apermags+AV/{survname}_observed.csv is missing the required IRSA dust maps. Rerun the command with additional argument --IRSA \n quitting.")
                quit()
                #copy of quality cuts used to live here.
            surveydfs_wext[survey] = obsdf

    return surveydfs_wext


def getchi_forone(pars,surveydata,obsdfs,colorsurvab,surv1,surv2,colorfilta,colorfiltb,yfilt1,yfilt2,
                  shifta=0,shiftb=0,shift1=0,shift2=0,off1=0,off2=0,offa=0,offb=0,
                  calspecslope=0,calspecmeanlambda=4383.15,ngslslope=0,ngslmeanlambda=5507.09,
                  doplot=False,subscript='', outputdir='synthetic'): #where the magic happens I suppose
    ssurv2 = survmap[surv2]
    df2 = surveydata[surv2]
    shift=df2['shift'][0]

    chi2 = 0

    #changed these back to dashes
    longfilta = survfiltmap[colorsurvab]+'-'+colorfilta ; longfiltb = survfiltmap[colorsurvab]+'-'+colorfiltb
    longfilt1 = survfiltmap[surv1]+'-'+yfilt1 ; longfilt2 = survfiltmap[surv2]+'-'+yfilt2
    obslongfilta = obssurvmap[colorsurvab]+'-'+colorfilta ; obslongfiltb = obssurvmap[colorsurvab]+'-'+colorfiltb
    obslongfilt1 = obssurvmap[surv1]+'-'+yfilt1 
    if ('CSP' in surv2.upper()):
        obslongfilt2 = obssurvmap[surv2]+'-'+yfilt2.replace('o','V').replace('m','V').replace('n','V')
    else:
        obslongfilt2 = obssurvmap[surv2]+'-'+yfilt2 #
    #JAXX stuff starts here 
    obsdf = obsdfs[surv2] #grabs the observed points from the relevant survey
    if DEBUG: print(obsdf.columns, surv2)
    yr=obsdf[obslongfilt1]-obsdf[obslongfilt2] #observed filter1 - observed filter 2 
    datacut = jnp.abs(yr)<1 #only uses things lower than 1
    datacolor = (obsdf[obslongfilta] -obsdf[obslongfilta+"_AV"])-(obsdf[obslongfiltb]-obsdf[obslongfiltb+"_AV"])
    datares = obsdf[obslongfilt1]-obsdf[obslongfilt1+'_AV']-(obsdf[obslongfilt2]-obsdf[obslongfilt2+'_AV'])
#     import pdb;pdb.set_trace()
    datacolor=datacolor[datacut]
    datares=datares[datacut]
    datax,datay,sigmadata,yresd,data_popt,data_pcov = itersigmacut_linefit(datacolor,
                                                         datares,# np.ones(datacut.sum(),dtype=bool),
                                                         niter=2,nsigma=3)

    synthxs,synthys, cats,synthpopts,synthpcovs,modelcolors,modelress =  ([] for i in range(7))

    
    for i,cat in enumerate((df2['standard_catagory'],~df2['standard_catagory'])):
        catname=['calspec23','stis_ngsl_v2'][i]
        if DEBUG: print(df2.columns, surv2, surv1) 
        synthcut = (cat) & \
           (~jnp.isnan(df2[longfilt2].astype('float'))) & \
           (~jnp.isnan(df2[longfilt1].astype('float')))

        modelfilta = df2[longfilta] ; modelfiltb = df2[longfiltb]
        modelfilt1 = df2[longfilt1] ; modelfilt2 = df2[longfilt2]
        
        modelcolor = -1*modelfilta+\
                    modelfiltb
        modelres = -1*modelfilt1+\
                    modelfilt2
        
        synthcut = (modelcolor > synth_gi_range[(catname) ][0]) & (modelcolor < synth_gi_range[catname][1]) & synthcut

        modelcolor=modelcolor[synthcut]
        modelres=modelres[synthcut]
        synthx,synthy,sigmamodel,yres,popt,pcov = itersigmacut_linefit(modelcolor,modelres,niter=1,nsigma=3)
        popt=jnp.array([popt[0],popt[1] + off2-off1 - popt[0]* (offb-offa)])
        synthxs.append(synthx); synthys.append(synthy)
    
        modelress.append(modelres +off2-off1)
        modelcolors.append(modelcolor + offb-offa)
        cats.append(catname)
        synthpopts.append(popt)
        synthpcovs.append(pcov)
        
        dms = datares - line(datacolor,*popt)
        #WHY IS THIS A MEAN
        chires = jnp.mean(dms)
        chireserrsq = (sigmadata/jnp.sqrt(dms.size ))**2+errfloors[surv2]**2
        chi2 += (chires**2/chireserrsq)

     
   
    
    #print(chi2)
    return chi2result(chi2=chi2,datax=datax,datay=datay,synthxs=synthxs,synthys=synthys,
     data_popt=data_popt,data_pcov=data_pcov,sigmadata=sigmadata, cats=cats,synthpopts=synthpopts,
     synthpcovs=synthpcovs,sigmasynth=sigmamodel,modelcolors=modelcolors,modelress=modelress,
     surv1=surv1,colorfilta=colorfilta,colorfiltb=colorfiltb,yfilt1=yfilt1,surv2=surv2,yfilt2=yfilt2,
     
     shift=shift,
     obslongfilt1=obslongfilt1,obslongfilt2=obslongfilt2,obslongfilta=obslongfilta,obslongfiltb=obslongfiltb)

#plotcomp2 used to live here

def unwravel_params(params,surveynames,fixsurveynames):
    i = 0
    outofbounds = False
    paramsdict = {}
    paramsnames = []
    for survey in surveynames:
        if survey in fixsurveynames: continue
        filts = obsfilts[survmap[survey]]
        
        for ofilt in filts:
            
            filt = snanafiltsr[survey][ofilt]
            #if ('PS1' not in survey) | (filt!='g'):
            paramsdict[survey+'-'+filt+'_offset'] = params[i]
            outofbounds= outofbounds | ((params[i]<-1.5) | (params[i]>1.5))
            paramsnames.append(survey+'-'+filt+'_offset')
            i+=1

    for survey in fixsurveynames:
        filts = obsfilts[survmap[survey]]
        for ofilt in filts:
            filt = snanafiltsr[survey][ofilt]
            paramsdict[survey+'-'+filt+'_offset'] = 0
            paramsnames.append(survey+'-'+filt+'_offset')

    return paramsdict,outofbounds,paramsnames


def full_likelihood(surveys_for_chisq, fixsurveynames,surveydata,obsdfs, params,doplot=False,subscript='',outputdir='',tableout=None):
    chisqtot=0
    paramsdict,outofbounds,paramsnames = unwravel_params(params,surveys_for_chisq,fixsurveynames)
    if doplot and tableout is None: raise ValueError('No table file provided')
    lp= jax.lax.cond(outofbounds, lambda : -np.inf, lambda :0.)

    surv1s = []
    surv2s = []
    filtas = []
    filtbs = []
    filt1s = []
    filt2s = []

     
    #TODO - there's gotta be a better way to load this information
    # because this requires basically every survey to be used at once.
    #I think this fucked me up again :p

    #Only documenting this one, the rest share the same setup. 
    if "DES" in surveys_for_chisq:
        surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1']) #always PS1
        surv2s.extend([  'DES',  'DES',  'DES',  'DES']) #Survey to compare
        filtas.extend([    'g',    'g',    'g',    'g']) #first filter for colour
        filtbs.extend([    'r',    'i',    'i',    'i']) #second filter for colour
        filt1s.extend([    'g',    'r',    'i',    'z']) # PS1 magnitude band
        filt2s.extend([    'g',    'r',    'i',    'z']) # DES magnitude band
    
    if "CSP" in surveys_for_chisq:
        surv1s.extend([    'PS1',    'PS1',    'PS1',    'PS1',    'PS1',   'PS1',   'PS1',   'PS1'])
        surv2s.extend([ 'CSP', 'CSP', 'CSP', 'CSP', 'CSP','CSP','CSP','CSP'])
        filtas.extend([      'g',      'g',      'g',      'g',      'g',     'g',     'g',     'g'])
        filtbs.extend([      'r',      'i',      'i',      'r',      'i',     'i',     'i',     'i'])
        filt1s.extend([      'g',      'r',      'i',      'g',      'r',     'r',     'r',     'r'])
        filt2s.extend([      'g',      'r',      'i',      'B',      'V',     'o',     'm',     'n'])

    if "PS1SN" in surveys_for_chisq:
        surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
        surv2s.extend([ 'PS1SN', 'PS1SN', 'PS1SN', 'PS1SN'])
        filtas.extend([    'g',    'g',    'g',    'r'])
        filtbs.extend([    'r',    'i',    'i',    'z'])
        filt1s.extend([    'g',    'r',    'i',    'z'])
        filt2s.extend([    'g',    'r',    'i',    'z'])

    if "Foundation" in surveys_for_chisq:
        surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
        surv2s.extend([ 'Foundation', 'Foundation', 'Foundation', 'Foundation'])
        filtas.extend([    'g',    'g',    'g',    'r'])
        filtbs.extend([    'r',    'i',    'i',    'z'])
        filt1s.extend([    'g',    'r',    'i',    'z'])
        filt2s.extend([    'g',    'r',    'i',    'z'])
   
    if "ZTF" in surveys_for_chisq:
        surv1s.extend([  'PS1',  'PS1',  'PS1'])
        surv2s.extend([ 'ZTF', 'ZTF', 'ZTF'])
        filtas.extend([    'g',    'g',    'g'])
        filtbs.extend([    'r',    'i',    'i'])
        filt1s.extend([    'g',    'r',    'i'])
        filt2s.extend([    'g',    'r',    'i'])

    if "ZTFS" in surveys_for_chisq:
        surv1s.extend([  'PS1',  'PS1',  'PS1'])
        surv2s.extend([ 'ZTFS', 'ZTFS', 'ZTFS'])
        filtas.extend([    'g',    'g',    'g'])
        filtbs.extend([    'r',    'i',    'i'])
        filt1s.extend([    'g',    'r',    'i'])
        filt2s.extend([    'g',    'r',    'i'])

    if "ZTFD" in surveys_for_chisq:
        surv1s.extend([  'PS1',  'PS1',  'PS1'])
        surv2s.extend([ 'ZTFD', 'ZTFD', 'ZTFD'])
        filtas.extend([    'g',    'g',    'g'])
        filtbs.extend([    'r',    'i',    'i'])
        filt1s.extend([    'g',    'r',    'i'])
        filt2s.extend([    'g',    'r',    'i'])

    if "SDSS" in surveys_for_chisq:
        surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
        surv2s.extend([ 'SDSS', 'SDSS', 'SDSS', 'SDSS'])
        filtas.extend([    'g',    'g',    'g',    'i'])
        filtbs.extend([    'r',    'i',    'i',    'z'])
        filt1s.extend([    'g',    'r',    'i',    'z'])
        filt2s.extend([    'g',    'r',    'i',    'z'])

    if "SNLS" in surveys_for_chisq:
        surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
        surv2s.extend([ 'SNLS', 'SNLS', 'SNLS', 'SNLS'])
        filtas.extend([    'g',    'g',    'g',    'r'])
        filtbs.extend([    'r',    'i',    'i',    'z'])
        filt1s.extend([    'g',    'r',    'i',    'z'])
        filt2s.extend([    'g',    'r',    'i',    'z'])

    if "CFA3S" in surveys_for_chisq:
        surv1s.extend([   'PS1',   'PS1',   'PS1',   'PS1'])
        surv2s.extend([ 'CFA3S', 'CFA3S', 'CFA3S', 'CFA3S'])
        filtas.extend([     'g',     'g',     'g',     'g'])
        filtbs.extend([     'r',     'i',     'i',     'i'])
        filt1s.extend([     'g',     'r',     'r',     'i'])
        filt2s.extend([     'B',     'V',    'R',     'I'])

    if "CFA3K" in surveys_for_chisq:
        surv1s.extend([   'PS1',   'PS1',   'PS1',   'PS1'])
        surv2s.extend([ 'CFA3K', 'CFA3K', 'CFA3K', 'CFA3K'])
        filtas.extend([     'g',     'g',     'g',     'g'])
        filtbs.extend([     'r',     'i',     'i',     'i'])
        filt1s.extend([     'g',     'r',     'r',     'i'])
        filt2s.extend([     'B',     'V',     'r',     'i'])
    
    totalchisq = 0

    weightsum = 0
    chi2v = []

    passsurvey={surv:{name: surveydata[surv][name].values for name in list(surveydata[surv])  if surveydata[surv][name].dtype in [np.dtype(int), np.dtype(float)] } for surv in surveydata}
    for surv in surveydata: 
        passsurvey[surv]['standard_catagory'] = surveydata[surv]['standard_catagory'].values=='calspec23'
    passobsdfs={surv:{name: obsdfs[surv][name].values for name in list(obsdfs[surv]) if obsdfs[surv][name].dtype in [np.dtype(int), np.dtype(float)] } for surv in obsdfs}
    for surv1,surv2,filta,filtb,filt1,filt2 in zip(surv1s,surv2s,filtas,filtbs,filt1s,filt2s):
        chi2results = getchi_forone(paramsdict,passsurvey, passobsdfs,surv1,surv1,surv2,filta,filtb,filt1,filt2,
                                off1=paramsdict[surv1+'-'+filt1+'_offset'],off2=paramsdict[surv2+'-'+filt2+'_offset'],
                                offa=paramsdict[surv1+'-'+filta+'_offset'],offb=paramsdict[surv1+'-'+filtb+'_offset'])
        if doplot: plot_forone(chi2results,subscript,outputdir,tableout)
        chi2v.append(chi2results.chi2) #Would like to add the survey info as well
        totalchisq+=chi2results.chi2
    lp += lnprior(paramsdict)
    return lp -.5*totalchisq


def lnprior(paramsdict):

    priordict = { 
        'PS1-g_offset':[0,.01],
        'PS1-r_offset':[0,.01],
        'PS1-i_offset':[0,.01],
        'PS1-z_offset':[0,.01],

        }
    
#        'DES-g_offset':[0,.01],
#        'DES-r_offset':[0,.01],
#        'DES-i_offset':[0,.01],
#        'DES-z_offset':[0,.01],

    '''
        'DES_g_lamshift':[0,20],
        'DES_r_lamshift':[0,20],
        'DES_i_lamshift':[0,20],
        'DES_z_lamshift':[0,20],

        'SDSS_g_offset':[0,.02],
        'SDSS_r_offset':[0,.02],
        'SDSS_i_offset':[0,.02],
        'SDSS_z_offset':[0,.02],
        'SDSS_g_lamshift':[0,50],
        'SDSS_r_lamshift':[0,50],
        'SDSS_i_lamshift':[0,50],
        'SDSS_z_lamshift':[0,50],

        'SNLS_g_offset':[0,.01],
        'SNLS_r_offset':[0,.01],
        'SNLS_i_offset':[0,.01],
        'SNLS_z_offset':[0,.01],
        'SNLS_g_lamshift':[0,20],
        'SNLS_r_lamshift':[0,20],
        'SNLS_i_lamshift':[0,20],
        'SNLS_z_lamshift':[0,20],
    
    '''
    

    lp = 0
    for priorparam,prior in priordict.items():
        mu = prior[0]
        sigma = prior[1]
        lp += -0.5*(paramsdict[priorparam]-mu)**2/sigma**2


    return lp

##Put old ugly code with plotting in here
def plot_forone(result,subscript, outputdir,tableout): 

    
    plt.clf()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(result.datax,result.datay,
            color='k',alpha=.3, edgecolor=None, label='Observed Mags Chisq %.2f'%(result.chi2),s=5,zorder=5)
    data_slope=result.data_popt[0] ; data_slope_err = (result.data_pcov[0,0]**2+result.sigmadata**2)**.5
    
    ax.plot(result.datax, line(np.array(result.datax),result.data_popt[0],result.data_popt[1]), c="k", lw=4, zorder=20)
    #here starts the two synthetics
    linecolors = ['goldenrod', "#0C6291"]
    for cat,popt,pcov,mc,mr,linecolor in zip(result.cats,result.synthpopts,result.synthpcovs
                                            ,result.modelcolors,result.modelress,linecolors):
        offmean = np.mean(line(result.datax,popt[0],popt[1]) - result.datay)
        offmed = np.median(line(result.datax,popt[0],popt[1]) - result.datay)
        synth_slope = popt[0]
        synth_slope_err = pcov[0,0]
        sigma = (data_slope-synth_slope)/np.sqrt(data_slope_err**2+synth_slope_err**2)

        ## Start plots here
        ax.plot(result.datax,line(result.datax,popt[0],popt[1]),
                lw=2, c=linecolor, zorder=19,
                label='Synthetic Pred: %s\nOffMean: %.3f\nOffMedian: %.3f\n'%(cat,offmean,offmed)) #lines
        ax.scatter(mc, mr, alpha=.3, s=5, edgecolor=None, zorder=10, c=linecolor) #Points
        ax.legend(framealpha=0)
        ax.set_xlabel(f'{result.obslongfilta} - {result.obslongfiltb}', alpha=0.8)
        ax.set_ylabel(f'{result.obslongfilt1} - {result.obslongfilt2}', alpha=0.8)

        labels = np.quantile(result.datax, np.arange(0, 1.1, 0.2))
        ax.set_xticks(ticks=labels)
        ax.set_xticklabels(np.around(labels,2), rotation=90)
        labels = np.quantile(result.datay, np.arange(0, 1.1, 0.2))
        ax.set_yticks(ticks=labels)
        ax.set_yticklabels(labels=np.around(labels,2))
        for speen in ['right', 'top', 'left', 'bottom']:
            ax.spines[speen].set_visible(False)

        ## End plot stuff

        tableout.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.4f\t%d\t%.3f+-%.3f\t%.3f+-%.3f\t%.1f\t%.1f\n'%(result.surv1,result.colorfilta,result.colorfiltb,result.yfilt1,result.surv2,result.yfilt2,cat,offmean,result.datax.size,data_slope,data_slope_err,synth_slope,synth_slope_err,sigma,result.shift))
    fname='overlay_on_obs_%s_%s-%s_%s_%s_%s_%s_%s.png'%(result.surv1,result.colorfilta,result.colorfiltb,result.yfilt1,result.surv2,result.yfilt2,'all',subscript)
    if outputdir: outpath= path.join('plots',path.join(outputdir, fname))
    else: outpath=path.join('plots',fname)
    plt.savefig(outpath, bbox_inches="tight")

    print(f'upload {outpath}')

    plt.close('all') #BRODIE - hopefully this doesn't break plots

def prep_config(args):

    REDO = args.IRSA #yes I know there's a discrepancy in naming here
    MCMC = args.MCMC
    DEBUG = args.DEBUG
    FAKES = args.FAKES

    return REDO, MCMC, DEBUG, FAKES

def get_args():
    parser = argparse.ArgumentParser()

    msg = "HELP menu for config options"

    msg = "Default False. Redo the IRSA dust maps. This is a necessary step and should be run before doing the full MCMC steps. \n I'm still working on a cleaner way of setting this up that doesn't require guesswork."
    parser.add_argument("--IRSA", help=msg, action='store_true')
    parser.set_defaults(IRSA=False)

    msg = "Default False. Run the full MCMC process to determine band offsets. \n Much of the debugging also lives in this code, so only set to true if you're ready."
    parser.add_argument("--MCMC", help=msg, action='store_true')
    parser.set_defaults(MCMC=False)

    msg = "Default False. Enables a host of print statements for Brodie to debug with."
    parser.add_argument("--DEBUG", help = msg, action="store_true")
    parser.set_defaults(DEBUG=False)

    msg = "Default False. Grabs fake stars to test recovery of input parameters."
    parser.add_argument("--FAKES", help = msg, action='store_true')
    parser.set_defaults(FAKES=False)
    
    parser.add_argument("--target_acceptance", help = "Target acceptance rate for hamiltonian MCMC", type=float, default=0.95)
    
    parser.add_argument("--n_adaptive", help = "Number of steps to adapt MCMC hyperparameters", type=int, default=2000)
    parser.add_argument("--output", help = "Path to write output chains to (.npz format)", type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    REDO, MCMC, DEBUG, FAKES = prep_config(args)
    tableout = open('preoffsetsaper.dat','w')
    tableout.write('COLORSURV COLORFILT1 COLORFILT2 OFFSETFILT1 OFFSETSURV OFFSETFILT2 SPECLIB OFFSET NDATA D_SLOPE S_SLOPE SIGMA SHIFT\n')
    print('reading in survey data')

    surveys_for_chisq = config['surveys_for_dovekie']
    if args.output is None: outname = config['chainsfile']
    else: outname= args.output
    #surveys_for_chisq = ['PS1', 'CFA3K', 'PS1SN'] #keep this one around for quick IRSA updates!
    fixsurveynames = []

    surveydata = get_all_shifts(surveys_for_chisq)
    obsdfs = get_all_obsdfs(surveys_for_chisq, REDO, FAKES)
    print('got all survey data')

    if REDO:
        print("Done acquiring IRSA maps. Quitting now to avoid confusion.")
        quit()

    nparams=0
    pos = []
    for s in surveys_for_chisq:
        #print(obsfilts)
        of = obsfilts[survmap[s]]
        if s in fixsurveynames:
            for f in of: 
                continue
        else:
            for f in of: 
                nparams+=1 #offset and lamshift
                pos.append(0)
    
    
    full_likelihood_data= partial(full_likelihood,surveys_for_chisq, fixsurveynames,surveydata,obsdfs)
    full_likelihood_data(pos,subscript='beforeaper',doplot=True,tableout=tableout)

    _,_,labels=unwravel_params(pos,surveys_for_chisq,fixsurveynames)

    tableout.close()

    if MCMC == False:
        print("Not running the full MCMC, quitting now.")
        quit()

    os.environ["OMP_NUM_THREADS"] = "1"

    if isshift:
        print("You are currently grabbing all the shifted values, quitting!")
        quit()


    key=random.PRNGKey(34581339453)
    initkey, samplekey= random.split(key)
    n_samples = 5000
    theta0 = random.normal(initkey, shape=(nparams,))*0.01
    
    sampler = NUTS(theta0, logp=full_likelihood_data, target_acceptance=target_acceptance, M_adapt=n_burnin)
    key, samples, step_size = sampler.sample(n_samples, samplekey)
    loglikes=jax.vmap(full_likelihood,in_axes=0)(samples)
    np.savez(outname,samples=samples,labels=labels,surveys_for_chisq=surveys_for_chisq)
    
    
    
