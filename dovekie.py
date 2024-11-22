import pandas as pd
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import os, sys
from scipy.optimize import curve_fit
from scipy import linalg
from scipy import interpolate, optimize as op, stats
from glob import glob
sys.path.insert(1, 'scripts/')
from helpers import *
import time, pickle
from astroquery.irsa_dust import IrsaDust
import astropy.coordinates as coord
import astropy.units as u
import argparse
from functools import partial
import jax
from jax import numpy as jnp
from jax import random as random
from jaxnuts.sampler import NUTS
from jax.scipy import linalg as jlinalg

from collections import namedtuple
from os import path

#search for "TODO"
isshift = False
global DEBUG
DEBUG = False

jsonload = 'DOVEKIE_DEFS.yml' #where all the important but unwieldy dictionaries live
config = load_config(jsonload)
survmap, survmap4shift, survfiltmap, obssurvmap, revobssurvmap, revobssurvmapforsnana, survcolormin, survcolormax, synth_gi_range, obsfilts, snanafilts, snanafiltsr, relativeweights, errfloors ,target_acceptance , n_burnin, whitedwarf_obs_loc = prep_config(config)


obscolors_by_survey = {'PS1':['PS1-g','PS1-i']} #dodgy, feel like this should be tonry

filter_means = pd.read_csv('filter_means.csv') 

filter_means = filter_means.set_index(['SURVEYFILTER']).to_dict()['MEANLAMBDA ']

chi2result=namedtuple('chi2result',['chi2','datax','datay','synthxs','synthys',
     'data_popt','data_pcov','cats','synthpopts',
     'synthpcovs','modelcolors','modelress','sigmasynth','sigmadata',
     'obslongfilt1','obslongfilt2','obslongfilta','obslongfiltb',
     'surv1','colorfilta','colorfiltb','yfilt1','surv2','yfilt2','shift'])

def get_whitedwarf_synths(surveys):
    whitedwarfsynths={}
    for survey in surveys:
        files = glob('output_synthetic_magsaper/bboyd_synth_%s_shift_*.000.txt'%survmap4shift[survey])
        if len(files):
            for fname in files: whitedwarfsynths[survey]= pd.read_csv(fname,sep='\s+') 
    return whitedwarfsynths

def get_all_shifts(surveys, reference_surveys): #acquires all the surveys and collates them. 
    surveydfs = {}
    for survey in surveys: #5/11/24 - need to upgrade this to handle new GAIA format
        files = glob('output_synthetic_magsaper/synth_%s_shift_*.000.txt'%survmap4shift[survey]) #TODO - better determination of whether or not there are lambda shifts and what to do if there are
        print(files)
        if len(files) > 1:
            print("Picking up shifts!")
            isshift = True
        dfl = []
        for f in files:
            try:
                tdf = pd.read_csv(f,sep=" ") #formerly delim_whitespace      
                if 'PS1_' in f:
                    tdf = tdf[(tdf['PS1-g']-tdf['PS1-i'])>.25]
                    tdf = tdf[(tdf['PS1-g']-tdf['PS1-i'])<1.6]
                    tdf = tdf[(tdf['PS1-g']-tdf['PS1-r'])>.25]
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
        if survey not in reference_surveys: #need to fix this to work with GAIA
            for refsurv in reference_surveys:
                surveydfs[survey] = pd.merge(surveydfs[survey],surveydfs[refsurv],left_on='standard',right_on='standard',suffixes=('','_b'))
    return surveydfs
    
def get_all_obsdfs(surveys, redo=False, fakes=False): 
    surveydfs = {}
    surveydfs_wext = {}
    for survey in surveys:
        survname = obssurvmap[survey]
        if survey == 'PS1': continue
        elif survey == "GAIA": continue
        realdirname = 'output_observed_apermags'
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
                if fakes:
                    obsdf = pd.read_csv(f'{fakes}/{survname}_observed.csv')
                else:
                    obsdf = pd.read_csv(f'{realdirname}+AV/{survname}_observed.csv')
                    obsdf = obsdf[(obsdf['PS1-g']-obsdf['PS1-g_AV'])-
                          (obsdf['PS1-i']-obsdf['PS1-i_AV']) < survcolormax[survey]]
                    obsdf = obsdf[(obsdf['PS1-g']-obsdf['PS1-g_AV'])-
                          (obsdf['PS1-i']-obsdf['PS1-i_AV']) > survcolormin[survey]]
                    obsdf = obsdf.replace(-999, np.nan)
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
                  filtshift=0,off1=0,off2=0,offa=0,offb=0,
                  calspecslope=0,calspecmeanlambda=4383.15,ngslslope=0,ngslmeanlambda=5507.09,
                  doplot=False,subscript='', outputdir='synthetic',speclibrary=None): #where the magic happens I suppose
    ssurv2 = survmap[surv2]
    df2 = surveydata[surv2]

    chi2 = 0

    #changed these back to dashes
    longfilta = 'PS1-'+colorfilta ; longfiltb = 'PS1-'+colorfiltb #11/5/24 changed survfiltmap[colorsurvab] to 'PS1'
    if "GAIA_" in surv1:
        #Do nothing for longfilt1 or longfilt2 
        survtemp1 = surv1.replace("GAIA_", '') ; survtemp2 = surv2.replace("GAIA_", '')
        obslongfilt1 = 'GAIA_'+obssurvmap[survtemp1]+'-'+yfilt1
        obslongfilta = 'PS1-'+colorfilta ; obslongfiltb = 'PS1-'+colorfiltb #should always be PS1
        if ('CSP' in surv2.upper()):
            obslongfilt2 = 'GAIA_'+obssurvmap[survtemp2]+'-'+yfilt2.replace('o','V').replace('m','V').replace('n','V')
        elif ('KAIT' in surv2.upper()) | ('NICKEL' in surv2.upper()):
            obslongfilt2 = 'GAIA_'+obssurvmap[survtemp2]+'-'+snanafilts[survtemp2][yfilt2]
        elif 'ASASSN' in surv2.upper():
            obslongfilt2 = 'GAIA_'+obssurvmap[survtemp2]+'-'+snanafilts[survtemp2][yfilt2]
        else:
            obslongfilt2 = obssurvmap[surv2]+'-'+yfilt2 #
    else:
        longfilt1 = survfiltmap[surv1]+'-'+yfilt1 ; longfilt2 = survfiltmap[surv2]+'-'+yfilt2
        obslongfilt1 = obssurvmap[surv1]+'-'+yfilt1
        if ('CSP' in surv2.upper()):
            obslongfilt2 = obssurvmap[surv2]+'-'+yfilt2.replace('o','V').replace('m','V').replace('n','V')
        elif ('KAIT' in surv2.upper()) | ('NICKEL' in surv2.upper()):
            obslongfilt2 = obssurvmap[surv2]+'-'+snanafilts[surv2][yfilt2]
        elif 'ASASSN' in surv2.upper():
            obslongfilt2 = obssurvmap[surv2]+'-'+snanafilts[surv2][yfilt2]
        else:
            obslongfilt2 = obssurvmap[surv2]+'-'+yfilt2 #
        obslongfilta = obssurvmap[colorsurvab]+'-'+colorfilta ; obslongfiltb = obssurvmap[colorsurvab]+'-'+colorfiltb

    #JAXX stuff starts here 
    obsdf = obsdfs[surv2] #grabs the observed points from the relevant survey
    #if DEBUG: print(obsdf.keys(), surv2)
    #if DEBUG: print(obssurvmap.keys(),surv1,yfilt1, surv2, yfilt2)
    yr=obsdf[obslongfilt1]-obsdf[obslongfilt2] #observed filter1 - observed filter 2 
    datacut = np.abs(yr)<1 #only uses things lower than 1
    if "GAIA_" in obslongfilt1:
        datacolor = (obsdf[obslongfilta]-obsdf[obslongfilta+"_AV"])-(obsdf[obslongfiltb] -obsdf[obslongfiltb+"_AV"])
        if ('CSP' in surv2.upper()):
            obslongfilt2 = obssurvmap[surv2]+'-'+yfilt2.replace('o','V').replace('m','V').replace('n','V')
        datares = (obsdf[obslongfilt2])-(obsdf[obslongfilt1]) #need to add Gaia in here
    else:
        datacolor = (obsdf[obslongfilta]-obsdf[obslongfilta+"_AV"])-(obsdf[obslongfiltb] -obsdf[obslongfiltb+"_AV"])
        datares = (obsdf[obslongfilt2]-obsdf[obslongfilt2+'_AV'])-(obsdf[obslongfilt1]-obsdf[obslongfilt1+'_AV']) #need to add Gaia in here
    datacolor=datacolor[datacut]
    datares=datares[datacut]
    if DEBUG: print(np.std(datacolor), np.std(datares))
    datax,datay,sigmadata,yresd,data_popt,data_pcov = itersigmacut_linefit(datacolor,
                                                         datares,# np.ones(datacut.sum(),dtype=bool),
                                                         niter=2,nsigma=3)

    synthxs,synthys, cats,synthpopts,synthpcovs,modelcolors,modelress =  ([] for i in range(7))

    if speclibrary is None: 
       libraries=['stis_ngsl_v2','calspec23']
    else: libraries=[speclibrary]
    for i,catname in enumerate(libraries):
        cat=df2['standard_catagory'].values==catname
        if DEBUG: print(df2.keys(), surv2, surv1) 
        if "GAIA_" in surv1:
            modelfilta = df2[longfilta] ; modelfiltb = df2[longfiltb]
            modelfilt1 = np.zeros(modelfilta.size) ; modelfilt2 = np.zeros(modelfiltb.size)
            synthcut = np.ones(modelfilta.size, dtype=bool)
        else:
            synthcut = (cat) & \
                       (~np.isnan(df2[longfilt2].astype('float'))) & \
                       (~np.isnan(df2[longfilt1].astype('float')))

            modelfilta = df2[longfilta] ; modelfiltb = df2[longfiltb]
            modelfilt1 = df2[longfilt1] ; modelfilt2 = df2[longfilt2]
        
        modelcolor = modelfilta - modelfiltb
        modelres = -1*modelfilt1+\
                    modelfilt2
        synthcut = (modelcolor > synth_gi_range[(catname) ][0]) & (modelcolor < synth_gi_range[catname][1]) & synthcut

        modelcolor=modelcolor[synthcut]
        modelres=modelres[synthcut]
        synthx,synthy,sigmamodel,yres,popt,pcov = itersigmacut_linefit(modelcolor,modelres,niter=1,nsigma=3)
        

        #CORRECT
        popt=jnp.array([popt[0] ,popt[1] +( off2-off1 - popt[0]* (offb-offa))])
        synthxs.append(synthx); synthys.append(synthy)
    
        try:
            modelress.append(modelres.values +off2-off1)
            modelcolors.append(modelcolor.values + offb-offa)
        except AttributeError:
            modelress.append(modelres +off2-off1)
            modelcolors.append(modelcolor + offb-offa)            
            if "GAIA" not in surv1:
                print(f"Model residual {surv1} - {surv2} is not stored in a pandas dataframe and may be wrong if you are seeing this error.")
        cats.append(catname)
        synthpopts.append(popt)
        synthpcovs.append(pcov)
        
        dms = datares.values - line(datacolor.values,*popt)
        #WHY IS THIS A MEAN
        chires = jnp.mean(dms)
        chireserrsq = (sigmadata/jnp.sqrt(dms.size ))**2+errfloors[surv2]**2
        chi2 += (chires**2/chireserrsq)     
   
    chi2/=len(libraries)
    #print(chi2)
    return chi2result(chi2=chi2,datax=datax,datay=datay,synthxs=synthxs,synthys=synthys,
     data_popt=data_popt,data_pcov=data_pcov,sigmadata=sigmadata, cats=cats,synthpopts=synthpopts,
     synthpcovs=synthpcovs,sigmasynth=sigmamodel,modelcolors=modelcolors,modelress=modelress,
     surv1=surv1,colorfilta=colorfilta,colorfiltb=colorfiltb,yfilt1=yfilt1,surv2=surv2,yfilt2=yfilt2,
     
     shift=filtshift,
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
            if 'GAIA' in survey:
                pass
            else:
                paramsdict[survey+'-'+filt+'_offset'] = params[i]
                paramsnames.append(survey+'-'+filt+'_offset')
            outofbounds= outofbounds | ((params[i]<-1.5) | (params[i]>1.5))
            if 'GAIA' in reference_surveys: 
                paramsdict['GAIA_'+survey+'-'+filt+'_offset'] = 0 #5/11/24 added to try and fix ?
            #paramsnames.append(survey+'-'+filt+'_offset')
            i+=1

    for survey in fixsurveynames:
        filts = obsfilts[survmap[survey]]
        for ofilt in filts:
            filt = snanafiltsr[survey][ofilt]
            paramsdict[survey+'-'+filt+'_offset'] = 0
            paramsnames.append(survey+'-'+filt+'_offset')

    return paramsdict,outofbounds,paramsnames
    
wdresult=namedtuple('wdresult',['chi2','resids','errs', 'errpars','covariance','filts'])

def calc_wd_chisq(paramsdict,whitedwarf_seds,whitedwarf_obs, storedvals=None):
    if storedvals is None:
        for surv in whitedwarf_seds: whitedwarf_seds[surv]=whitedwarf_seds[surv].rename(columns={'standard':'Object'})
        
        wdsurveys=np.unique([x.split('-')[0] for x in list(whitedwarf_obs) if '-' in x])
        accum=whitedwarf_seds[wdsurveys[0]]
        for key in wdsurveys[1:]:
            accum=pd.merge(accum, whitedwarf_seds[key], left_index=True, right_index=True,suffixes=('','_y'))
            accum=accum.drop(
                accum.filter(regex='_y$').columns, axis=1)       
        filts=[(survey+'-' + filt) for filt in 'griz' for survey in wdsurveys ]
        isbad=(((np.isnan(accum[filts])| (accum[filts]<-20)).sum(axis=1))>0)
        print(f'{(isbad).sum():d} bad SED samples' )
        accum=accum[~isbad]
        grouped=accum.groupby('Object')
        covsbyobject= [np.cov(grouped.get_group(group)[filts].values.T) for group in grouped.groups]
        overallcov=np.mean([x for x,group in  zip(covsbyobject,grouped.groups)],axis=0)
    #     plt.imshow(overallcov)
        whitedwarftotal=pd.merge(whitedwarf_obs,grouped[filts].mean(),left_index=True,right_index=True,suffixes=('_obs','_synth'))
        whitedwarftotal=whitedwarftotal.replace(-999.,np.nan)
        resids={}
        errs={}
        errpars={}
        residcorrected={}
        rescalederr={}
        def negloglike(x,synth,obs,obserr):
            mean,errscale,errfloor=x
            return -stats.norm.logpdf(synth-obs,loc=mean,scale=np.hypot(errscale*obserr,errfloor)).sum()
        from scipy import optimize as op
        for filt in filts:
            synth,obs,obserr=whitedwarftotal[filt+'_synth'],whitedwarftotal[filt+'_obs'], whitedwarftotal[filt+'-err']
            isgood=(synth>0) &( obs>0)&(~np.isnan(obs))&(~np.isnan(obserr))
            result=op.minimize(negloglike,[0,1,0],args=(synth[isgood],obs[isgood],(obserr[isgood])),bounds=[(-.2,.2),(0,10),(0,.1)])
            errscale,errfloor=result.x[1:]
            errpars[filt]=errscale,errfloor
    
            resids[filt]=(obs-synth)[isgood].values#-paramsdict[filt+'_offset']
            errs[filt]=obserr[isgood].values
            rescalederr[filt]=np.hypot(errscale*obserr[isgood],errfloor).values
    else:
        resids=storedvals.resids
        filts=storedvals.filts
        overallcov=storedvals.covariance
        errs=storedvals.errs
        errpars=storedvals.errpars
        rescalederr={}
        for filt in filts:
            errscale,errfloor=storedvals.errpars[filt]
            rescalederr[filt]=np.hypot(errscale*errs[filt],errfloor)
            
    covs=np.diag(np.concatenate([rescalederr[x] for x in filts])**2) 
    indexes=np.concatenate([np.repeat(i, resids[filt].size) for i, filt in enumerate(filts)]  )
    covs+=np.array([[overallcov[i,j] for i in indexes] for j in indexes])
    
    allresids=jnp.concatenate([resids[x]-paramsdict[x+'_offset'] for x in filts])
    transform=linalg.cholesky(covs,lower=True)
    design=jlinalg.solve_triangular(transform,allresids,lower=True)
    chisq= design @ design
    return wdresult(chisq, resids,errs, errpars,overallcov,filts)
    

def plotwhitedwarfresids(filt, outdir, wdresults,paramsdict,):
    fig, ax = plt.subplots()
    plt.xlim(1e-3,.2)
    plt.ylim(-.1,.1)

    colourz=['#111111', '#01417d', '#797677', '#c7b168', '#f1f1f1']

    errscale,errfloor=wdresults.errpars[filt]
    line1=plt.errorbar(wdresults.errs[filt],wdresults.resids[filt],yerr=wdresults.errs[filt],fmt='o',label=f'raw errors ({wdresults.resids[filt].size} points)', c=colourz[1], alpha=0.5)
    errscale,errfloor=wdresults.errpars[filt]
    scalederr=np.hypot(errscale*wdresults.errs[filt],errfloor)
    
    mean=np.average((wdresults.resids)[filt],weights=1/(scalederr**2))
    line1=plt.errorbar(np.clip(scalederr,*plt.xlim()),np.clip((wdresults.resids)[filt],*plt.ylim()),yerr=scalederr,fmt='o', c=colourz[3], alpha=0.5,
                       label=f'rescaled errors\n$\sigma \leftarrow \sqrt{{({errscale:.2f}\sigma )^2 + {errfloor:.3f} ^2 }}$')
    chi2=((((wdresults.resids)[filt]-mean)/scalederr)**2).sum()
    plt.axhline(mean,color='k',linestyle=':',label=f'WD mean: $\\chi^2$= {chi2:.2f}', alpha=0.5)
    if paramsdict is not None: 
        chi2=((((wdresults.resids)[filt]-paramsdict[filt+'_offset'])/scalederr)**2).sum()
        plt.axhline(paramsdict[filt+'_offset'],color='dimgrey',linestyle='--',label=f'Derived offset: $\\chi^2$= {chi2:.2f}')
    plt.plot(np.linspace(*plt.xlim(),100),mean+np.linspace(*plt.xlim(),100),c='dimgrey', alpha=0.5)
    plt.plot(np.linspace(*plt.xlim(),100),mean-np.linspace(*plt.xlim(),100),c='dimgrey', alpha=0.5)
    
    text=plt.text(.5,.2,'',transform=plt.gca().transAxes)
    plt.xscale('log')
    plt.xlabel('Photo-error', fontsize=15, color="dimgrey")
    plt.ylabel('WD Residual off mean', fontsize=15, color="dimgrey")
    plt.legend(loc='upper left', frameon=False)
    plt.title(filt)
    plt.tight_layout()
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.tick_params(axis="both", which="both", labelsize=12, color="dimgrey")
    [t.set_color('dimgrey') for t in ax.xaxis.get_ticklabels()]
    [t.set_color('dimgrey') for t in ax.yaxis.get_ticklabels()]
    fname=f'whitedwarf_resids_{filt}.pdf'
    if outdir: outpath= path.join('plots',path.join(outdir, fname))
    else: outpath=path.join('plots',fname)
    plt.savefig(outpath)
    print(f'writing white dwarf residuals to {outpath}')

def full_likelihood(surveys_for_chisq, fixsurveynames,surveydata,obsdfs,reference_surveys, params,doplot=False,subscript='',outputdir='',tableout=None, whitedwarf_seds=None,whitedwarf_obs= None,biasestimates=None,speclibrary=None):

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
    if 'PS1' in reference_surveys:
        if "DES" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1']) #always PS1
            surv2s.extend([  'DES',  'DES',  'DES',  'DES']) #Survey to compare
            filtas.extend([    'g',    'g',    'g',    'g']) #first filter for colour
            filtbs.extend([    'i',    'i',    'i',    'i']) #second filter for colour
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
            filtbs.extend([    'i',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'i',    'z'])
            filt2s.extend([    'g',    'r',    'i',    'z'])
    
        if "Foundation" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
            surv2s.extend([ 'Foundation', 'Foundation', 'Foundation', 'Foundation'])
            filtas.extend([    'g',    'g',    'g',    'r'])
            filtbs.extend([    'i',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'i',    'z'])
            filt2s.extend([    'g',    'r',    'i',    'z'])
       
        if "ZTF" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1', 'PS1', 'PS1', 'PS1'])
            surv2s.extend([  'ZTF',  'ZTF',  'ZTF', 'ZTF', 'ZTF', 'ZTF'])
            filtas.extend([    'g',    'g',    'g', 'g', 'g', 'g'])
            filtbs.extend([    'i',    'i',    'i', 'i', 'i', 'i'])
            filt1s.extend([    'g',    'r',    'i', 'g', 'r', 'i'])
            filt2s.extend([    'g',    'r',    'i', 'G', 'R', 'I'])
    
        if "ZTFD" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1'])
            surv2s.extend([ 'ZTFD', 'ZTFD', 'ZTFD'])
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
    
    
        if "SDSS" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
            surv2s.extend([ 'SDSS', 'SDSS', 'SDSS', 'SDSS'])
            filtas.extend([    'g',    'g',    'g',    'r'])
            filtbs.extend([    'i',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'i',    'z'])
            filt2s.extend([    'g',    'r',    'i',    'z'])
    
        if "SNLS" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
            surv2s.extend([ 'SNLS', 'SNLS', 'SNLS', 'SNLS'])
            filtas.extend([    'g',    'g',    'g',    'r'])
            filtbs.extend([    'i',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'i',    'z'])
            filt2s.extend([    'g',    'r',    'i',    'z'])
    
        if "CFA3S" in surveys_for_chisq:
            surv1s.extend([   'PS1',   'PS1',   'PS1',   'PS1'])
            surv2s.extend([ 'CFA3S', 'CFA3S', 'CFA3S', 'CFA3S'])
            filtas.extend([     'g',     'g',     'g',     'g'])
            filtbs.extend([     'i',     'i',     'i',     'i'])
            filt1s.extend([     'g',     'r',     'r',     'i'])
            filt2s.extend([     'B',     'V',    'R',     'I'])
    
        if "CFA3K" in surveys_for_chisq:
            surv1s.extend([   'PS1',   'PS1',   'PS1',   'PS1'])
            surv2s.extend([ 'CFA3K', 'CFA3K', 'CFA3K', 'CFA3K'])
            filtas.extend([     'g',     'g',     'g',     'g'])
            filtbs.extend([     'r',     'i',     'i',     'i'])
            filt1s.extend([     'g',     'r',     'r',     'i'])
            filt2s.extend([     'B',     'V',     'r',     'i'])
        
        if "KAIT1MO" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
            surv2s.extend(['KAIT1MO','KAIT1MO','KAIT1MO','KAIT1MO'])
            filtas.extend([    'g',    'g',    'g',    'g'])
            filtbs.extend([    'r',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'r',    'i'])
            filt2s.extend([    'a',    'b',    'c',    'd'])
        
        if "KAIT2MO" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
            surv2s.extend(['KAIT2MO','KAIT2MO','KAIT2MO','KAIT2MO'])
            filtas.extend([    'g',    'g',    'g',    'g'])
            filtbs.extend([    'r',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'r',    'i'])
            filt2s.extend([    'e',    'f',    'g',    'h'])
         
        if "NICKEL1MO" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
            surv2s.extend(['NICKEL1MO','NICKEL1MO','NICKEL1MO','NICKEL1MO'])
            filtas.extend([    'g',    'g',    'g',    'g'])
            filtbs.extend([    'r',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'r',    'i'])
            filt2s.extend([    'I',    'J',    'K',    'L'])
            surv1s.extend([  'PS1',  'PS1', 'PS1',  'PS1'])
    
        if 'NICKEL2MO' in surveys_for_chisq:
            surv2s.extend(['NICKEL2MO','NICKEL2MO','NICKEL2MO','NICKEL2MO'])
            filtas.extend([    'g',    'g',    'g',    'g'])
            filtbs.extend([    'r',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'r',    'i'])
            filt2s.extend([    'M',    'N',    'O',    'P'])
    
        if "KAIT3MO" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
            surv2s.extend(['KAIT3MO','KAIT3MO','KAIT3MO','KAIT3MO'])
            filtas.extend([    'g',    'g',    'g',    'g'])
            filtbs.extend([    'r',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'r',    'i'])
            filt2s.extend([    'A',    'B',    'C',    'D'])
    
        if "KAIT4MO" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
            surv2s.extend(['KAIT4MO','KAIT4MO','KAIT4MO','KAIT4MO'])
            filtas.extend([    'g',    'g',    'g',    'g'])
            filtbs.extend([    'r',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'r',    'i'])
            filt2s.extend([    'E',    'F',    'G',    'H'])
        
        if "KAIT3" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
            surv2s.extend(['KAIT3','KAIT3','KAIT3','KAIT3'])
            filtas.extend([    'g',    'g',    'g',    'g'])
            filtbs.extend([    'r',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'r',    'i'])
            filt2s.extend([    'A',    'B',    'C',    'D'])
    
        if "KAIT4" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
            surv2s.extend(['KAIT4','KAIT4','KAIT4','KAIT4'])
            filtas.extend([    'g',    'g',    'g',    'g'])
            filtbs.extend([    'r',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'r',    'i'])
            filt2s.extend([    'E',    'F',    'G',    'H'])
        
        if "NICKEL1" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
            surv2s.extend(['NICKEL1','NICKEL1','NICKEL1','NICKEL1'])
            filtas.extend([    'g',    'g',    'g',    'g'])
            filtbs.extend([    'r',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'r',    'i'])
            filt2s.extend([    'I',    'J',    'K',    'L'])
    
        if "NICKEL2" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1', 'PS1',  'PS1'])
            surv2s.extend(['NICKEL2','NICKEL2','NICKEL2','NICKEL2'])
            filtas.extend([    'g',    'g',    'g',    'g'])
            filtbs.extend([    'r',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'r',    'i'])
            filt2s.extend([    'M',    'N',    'O',    'P'])
    
        if "SWIFT" in surveys_for_chisq:
            surv1s.extend([  'PS1',       'PS1'])
            surv2s.extend(['SWIFT', 'SWIFT'])
            filtas.extend([    'g',         'g'])
            filtbs.extend([    'r',         'i'])
            filt1s.extend([    'g',         'r'])
            filt2s.extend([    'B',         'V'])
    
        if "ASASSN2" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
            surv2s.extend(['ASASSN2','ASASSN2','ASASSN2','ASASSN2'])
            filtas.extend([    'g',    'g',    'g',    'g'])
            filtbs.extend([    'r',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'r',    'i'])
            filt2s.extend([    'g',    'j',    'i',    'h'])
    
        if "ASASSN1" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
            surv2s.extend(['ASASSN1','ASASSN1','ASASSN1','ASASSN1'])
            filtas.extend([    'g',    'g',    'g',    'g'])
            filtbs.extend([    'r',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'r',    'i'])
            filt2s.extend([    'a',    'b',    'd',    'c'])
    
        if "CFA4P1" in surveys_for_chisq:
            surv1s.extend([  'PS1',       'PS1',     'PS1',    'PS1'])
            surv2s.extend(['CFA4P1', 'CFA4P1','CFA4P1','CFA4P1'])
            filtas.extend([    'g',         'g',       'g',      'g'])
            filtbs.extend([    'r',         'i',       'i',      'i'])
            filt1s.extend([    'g',         'r',       'r',      'i'])
            filt2s.extend([    'B',         'V',       'r',      'i'])
        
        if "CFA4P2" in surveys_for_chisq:
            surv1s.extend([  'PS1',       'PS1',     'PS1',    'PS1'])
            surv2s.extend(['CFA4P2', 'CFA4P2', 'CFA4P2','CFA4P2'])
            filtas.extend([    'g',         'g',       'g',      'g'])
            filtbs.extend([    'r',         'i',       'i',      'i'])
            filt1s.extend([    'g',         'r',       'r',      'i'])
            filt2s.extend([    'B',         'V',       'r',      'i'])

    if 'GAIA' in reference_surveys: #Need to clean this up for the new GAIA stuff 
        if "DES" in surveys_for_chisq:
            surv1s.extend([  'GAIA_DES',  'GAIA_DES',  'GAIA_DES',  'GAIA_DES']) #5/11/24 - new formatting frontier for Gaia integration 
            surv2s.extend([  'DES',  'DES',  'DES',  'DES']) #Survey to compare
            filtas.extend([    'g',    'g',    'g',    'g']) #first filter for colour
            filtbs.extend([    'i',    'i',    'i',    'i']) #second filter for colour
            filt1s.extend([    'g',    'r',    'i',    'z']) # PS1 magnitude band
            filt2s.extend([    'g',    'r',    'i',    'z']) # DES magnitude band
        
        if "CSP" in surveys_for_chisq:
            surv1s.extend([    'GAIA_CSP',    'GAIA_CSP',    'GAIA_CSP',    'GAIA_CSP',    'GAIA_CSP',   'GAIA_CSP',   'GAIA_CSP',   'GAIA_CSP'])
            surv2s.extend([ 'CSP', 'CSP', 'CSP', 'CSP', 'CSP','CSP','CSP','CSP'])
            filtas.extend([      'g',      'g',      'g',      'g',      'g',     'g',     'g',     'g'])
            filtbs.extend([      'r',      'i',      'i',      'r',      'i',     'i',     'i',     'i'])
            filt1s.extend([      'g',      'r',      'i',      'B',      'V',     'o',     'm',     'n'])
            filt2s.extend([      'g',      'r',      'i',      'B',      'V',     'o',     'm',     'n'])
    
        if "PS1SN" in surveys_for_chisq:
            surv1s.extend([  'GAIA_PS1SN',  'GAIA_PS1SN',  'GAIA_PS1SN',  'GAIA_PS1SN'])
            surv2s.extend([ 'PS1SN', 'PS1SN', 'PS1SN', 'PS1SN'])
            filtas.extend([    'g',    'g',    'g',    'r'])
            filtbs.extend([    'i',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'i',    'z'])
            filt2s.extend([    'g',    'r',    'i',    'z'])
    
        if "Foundation" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1',  'PS1'])
            surv2s.extend([ 'Foundation', 'Foundation', 'Foundation', 'Foundation'])
            filtas.extend([    'g',    'g',    'g',    'r'])
            filtbs.extend([    'i',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'i',    'z'])
            filt2s.extend([    'g',    'r',    'i',    'z'])
       
        if "ZTF" in surveys_for_chisq:
            surv1s.extend([  'GAIA_ZTF',  'GAIA_ZTF',  'GAIA_ZTF', 'GAIA_ZTF', 'GAIA_ZTF', 'GAIA_ZTF'])
            surv2s.extend([  'ZTF',  'ZTF',  'ZTF', 'ZTF', 'ZTF', 'ZTF'])
            filtas.extend([    'g',    'g',    'g', 'g', 'g', 'g'])
            filtbs.extend([    'i',    'i',    'i', 'i', 'i', 'i'])
            filt1s.extend([    'g',    'r',    'i', 'g', 'r', 'i'])
            filt2s.extend([    'g',    'r',    'i', 'G', 'R', 'I'])
    
        if "ZTFD" in surveys_for_chisq:
            surv1s.extend([  'PS1',  'PS1',  'PS1'])
            surv2s.extend([ 'ZTFD', 'ZTFD', 'ZTFD'])
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
    
    
        if "SDSS" in surveys_for_chisq:
            surv1s.extend([  'GAIA_SDSS',  'GAIA_SDSS',  'GAIA_SDSS',  'GAIA_SDSS'])
            surv2s.extend([ 'SDSS', 'SDSS', 'SDSS', 'SDSS'])
            filtas.extend([    'g',    'g',    'g',    'r'])
            filtbs.extend([    'i',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'i',    'z'])
            filt2s.extend([    'g',    'r',    'i',    'z'])
    
        if "SNLS" in surveys_for_chisq:
            surv1s.extend([  'GAIA_SNLS',  'GAIA_SNLS',  'GAIA_SNLS',  'GAIA_SNLS'])
            surv2s.extend([ 'SNLS', 'SNLS', 'SNLS', 'SNLS'])
            filtas.extend([    'g',    'g',    'g',    'r'])
            filtbs.extend([    'i',    'i',    'i',    'i'])
            filt1s.extend([    'g',    'r',    'i',    'z'])
            filt2s.extend([    'g',    'r',    'i',    'z'])
    
        if "CFA3S" in surveys_for_chisq:
            surv1s.extend([   'GAIA_CFA3S',   'GAIA_CFA3S',   'GAIA_CFA3S',   'GAIA_CFA3S'])
            surv2s.extend([ 'CFA3S', 'CFA3S', 'CFA3S', 'CFA3S'])
            filtas.extend([     'g',     'g',     'g',     'g'])
            filtbs.extend([     'i',     'i',     'i',     'i'])
            filt1s.extend([     'B',     'V',     'R',     'I'])
            filt2s.extend([     'B',     'V',     'R',     'I'])
    
        if "CFA3K" in surveys_for_chisq:
            surv1s.extend([   'GAIA_CFA3K',   'GAIA_CFA3K',   'GAIA_CFA3K',   'GAIA_CFA3K'])
            surv2s.extend([ 'CFA3K', 'CFA3K', 'CFA3K', 'CFA3K'])
            filtas.extend([     'g',     'g',     'g',     'g'])
            filtbs.extend([     'i',     'i',     'i',     'i'])
            filt1s.extend([     'B',     'V',     'r',     'i'])
            filt2s.extend([     'B',     'V',     'r',     'i'])        
    
        if "CFA4P1" in surveys_for_chisq:
            surv1s.extend(['GAIA_CFA4P1',       'GAIA_CFA4P1',     'GAIA_CFA4P1',    'GAIA_CFA4P1'])
            surv2s.extend(['CFA4P1', 'CFA4P1','CFA4P1','CFA4P1'])
            filtas.extend([    'g',         'g',       'g',      'g'])
            filtbs.extend([    'r',         'i',       'i',      'i'])
            filt1s.extend([    'B',         'V',       'r',      'i'])
            filt2s.extend([    'B',         'V',       'r',      'i'])
        
        if "CFA4P2" in surveys_for_chisq:
            surv1s.extend([  'GAIA_CFA4P2',       'GAIA_CFA4P2',     'GAIA_CFA4P2',    'GAIA_CFA4P2'])
            surv2s.extend(['CFA4P2', 'CFA4P2', 'CFA4P2','CFA4P2'])
            filtas.extend([    'g',         'g',       'g',      'g'])
            filtbs.extend([    'r',         'i',       'i',      'i'])
            filt1s.extend([    'B',         'V',       'r',      'i'])
            filt2s.extend([    'B',         'V',       'r',      'i'])

 

    totalchisq = 0


    weightsum = 0
    chi2v = []

#     passsurvey={surv:{name: surveydata[surv][name].values for name in list(surveydata[surv])  if surveydata[surv][name].dtype in [np.dtype(int), np.dtype(float)] } for surv in surveydata}
#     for surv in surveydata: 
#         passsurvey[surv]['standard_catagory'] = surveydata[surv]['standard_catagory'] 
#     passobsdfs={surv:{name: obsdfs[surv][name].values for name in list(obsdfs[surv]) if obsdfs[surv][name].dtype in [np.dtype(int), np.dtype(float)] } for surv in obsdfs}
    for surv1,surv2,filta,filtb,filt1,filt2 in zip(surv1s,surv2s,filtas,filtbs,filt1s,filt2s):
        allshifts=np.unique((surveydata[surv2]['shift'].values))
        for shift in allshifts:
            if DEBUG: print(surv1, filta, surv2, filtb)
            passsurvey={surv2: surveydata[surv2][surveydata[surv2]['shift']==shift] }
            if "GAIA" in surv1:
                chi2results = getchi_forone(paramsdict,passsurvey, obsdfs,surv1,surv1,surv2,filta,filtb,filt1,filt2,filtshift=shift,
                                    off1=paramsdict[surv1+'-'+filt1+'_offset'],off2=paramsdict[surv2+'-'+filt2+'_offset'],
                                        offa=0,offb=0)
            else:
                chi2results = getchi_forone(paramsdict,passsurvey, obsdfs,surv1,surv1,surv2,filta,filtb,filt1,filt2,filtshift=shift,
                                    off1=paramsdict[surv1+'-'+filt1+'_offset'],off2=paramsdict[surv2+'-'+filt2+'_offset'],
                                        offa=paramsdict[surv1+'-'+filta+'_offset'],offb=paramsdict[surv1+'-'+filtb+'_offset'])
            if doplot: plot_forone(chi2results,subscript,outputdir,tableout,biasestimates)
            chi2v.append(chi2results.chi2) #Would like to add the survey info as well
        if len(allshifts)>1: 
            print('WARNING: Multiple shifts indicated, no chi2 calculated')
        else:
            totalchisq+=chi2results.chi2
    
    if not (whitedwarf_seds is None):
    
        whitedwarfresults=calc_wd_chisq(paramsdict,whitedwarf_seds,whitedwarf_obs)
        totalchisq+=whitedwarfresults.chi2
        if doplot:
            for filt in whitedwarfresults.resids:
                plotwhitedwarfresids(filt, outputdir, whitedwarfresults,paramsdict)
            
    
    lp += lnprior(paramsdict)
    return lp -.5*totalchisq


def lnprior(paramsdict):

    priordict = { 
        'PS1-g_offset':[0,.02],
        'PS1-r_offset':[0,.02],
        'PS1-i_offset':[0,.02],
        'PS1-z_offset':[0,.02],
        'PS1SN-g_offset':[0,.01],
        'PS1SN-r_offset':[0,.01],
        'PS1SN-i_offset':[0,.01],
        'PS1SN-z_offset':[0,.01],
        'DES-g_offset':[0,.01],
        'DES-r_offset':[0,.01],
        'DES-i_offset':[0,.01],
        'DES-z_offset':[0,.01],
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
        try:
            mu = prior[0]
            sigma = prior[1]
            lp += -0.5*(paramsdict[priorparam]-mu)**2/sigma**2
        except KeyError:
            raise ValueError(f'Missed parameter {priorparam}, missing a survey')

    return lp

##Put old ugly code with plotting in here
def plot_forone(result,subscript, outputdir,tableout,biasestimates): 

    
    plt.clf()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(result.datax,result.datay,
            color='k',alpha=.3, edgecolor=None, label='Observed Mags Chisq %.2f'%(result.chi2),s=5,zorder=5)
    data_slope=result.data_popt[0] 
    data_slope_err = np.sqrt(result.data_pcov[0,0] )
    
    ax.plot(result.datax, line(np.array(result.datax),result.data_popt[0],result.data_popt[1]), c="k", lw=4, zorder=20)
    #here starts the two synthetics
    linecolors = ['goldenrod', "#0C6291"]
    for cat,popt,pcov,mc,mr,linecolor in zip(result.cats,result.synthpopts,result.synthpcovs
                                            ,result.modelcolors,result.modelress,linecolors):
        offmean = np.mean(line(result.datax,popt[0],popt[1]) - result.datay)
        offmed = np.median(line(result.datax,popt[0],popt[1]) - result.datay)
        synth_slope = popt[0]
        synth_slope_err = np.sqrt(pcov[0,0])
        diff = (data_slope-synth_slope)

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
        for speen in ['right', 'top', 'left', 'bottom']: #hehe speen
            ax.spines[speen].set_visible(False)

        ## End plot stuff
        if (biasestimates is None) or (len(biasestimates) == 0):
            preddiff,scatter=0,0
        else:
            preddiff,scatter=biasestimates[result.surv2+'-' + result.yfilt2+'-'+cat]
        tableout.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.4f\t%d\t%.3f+-%.3f\t%.3f+-%.3f\t%.3f\t%.3f\t%.1f\t%.1f\n'%(result.surv1,result.colorfilta,result.colorfiltb,result.yfilt1,result.surv2,result.yfilt2,cat,offmean,result.datax.size,data_slope,data_slope_err,synth_slope,synth_slope_err, diff,preddiff, (diff-preddiff)/scatter,result.shift))
    fname='overlay_on_obs_%s_%s-%s_%s_%s_%s_%s_%s.png'%(result.surv1,result.colorfilta,result.colorfiltb,result.yfilt1,result.surv2,result.yfilt2,'all',subscript)
    if outputdir: outpath= path.join('plots',path.join(outputdir, fname))
    else: outpath=path.join('plots',fname)
    os.makedirs(path.split(outpath)[0],exist_ok=True)
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
    parser.add_argument("--FAKES", help = msg, type=str,default=None)
    msg = "Use simbiases.txt to bias correct slopes "
    parser.add_argument("--BIASCOR", help = msg, type=bool,default=True)
    parser.set_defaults(FAKES=False)
    parser.set_defaults(BIASCOR=False)

    parser.add_argument('--speclibrary', help='Spectral library to use',type=str,default=None)
    parser.add_argument('--outputdir', help='Directory for all output',type=str,default=None)
    parser.add_argument("--target_acceptance", help = "Target acceptance rate for hamiltonian MCMC", type=float, default=-9)
    
    parser.add_argument("--n_adaptive", help = "Number of steps to adapt MCMC hyperparameters", type=int, default=2000)
    parser.add_argument("--output", help = "Path to write output chains to (.npz format)", type=str, default=None)
    msg = "Default False. Prints a nice bird :)"
    parser.add_argument("--BIRD", help = msg, action="store_true")


    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    REDO, MCMC, DEBUG, FAKES = prep_config(args)
    if args.target_acceptance != -9:
        target_acceptance = args.target_acceptance
        print(f"Using new acceptance rate of {target_acceptance}")
    if args.BIRD:
        with open("scripts/birds.txt", "rb") as f:
            for line in f:
                print(line)
        quit()
    tablefile='preprocess_dovekie.dat'
    if args.outputdir is None:
        if FAKES: 
            if FAKES.endswith('/'): FAKES=FAKES[:-1]
            tablefile=path.join(path.join('plots',f'fakes_{path.split(FAKES)[1]}'),tablefile)
            os.makedirs(path.split(tablefile)[0],exist_ok=True)
    else: 
        tablefile=path.join(args.outputdir,tablefile)
        os.makedirs(path.split(tablefile)[0],exist_ok=True)
    tableout = open(tablefile,'w')
    tableout.write('COLORSURV COLORFILT1 COLORFILT2 OFFSETFILT1 OFFSETSURV OFFSETFILT2 SPECLIB OFFSET NDATA D_SLOPE S_SLOPE DIFF PRED_DIFF SIGMA SHIFT\n')
    print('reading in survey data')

    surveys_for_chisq = config['surveys_for_dovekie']
    reference_surveys= config['reference_surveys']
    if args.output is None: outname = config['chainsfile']
    else: outname= args.output
    #surveys_for_chisq = ['PS1', 'CFA3K', 'PS1SN'] #keep this one around for quick IRSA updates!
    fixsurveynames = []

    surveydata = get_all_shifts(surveys_for_chisq, config['reference_surveys'])
    obsdfs = get_all_obsdfs(surveys_for_chisq, REDO, FAKES)
    print('got all survey data')

    if REDO:
        print("Done acquiring IRSA maps. Quitting now to avoid confusion.")
        quit()

    if whitedwarf_obs_loc:
        if ( args.FAKES ):
            whitedwarf_obs_loc=args.FAKES+'/WD_simmed.csv'
        print('Loading white dwarf data')
        whitedwarf_obs = pd.read_csv(whitedwarf_obs_loc,index_col='Object')
        whitedwarf_obs=whitedwarf_obs.rename(columns={x:x.replace('_','-') for x in list(whitedwarf_obs) if '_' in x})
        whitedwarf_seds= get_whitedwarf_synths(surveys_for_chisq)
    else:
        whitedwarf_obs = None
        whitedwarf_seds= None
    if args.BIASCOR:
        biasestimates=pd.read_csv('simbiases.txt',sep='\s+' ) 
        biasestimates={ x.FILTER+'-'+x.SPECLIB:(x.SLOPEBIAS,x.SLOPEERROR) for _,x in biasestimates.iterrows()}
        print('Bias corrections applied from simbiases.txt')
    else:
        biasestimates=None
        print("WARNING: NO BIAS CORRECTIONS APPLIED")
        
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
    
    
    full_likelihood_data= partial(full_likelihood,surveys_for_chisq, fixsurveynames,surveydata,obsdfs,reference_surveys, whitedwarf_seds=whitedwarf_seds,whitedwarf_obs= whitedwarf_obs, speclibrary=args.speclibrary)
    full_likelihood_data(pos,subscript='preprocess',doplot=True,tableout=tableout,outputdir=(args.outputdir if args.outputdir is not None else  (f'fakes_{path.split(FAKES)[1]}' if FAKES else None) ),biasestimates=biasestimates)

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
    #key = random.PRNGKey(23490268954)
    initkey, samplekey= random.split(key)
    n_samples = 5000
    theta0 = random.normal(initkey, shape=(nparams,))*0.01

    sampler = NUTS(theta0, logp=full_likelihood_data, target_acceptance=target_acceptance, M_adapt=n_burnin)
    key, samples, step_size = sampler.sample(n_samples, samplekey)
    loglikes=jax.vmap(full_likelihood_data,in_axes=0)(samples)
    if args.outputdir and not ( '/' in outname): outname= path.join(args.outputdir,outname)
    np.savez(outname,samples=samples,labels=labels,surveys_for_chisq=surveys_for_chisq)
    final=np.mean(samples,axis=0)
    if args.FAKES: sys.exit(0)
    paramsdict=unwravel_params(final,surveys_for_chisq,fixsurveynames)[0]
    whitedwarfresults=calc_wd_chisq(paramsdict,whitedwarf_seds,whitedwarf_obs)
    for filt in whitedwarfresults.resids:
        plotwhitedwarfresids(filt, '', whitedwarfresults,paramsdict)


    
    
