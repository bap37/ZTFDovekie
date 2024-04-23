import pandas as pd
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import os
from scipy.optimize import curve_fit
from glob import glob
import sys
sys.path.insert(1, 'scripts/')
from helpers import *
import time
import pickle
import emcee
from astroquery.irsa_dust import IrsaDust
import astropy.coordinates as coord
import astropy.units as u
from scipy import interpolate
import argparse

global surveys_for_chisq
global fixsurveynames
global surveydata
global obsdfs    
global obsdict
global synthdict


#search for "TODO"
isshift = False
global DEBUG
DEBUG = False

jsonload = 'FILTERDEFS.yml' #where all the important but unwieldy dictionaries live
config = load_config(jsonload)
survmap, survmap4shift, survfiltmap, obssurvmap, revobssurvmap, revobssurvmapforsnana, survcolormin, survcolormax, synth_gi_range, obsfilts, snanafilts, snanafiltsr, relativeweights, errfloors = prep_config(config)


obscolors_by_survey = {'PS1':['PS1-g','PS1-i']} #dodgy, feel like this should be tonry

filter_means = pd.read_csv('filter_means.csv') 

filter_means = filter_means.set_index(['SURVEYFILTER']).to_dict()['MEANLAMBDA ']


tableout = open('preoffsetsaper.dat','w')
tableout.write('COLORSURV COLORFILT1 COLORFILT2 OFFSETFILT1 OFFSETSURV OFFSETFILT2 SPECLIB OFFSET NDATA D_SLOPE S_SLOPE SIGMA SHIFT\n')


def get_all_shifts(surveys): #acquires all the surveys and collates them. Notably from magsaper ??
    surveydfs = {}
    for survey in surveys:
        files = glob('output_synthetic_magsaper/synth_%s_shift_*.000.txt'%survmap4shift[survey]) #originally just the 0 shift
        print(files)
        if len(files) > 1:
            print("Picking up shifts!")
            isshift = True
        #mags aper is the aperture mag from PS1, important to keep around

        dfl = []
        for f in files:
            try:
                tdf = pd.read_csv(f,delim_whitespace=True)      
                if 'PS1_' in f:
                    tdf = tdf[-1*tdf['PS1-g']+tdf['PS1-i']>.25]
                    tdf = tdf[-1*tdf['PS1-g']+tdf['PS1-i']<1.6]

                dfl.append(tdf)
            except:
                print('WARNING: Could not read in ',f) 
        df = pd.concat(dfl, axis=0, ignore_index=True)
        df = df.sort_values(by=['standard','shift'])
        if len(df) < 2:
            print("You have an empty dataframe!")
            quit()

        surveydfs[survey] = df

    for survey in surveys:
        if survey != 'PS1':
            surveydfs[survey] = pd.merge(surveydfs[survey],surveydfs['PS1'],left_on='standard',right_on='standard',suffixes=('','_b'))
    return surveydfs
    
def get_all_obsdfs(surveys, redo=False, fakes=False): #TODO - better checks on IRSA loading and such
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
                print(f"output_observed_apermags+AV/{obssurvmap[survey]}_observed.csv is missing the required IRSA dust maps. Rerun the command with additional argument --IRSA \n quitting.")
                quit()

            obsdf = obsdf[obsdf['PS1-g']-obsdf['PS1-g_AV']>14.3]
            obsdf = obsdf[obsdf['PS1-r']-obsdf['PS1-r_AV']>14.4]
            obsdf = obsdf[obsdf['PS1-i']-obsdf['PS1-i_AV']>14.6]
            obsdf = obsdf[obsdf['PS1-z']-obsdf['PS1-z_AV']>14.1]
            obsdf = obsdf[(obsdf['PS1-g']-obsdf['PS1-g_AV'])-
                          (obsdf['PS1-i']-obsdf['PS1-i_AV']) < survcolormax[survey]]
            obsdf = obsdf[(obsdf['PS1-g']-obsdf['PS1-g_AV'])-
                          (obsdf['PS1-i']-obsdf['PS1-i_AV']) > survcolormin[survey]]

            surveydfs_wext[survey] = obsdf

    return surveydfs_wext

def getchi_forone(pars,surveydata,obsdfs,colorsurvab,surv1,surv2,colorfilta,colorfiltb,yfilt1,yfilt2,
                  shifta=0,shiftb=0,shift1=0,shift2=0,off1=0,off2=0,offa=0,offb=0,
                  calspecslope=0,calspecmeanlambda=4383.15,ngslslope=0,ngslmeanlambda=5507.09,
                  obsdict=None,synthdict=None,doplot=False,subscript='',first=False): #where the magic happens I suppose

    ssurv2 = survmap[surv2]
    df2x = surveydata[surv2]

    for shift in np.unique(df2x['shift'].values):
        df2 = df2x.loc[df2x['shift'] == shift]

        chi2 = 0
        npoints = 0
    
        #changed these back to dashes
        longfilta = survfiltmap[colorsurvab]+'-'+colorfilta
        longfiltb = survfiltmap[colorsurvab]+'-'+colorfiltb
        longfilt1 = survfiltmap[surv1]+'-'+yfilt1
        longfilt2 = survfiltmap[surv2]+'-'+yfilt2

    
        obslongfilta = obssurvmap[colorsurvab]+'-'+colorfilta
        obslongfiltb = obssurvmap[colorsurvab]+'-'+colorfiltb
        obslongfilt1 = obssurvmap[surv1]+'-'+yfilt1 
        if ('KAIT' in surv2.upper()) | ('NICKEL' in surv2.upper()):
            obslongfilt2 = obssurvmap[surv2]+'-'+snanafilts[surv2][yfilt2]
        elif ('CSP' in surv2.upper()):
            obslongfilt2 = obssurvmap[surv2]+'-'+yfilt2.replace('o','V').replace('m','V').replace('n','V')
        elif ('ASASSN' in surv2.upper()):
            obslongfilt2 = obssurvmap[surv2]+'-'+snanafilts[surv2][yfilt2]
        else:
            obslongfilt2 = obssurvmap[surv2]+'-'+yfilt2 #

        if first:
            obsdf = obsdfs[surv2] #grabs the observed points from the relevant survey
            if DEBUG: print(obsdf.columns, surv2)
            yr=obsdf[obslongfilt1]-obsdf[obslongfilt2] #observed filter1 - observed filter 2 
            wwyr = np.abs(yr)<1 #only uses things lower than 1
            datacolor = (obsdf[obslongfilta][wwyr]-obsdf[obslongfilta+"_AV"][wwyr])-(obsdf[obslongfiltb][wwyr]-obsdf[obslongfiltb+"_AV"][wwyr])
            datares = obsdf[obslongfilt1][wwyr]-obsdf[obslongfilt1+'_AV'][wwyr]-(obsdf[obslongfilt2][wwyr]-obsdf[obslongfilt2+'_AV'][wwyr])
        
            obsdict[surv2+obslongfilt1+obslongfilt2] = {}
            obsdict[surv2+obslongfilt1+obslongfilt2]['datacolor'] = datacolor
            obsdict[surv2+obslongfilt1+obslongfilt2]['datares'] =datares
            xd,yd,sigmadata,yresd,poptd,pcovd = itersigmacut_linefit(datacolor.astype('float'),
                                                                 datares.astype('float'),
                                                                 niter=2,nsigma=3)    
            obsdict[surv2+obslongfilt1+obslongfilt2]['sigmadata'] = sigmadata

            synthdict[surv2+obslongfilt1+obslongfilt2] = {}
            for cat in np.unique(df2['standard_catagory']):
                synthdict[surv2+obslongfilt1+obslongfilt2][cat] = {}
                if DEBUG: print(df2.columns, surv2, surv1) 
                ww = (df2['standard_catagory']==cat) & \
                   (~np.isnan(df2[longfilt2].astype('float'))) & \
                   (~np.isnan(df2[longfilt1].astype('float')))
                modelfilta = df2[longfilta][ww]
                modelfiltb = df2[longfiltb][ww]
                modelfilt1 = df2[longfilt1][ww]
                modelfilt2 = df2[longfilt2][ww]
                modelcolor = -1*df2[longfilta][ww]+offa+df2[longfiltb][ww]-offb
                modelres = -1*df2[longfilt1][ww]+off1+df2[longfilt2][ww]-off2
                ww2 = (modelcolor > synth_gi_range[cat][0]) & (modelcolor < synth_gi_range[cat][1])

                synthdict[surv2+obslongfilt1+obslongfilt2][cat]['modelfilta'] = modelfilta[ww2].astype('float')
                synthdict[surv2+obslongfilt1+obslongfilt2][cat]['modelfiltb'] = modelfiltb[ww2].astype('float')
                synthdict[surv2+obslongfilt1+obslongfilt2][cat]['modelfilt1'] = modelfilt1[ww2].astype('float')
                synthdict[surv2+obslongfilt1+obslongfilt2][cat]['modelfilt2'] = modelfilt2[ww2].astype('float')
        else:
            datacolor = obsdict[surv2+obslongfilt1+obslongfilt2]['datacolor']
            datares = obsdict[surv2+obslongfilt1+obslongfilt2]['datares']
            sigmadata = obsdict[surv2+obslongfilt1+obslongfilt2]['sigmadata']

        #End of "if first" statement. 
        cats, popts, pcovs, modelcolors, modelress, dataress, datacolors, modellines, lines, resres, reserr, xds, ms, yds, xdsc, ydsc =  ([] for i in range(16))

        for cat in np.unique(df2['standard_catagory']):
            modelcolor = -1*synthdict[surv2+obslongfilt1+obslongfilt2][cat]['modelfilta']+offa+\
                        synthdict[surv2+obslongfilt1+obslongfilt2][cat]['modelfiltb']-offb
            modelres = -1*synthdict[surv2+obslongfilt1+obslongfilt2][cat]['modelfilt1']+off1+\
                        synthdict[surv2+obslongfilt1+obslongfilt2][cat]['modelfilt2']-off2-0.0065
            x,y,sigmamodel,yres,popt,pcov = itersigmacut_linefit(modelcolor,modelres,niter=1,nsigma=3)

            if doplot:
                modelress.append(modelres.astype('float'))
                modelcolors.append(modelcolor.astype('float'))
                xds.extend(xd)
                yds.extend(yd)
                xdsc.append(xd)
                ydsc.append(yd)
                cats.append(cat)
                popts.append(popt)
                pcovs.append(pcov)
        
        
            reserr = datares*0+sigmadata/np.sqrt(len(datares)) #uhhh why is datares multiplied by 0 
            dms = datares - line(datacolor,popt[0],popt[1])
            chires = np.mean(dms)
            chireserrsq = (sigmadata/np.sqrt(len(datares)))**2+errfloors[surv2]**2
            chi2 += chires**2/chireserrsq/2

        #print(chi2)

        if doplot:
            plt.clf()
            plt.scatter(xds,yds,
                    color='k',alpha=.5,label='Observed Mags Chisq %.2f'%(chi2),s=5,zorder=9999)
            _,_,sigmad,_,data_popt,data_pcov = itersigmacut_linefit(np.array(xds),np.array(yds),niter=1,nsigma=5)
            data_slope=data_popt[0] ; data_slope_err = (data_pcov[0,0]**2+sigmad**2)**.5
            ndata = len(datares)

            for cat,popt,pcov,mc,mr in zip(cats,popts,pcovs,modelcolors,modelress):
                offmean = np.mean(line(xd,popt[0],popt[1]) - yd)
                offmed = np.median(line(xd,popt[0],popt[1]) - yd)
                synth_slope = popt[0]
                synth_slope_err = pcov[0,0]
                sigma = (data_slope-synth_slope)/np.sqrt(data_slope_err**2+synth_slope_err**2)
                plt.plot(xd,line(xd,popt[0],popt[1]),label='Synthetic Pred: %s\nOffMean: %.3f\nOffMedian: %.3f\n'%(cat,offmean,offmed))
                plt.scatter(mc,mr,alpha=.9,s=4)
                tableout.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.4f\t%d\t%.3f+-%.3f\t%.3f+-%.3f\t%.1f\t%.1f\n'%(surv1,colorfilta,colorfiltb,yfilt1,surv2,yfilt2,cat,offmean,ndata,data_slope,data_slope_err,synth_slope,synth_slope_err,sigma,shift))  
                plt.legend()
                plt.xlabel('%s - %s'%(obslongfilta,obslongfiltb))
                plt.ylabel('%s - %s'%(obslongfilt1,obslongfilt2))
                plt.legend()
                plt.savefig('plots/synthetic/overlay_on_obs_%s_%s-%s_%s_%s_%s_%s_%s.png'%(surv1,colorfilta,colorfiltb,yfilt1,surv2,yfilt2,'all',subscript))
                print('upload plots/synthetic/overlay_on_obs_%s_%s-%s_%s_%s_%s_%s_%s.png'%(surv1,colorfilta,colorfiltb,yfilt1,surv2,yfilt2,'all',subscript))

    return chi2,npoints,cats,popts,pcovs,obsdict,synthdict,chires

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
            if (params[i]<-1.5) | (params[i]>1.5): outofbounds=True
            paramsnames.append(survey+'-'+filt+'_offset')
            i+=1

    for survey in fixsurveynames:
        filts = obsfilts[survmap[survey]]
        for ofilt in filts:
            filt = snanafiltsr[survey][ofilt]
            paramsdict[survey+'-'+filt+'_offset'] = 0
            paramsnames.append(survey+'-'+filt+'_offset')

    return paramsdict,outofbounds,paramsnames

def remote_full_likelihood(params,surveys_for_chisqin=None,fixsurveynamesin=None,surveydatain=None,obsdfin=None,doplot=False,subscript='',debug=False,first=False):
    global surveys_for_chisq
    surveys_for_chisq = surveys_for_chisqin
    global fixsurveynames
    fixsurveynames = fixsurveynamesin
    global surveydata
    surveydata = surveydatain
    global obsdfs
    obsdfs = obsdfin
    chi2,chi2v = full_likelihood(params,doplot=doplot,subscript=subscript,first=first, remote=True)
    return chi2,chi2v

def full_likelihood(params,doplot=False,subscript='',debug=False,first=False, remote=False):

    if first:
        global obsdict
        global synthdict
        obsdict = {}
        synthdict = {}

    chisqtot=0

    paramsdict,outofbounds,paramsnames = unwravel_params(params,surveys_for_chisq,fixsurveynames)

    if outofbounds:
        return -np.inf

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

    
    offsetsdict = {}
    shiftssdict = {}

    totalchisq = 0

    weightsum = 0
    chi2v = []
    for surv1,surv2,filta,filtb,filt1,filt2 in zip(surv1s,surv2s,filtas,filtbs,filt1s,filt2s):
        chi2,npoints,cats,popts,pcovs,obsdict,synthdict,off = getchi_forone(paramsdict,surveydata,obsdfs,surv1,surv1,surv2,filta,filtb,filt1,filt2,
                                                      off1=paramsdict[surv1+'-'+filt1+'_offset'],
                                                      off2=paramsdict[surv2+'-'+filt2+'_offset'],
                                                      offa=paramsdict[surv1+'-'+filta+'_offset'],
                                                      offb=paramsdict[surv1+'-'+filtb+'_offset'],
                                                      doplot=doplot,subscript=subscript,first=first,
                                                      obsdict=obsdict,synthdict=synthdict)
        chi2v.append(chi2) #Would like to add the survey info as well
        totalchisq+=chi2
        if first: 
            paramsdict[surv2+'-'+filt2+'_preoffset'] = off
            paramsdict[surv1+'-'+filt1+'_preoffset'] = 0
    lp = lnprior(paramsdict)
    if doplot:
        print('Likelihood %.2f -chisq/2 %.2f lp %.2f'%(lp -.5*totalchisq,-.5*totalchisq,lp))
    if remote:
        return lp -.5*totalchisq, chi2v
    if first:
        return paramsdict, obsdict, synthdict
        
    return lp -.5*totalchisq

def lnprior(paramsdict):

    priordict = { 
        'PS1-g_offset':[0,.02],
        'PS1-r_offset':[0,.02],
        'PS1-i_offset':[0,.02],
        'PS1-z_offset':[0,.02],

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


def prep_config(args):

    REDO = args.IRSA #yes I know there's a discrepancy in naming here
    MCMC = args.MCMC
    DEBUG = args.DEBUG

    return REDO, MCMC, DEBUG

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

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    REDO, MCMC, DEBUG = prep_config(args)

    print('reading in survey data')

    surveys_for_chisq = ['PS1', 'PS1SN', 'DES', 'Foundation', 'SNLS', 'SDSS', 'CSP', 'CFA3K', 'CFA3S', 'ZTF'] 
    #  ^ This controls both the surveys that will be used in the mcmc and also those that you will be grabbing IRSA from. When in IRSA mode, be very careful! 
    #surveys_for_chisq = ['PS1', 'CFA3K', 'PS1SN'] #keep this one around for quick IRSA updates!
    fixsurveynames = []

    surveydata = get_all_shifts(surveys_for_chisq)
    obsdfs = get_all_obsdfs(surveys_for_chisq, REDO)
    print('got all survey data')

    if REDO:
        print("Done acquiring IRSA maps. Quitting now to avoid confusion.")
        quit()

    #surv1='CSP'
    #surv2='SDSS'  

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

    #### TESTING ###############
    
    pos = np.array(pos)
    pdict,obsdict,synthdict = full_likelihood(pos,subscript='beforeaper_v8_150',doplot=True,first=True)
    prepos = []

    for s in surveys_for_chisq:
        ofs = obsfilts[survmap[s]]
        for of in ofs:
            prepos.append(pdict[s+'-'+snanafiltsr[s][of]+'_preoffset'])
    prepos = np.array(prepos)
    offsets,obsdict,synthdict = full_likelihood(-1*prepos,subscript='preoffsetsaper_v8_150',doplot=True,first=True)
    
    ###########################

    pos = np.random.randn(2*nparams, nparams)/100
    nwalkers, ndim = pos.shape

    _,_,labels=unwravel_params(pos[0,:],surveys_for_chisq,fixsurveynames)

    tableout.close()

    if MCMC == False:
        print("Not running the full MCMC, quitting now.")
        quit()

    os.environ["OMP_NUM_THREADS"] = "1"

    if isshift:
        print("You are currently grabbing all the shifted values, quitting!")
        quit()

    from multiprocessing import Pool

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, full_likelihood, pool=pool)
        #start = time.time()
        for i in range(1000):
            sampler.run_mcmc(pos, 100, progress=True)
            samples = sampler.get_chain()
            pos = samples[-1,:,:]
            np.savez('TEST.npz',samples=samples,labels=labels,surveys_for_chisq=surveys_for_chisq)
