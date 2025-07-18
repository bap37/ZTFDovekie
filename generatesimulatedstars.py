#!/usr/bin/env python

import shutil
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import csv
import os
import json
import jax
from jax import numpy as jnp, scipy as jsci
from iminuit import Minuit
from functools import reduce, partial
from scripts.helpers import load_config
import argparse
import pandas as pd
from dovekie import get_whitedwarf_synths
jsonload = 'DOVEKIE_DEFS.yml' #where all the important but unwieldy dictionaries live
config = load_config(jsonload)


def loadsynthphot(fname):
    result=np.genfromtxt(fname,names=True,dtype=None,encoding='utf-8',delimiter=' ')
    for key,type in result.dtype.descr:
        if type=='<f8':
            result[key]*=1
    return result

def getdata(survname):
    if 'ZTF' in survname:
        isdouble=survname[-1]=='D'
        survname='ZTF'
    synth=loadsynthphot(f'output_synthetic_magsaper/synth_{survname}_shift_0.000.txt')
    obs=np.genfromtxt(f'output_observed_apermags+AV/{survname}_observed.csv',names=True,dtype=None,encoding='utf-8',delimiter=',')
    for name in obs.dtype.names:
        if '_AV' in name or name in ['survey','RA','DEC']:
            pass
        else:
            try:
                obs[name]=obs[name]- obs[name+'_AV']
            except ValueError:
                pass
    if survname=='ZTF':
        if isdouble: 
            obscut=np.all([obs[x]<80 for x in ['ZTFG','ZTFR','ZTFI']],axis=0)#
        else:
            obscut= np.all([obs[x]<80 for x in ['ZTFg','ZTFr','ZTFi']],axis=0)#
        obs=obs[obscut]
        cutfilts=(['ZTF'+(x if isdouble else x.upper()) for x in 'gri'])
        obs=obs[[x for x in obs.dtype.names if x not in cutfilts]]
        obs.dtype.names=[((x[:-1]+('D' if isdouble else 'S')+x[-1].lower() )if ((len(x)==4) and (x[:3]=='ZTF') ) else x)for x in obs.dtype.names]
        synth=synth[[x for x in synth.dtype.names if x not in cutfilts]]
        synth.dtype.names=[((x[:-1]+('D' if isdouble else 'S')+x[-1].lower() )if ((len(x)==4) and (x[:3]=='ZTF') ) else x)for x in synth.dtype.names]
    return synth,obs


@jax.jit
def unpack(data,colorinds,pars):
    nfilts,nstars=data.shape
    
    mags=pars[:nstars]; pars=pars[nstars:]
    
    colors=pars[:nstars]; pars=pars[nstars:]
    
    slopes=pars[:nfilts]; pars=pars[nfilts:]
    
    noise=pars[:nfilts]; pars=pars[nfilts:]
    
    intrinsiccolor=pars[:nfilts]; pars=pars[nfilts:]
    
    fout=pars[0]
    
    intrinsiccolor=intrinsiccolor.at[colorinds[0]].set(0)
    slopes=slopes.at[jnp.array(colorinds)].set([0,-1])
    noise=jnp.exp(noise)
    fout=jax.nn.sigmoid(fout)*.2
    return mags,colors, slopes,noise,intrinsiccolor,fout


def linearfitwithoutliers(data,colorinds,outlierwidth,pars):
    mags,colors, slopes,noise,intrinsiccolor,fout=unpack(data,colorinds,pars)
    modelled=mags+colors[np.newaxis,:]*slopes[:,np.newaxis]+intrinsiccolor[:,np.newaxis]
    loglikes=jnp.vectorize(jsci.stats.norm.logpdf)(data,modelled,jnp.tile(noise,(data.shape[1],1)).T).sum(axis=0)
    outlikes=jnp.vectorize(jsci.stats.norm.logpdf)(data,modelled,jnp.tile(outlierwidth,(data.shape))).sum(axis=0)
    return -jsci.special.logsumexp(jnp.stack([jnp.log(fout) + outlikes , jnp.log(1-fout) + loglikes] ),axis=0).sum()


def linearfit(data,colorinds,pars):
    mags,colors, slopes,noise,intrinsiccolor,fout=unpack(data,colorinds,pars)
    modelled=mags+colors[np.newaxis,:]*slopes[:,np.newaxis]+intrinsiccolor[:,np.newaxis]
    loglikes=jnp.vectorize(jsci.stats.norm.logpdf)(data,modelled,jnp.tile(noise,(data.shape[1],1)).T).sum(axis=0)
    return -loglikes.sum()

grad= jax.jit(jax.grad(linearfit,argnums=2),static_argnums=1)
linearfit=jax.jit(linearfit,static_argnums=1)

gradoutliers= jax.jit(jax.grad(linearfitwithoutliers,argnums=2),static_argnums=1)
linearfitwithoutliers=jax.jit(linearfitwithoutliers,static_argnums=1)


def decomposedata(data,colorinds,outlierwidth=None,maxiter=100000,init=None):
    #Decompose each star into a sum of four components; a mean/intrinsic color, a color described by fixed linear relations in each band, a magnitude, and noise
    assert(np.isnan(data).sum()==0)
    nfilts,nstars=data.shape
    magind,colorind=colorinds
    if init is None:
        #Initialize color and magnitude for each star based on the observed value
        maginit=data[magind]  #+ np.random.normal(scale =1e-3, size=data.shape[1])
        colorinit=(data[magind]-data[colorind]) #+ np.random.normal(scale =1e-3, size=data.shape[1])
        #Initialize slopes based on a SVD
        slopesinit=np.linalg.svd(data)[0][:,1]
        slopesinit=slopesinit-slopesinit[magind]
        slopesinit/=-slopesinit[colorind]
        #Guess the noise
        noiseinit= np.tile(np.log(0.02),nfilts)
        meancolorinit=np.mean( (data -data[magind,:][np.newaxis,:])  ,axis=1)
        fout=-3
        if outlierwidth is None: 
            fout=-100
    
        parsinit=np.concatenate( [maginit,colorinit,slopesinit,noiseinit,meancolorinit,[fout] 
        ])
    else: parsinit=init
    #Minuits function call limit wasn't working, so I wrote my own
    class counter:
        def __init__(self):
            self.steps=0
            self.vals=0
            self.minval=np.inf
    tracker=counter()
    
    gradpartial=partial(grad,data,colorinds)
    def passthrough(x):
        result=linearfit(data,colorinds,x)
        tracker.steps+=1
        if tracker.minval >result : tracker.vals=x
        if tracker.steps > maxiter: raise RuntimeError()
        return result
    
    
    isnoise=np.zeros(parsinit.size,dtype=bool)
    isnoise[2*nstars+nfilts:2*nstars+2*nfilts]=True
    isoutlier=np.zeros(parsinit.size,dtype=bool)
    isoutlier[-1]=True
    
    #Fit all parameters other than noise first
    optimizer=Minuit( passthrough, parsinit,grad=gradpartial  )
    optimizer.fixed=isnoise | isoutlier
    
    try:
        result=optimizer.migrad()
        vals=np.array(result.values)
    except RuntimeError as e:
        vals=tracker.vals
    #Fit noise
    tracker.steps=0
    optimizer=Minuit( passthrough, vals,grad= gradpartial )
    optimizer.fixed=~isnoise
    try:
        result=optimizer.migrad()
        vals=np.array(result.values)
    except RuntimeError as e:
        vals=tracker.vals
    if outlierwidth is None:
        pass
    else: 
        if outlierwidth is None: gradpartial=partial(gradoutlier,data, outlierwidth,colorinds)
        def passthrough(x):
            result=linearfitwithoutliers(data,colorinds,outlierwidth,x)
            tracker.steps+=1
            if tracker.minval >result : tracker.vals=x
            if tracker.steps > maxiter: raise RuntimeError()
            return result
        tracker.steps=0
        optimizer=Minuit( passthrough, vals,grad= gradpartial )
        optimizer.fixed=~isoutlier
        try:
            result=optimizer.migrad()
            vals=np.array(result.values)
        except RuntimeError as e:
            vals=tracker.vals
        tracker.steps=0
        optimizer=Minuit( passthrough, vals,grad= gradpartial )
        optimizer.fixed=~ (isoutlier |  isnoise)
        try:
            result=optimizer.migrad()
            vals=np.array(result.values)
        except RuntimeError as e:
            vals=tracker.vals

    return unpack(data,colorinds,np.array(vals))
    



class survey:
    
    def __init__(self,survname, magdist, colordist, magcolorinds, filtnames,obs, survsynth,ps1synth,survoffsets,ps1offsets,outlierwidth,isps1survey=False):
        """
        Initialize an object to produce simulated data from a given survey.
    
        Parameters:
            survname (str): survey name
            magdist (array_like): Magnitude distribution.
            colordist (array_like): Color distribution.
            magcolorinds (tuple): Indices for magnitude and color.
            filtnames (list): List of filter names.
            obs (dict): Observed data.
            survsynth (dict): Synthetic data for survey.
            ps1synth (dict): Synthetic data for PS1.
            survoffsets (array_like): Offsets for survey.
            ps1offsets (array_like): Offsets for PS1.
            isps1survey (bool, optional): Flag indicating if it's a PS1 survey. Defaults to False.
        """
        self.name=survname
        self.magind,self.colorind=magcolorinds
        assert(self.magind!=self.colorind)
        self.nobs=obs.size
        self.magdist=magdist
        self.colordist=colordist
        self.filtnames= filtnames
        self.isps1survey= isps1survey
        self.outlierwidth=outlierwidth
        #Determine excess variance of observed populations
        data=np.array([obs[x] for x in filtnames]+ [obs['PS1'+x] for x in 'griz'])
        #Use factor analysis on observed data
        _,_,self.slopes_data,noise,self.intrinsic_data,fout=decomposedata(data,magcolorinds,outlierwidth)
        self.variance=noise**2
        self.fout=fout
        #If it's a PS1 survey, the synthetic photometry will be identical to PS1, so don't regress on identical data
        if self.isps1survey: 
            mags=np.array([survsynth[x] for x in filtnames])
        else:
            mags=np.array([survsynth[x] for x in filtnames]+ [ps1synth['PS1'+x] for x in 'griz'])
        #Use factor analysis to reduce the dimensionality of the synthetic photometry
        _,_,self.slopes,_,self.intrinsic,_=decomposedata(mags,magcolorinds)
        
        self.offsets=np.concatenate([survoffsets,ps1offsets] )
        
    def genstar(self,size=1,addvariance=True):
        """
        Generate synthetic stars.
    
        Parameters:
            size (int, optional): Number of stars to generate. Defaults to 1.
    
        Returns:
            tuple or np.ndarray: Tuple of magnitudes if size is 1, else an array of magnitudes.

        """

        if size==1:
            def genonesetmag():
                #Draw magnitude and color
                mag=self.magdist()
                color=self.colordist()
                #Generate  magnitudes from factor analysis
                mags= self.slopes*color + self.intrinsic + mag
                if self.isps1survey:
                    mags=np.concatenate((mags,mags))
                #Add noise
                if addvariance: 
                    if np.random.random()< self.fout:
                        mags+=np.random.normal(0,self.outlierwidth,size=self.variance.size)
                    else:
                        mags+=(np.random.normal(0,np.sqrt(self.variance)))
                #Add zero-point offsets
                mags+=self.offsets
                return tuple(mags)
            mags=genonesetmag()
            nfilt=len(self.filtnames)
            
            minmag= np.array([14.3,14.4,14.6,14.1])
            
            while not ((.25<( mags[nfilt]-mags[nfilt+2]) < 1.) and (np.array(mags[nfilt:])> minmag).all() and (config['survcolormin'][self.name] <( mags[nfilt]-mags[nfilt+2]) < config['survcolormax'][self.name])):
                mags=genonesetmag()
            return mags
        else:
            return np.array([(self.genstar(1,addvariance)) for i in range(size)], 
                                         dtype=list(zip(   list(self.filtnames)+['PS1'+x for x in 'griz'],
                                         [float]*(len(self.filtnames)+4))))

    def showcolordists(self,obs):
        simdata=self.genstar(obs.size)
        
        fig,axes=plt.subplots(2,1)
        handles=[]
        for data,color,label in [(simdata,'k','simulated'),(obs,'b','observed')]:
            secondcolor=1 if len(self.filtnames)<4 else 3
            handles+=[axes[0].plot(data[self.filtnames[0]]-data[self.filtnames[1]],data[self.filtnames[2]]-data[self.filtnames[secondcolor]] ,color+'.',alpha=.3,markersize=3,label=label)]
            axes[1].plot(data[self.filtnames[self.magind]],data[self.filtnames[self.magind]]-data[self.filtnames[self.colorind]],color+'.',alpha=.3,markersize=3)
        axes[1].set_ylabel(f'{self.filtnames[self.magind][-1]}-{self.filtnames[self.colorind][-1]}' )
        axes[1].set_xlabel(f'{self.filtnames[self.magind][-1]}' )
        axes[0].set_xlabel(f'{self.filtnames[self.magind][-1]}-{self.filtnames[self.colorind][-1]}' )
        axes[0].set_ylabel(f'{self.filtnames[2][-1]}-{self.filtnames[secondcolor][-1]}' )
        plt.legend(handles,labels=['simulated','observed'])



def generatesurveyoffsets():
    ps1offsets=np.random.normal(0,0.01,4)
    survoffsets={'PS1':ps1offsets }

    name='SNLS'
    survoffsets[name]= np.random.normal(0,0.01,4)
    
    name='SDSS'
    survoffsets[name]= np.random.normal(0,0.01,4)
    
    name='CFA3K'
    filts=[name+x for x in 'BVri']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))
    
    name='CFA3S'
    filts=[name+x for x in 'BVRI']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))
    
    name='CSP'
    filts=[name+x for x in 'BVgri']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))
    
    name='Foundation'
    filts=[name+x for x in 'griz']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))
    
    name='PS1SN'
    filts=[name+x for x in 'griz']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))
    
    name='DES'
    filts=[name+x for x in 'griz']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))

    name='D3YR'
    filts=[name+x for x in 'griz']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))


    name='ZTFS'
    filts=[name+x for x in 'gri']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))
    
    name='ZTFD'
    filts=[name+x for x in 'gri']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))

    name='ASASSN1'
    filts=[name+x for x in 'BVri']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))

    name='ASASSN2'
    filts=[name+x for x in 'BVi']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))

    name='SWIFT'
    filts=[name+x for x in 'BV']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))

    name='KAIT1MO'
    filts=[name+x for x in 'BVRI']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))

    name='KAIT2MO'
    filts=[name+x for x in 'BVRI']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))

    name='KAIT3MO'
    filts=[name+x for x in 'BVRI']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))

    name='KAIT4MO'
    filts=[name+x for x in 'BVRI']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))

    name='NICKEL1MO'
    filts=[name+x for x in 'BVRI']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))

    name='NICKEL2MO'
    filts=[name+x for x in 'BVRI']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))

    name='KAIT3'
    filts=[name+x for x in 'BVRI']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))

    name='KAIT4'
    filts=[name+x for x in 'BVRI']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))

    name='NICKEL1'
    filts=[name+x for x in 'BVRI']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))

    name='NICKEL2'
    filts=[name+x for x in 'BVRI']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))

    name='CFA4P1'
    filts=[name+x for x in 'BVri']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))

    name='CFA4P2'
    filts=[name+x for x in 'BVri']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))


    return survoffsets

__surveycache__= {}
def generatesurvey(name,survoffsets,forcereload=False,speclibrary='calspec23'):
    if name in __surveycache__ and not forcereload:
        print(f'Retrieving {name} from cache', flush=True)
        return __surveycache__[name]
    ps1synth=loadsynthphot('output_synthetic_magsaper/synth_PS1_shift_0.000.txt')
    cut=(ps1synth['standard_catagory']==speclibrary)& ( ps1synth['PS1g']-ps1synth['PS1r'] > 0) &(ps1synth['PS1g']-ps1synth['PS1r'] <.8)
    print(f'Preparing {name}', flush=True)
    
    if name=='SNLS':
        synth,obs=getdata(name)
        surv=survey(name,lambda obs=obs: stats.gaussian_kde(obs['SNLSg']).resample(1)[0][0],stats.exponnorm(.1,loc=.2,scale=.3).rvs,
              (0,1),['SNLS'+x for x in 'griz'],obs, synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'], .3 )
        return surv
    
    elif name=='SDSS':
        synth,obs=getdata(name)
        surv=survey(name,lambda obs=obs: stats.gaussian_kde(obs['SDSSg']).resample(1)[0][0],  stats.exponnorm(.1,loc=.2,scale=.3).rvs ,
              (0,1),[name+x for x in 'griz'],obs, synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'] , .2)
    
    elif name=='CFA3K':
        filts=[name+x for x in 'BVri']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['CFA3Kr']-obs['CFA3Ki'] > -.5) & (obs['CFA3Kr']-obs['CFA3Ki'] < 20)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['CFA3KB']).resample(1)[0][0],  stats.exponnorm(.001,loc=.2,scale=.3).rvs ,
              (1,2),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'] ,.15 )

    elif name=='CFA3S':
        filts=[name+x for x in 'BVRI']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['CFA3SR']-obs['CFA3SI'] > -.2)& (obs['CFA3SB']-obs['CFA3SV'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['CFA3SB']).resample(1)[0][0],  stats.exponnorm(1e-4,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.4 )
    
    elif name=='CSP':
        synth,obs=getdata(name)
        synth=np.array(list(synth), dtype=[(x.replace('CSP_TAMU',name),synth.dtype.fields[x][0]) for x in synth.dtype.fields])
        filts=[name+x for x in 'BVgri']
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )#& (obs['CFA3SR']-obs['CFA3SI'] > -.2)& (obs['CFA3SB']-obs['CFA3SV'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['CSPB']).resample(1)[0][0],  stats.exponnorm(.1,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'] ,.2)
    
    elif name=='DES':
        filts=[name+x for x in 'griz']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['DESg']-obs['DESr'] > .2)& (obs['DESg']-obs['DESr'] <1)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['DESg']).resample(1)[0][0],  stats.exponnorm(1e-2,loc=.2,scale=.3).rvs,    
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.1 )

    elif name=='D3YR':
        filts=[name+x for x in 'griz']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['D3YRg']-obs['D3YRr'] > .2)& (obs['D3YRg']-obs['D3YRr'] <1)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['D3YRg']).resample(1)[0][0],  stats.exponnorm(1e-2,loc=.2,scale=.3).rvs,    
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.1 )

    
    elif name=='Foundation':
        filts=[name+x for x in 'griz']
        synth,obs=getdata(name)
        nans=(~reduce(lambda x,y: x|y,[np.isnan(synth[x]) for x in filts],False))
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )#& (obs['Foundationg']-obs['DESr'] > .2)& (obs['DESg']-obs['DESr'] <1)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['Foundationg']).resample(1)[0][0],  stats.exponnorm(1e-2,loc=.2,scale=.3).rvs,      
              (0,1),filts,obs[obscut], synth[cut&nans],ps1synth[cut&nans], survoffsets[name],survoffsets['PS1'],.2 ,True)
    
    elif name=='PS1SN':
        filts=[name+x for x in 'griz']
        synth,obs=getdata(name)
        synth=np.array(list(synth), dtype=[(x.replace('Foundation',name),synth.dtype.fields[x][0]) for x in synth.dtype.fields])
        nans=(~reduce(lambda x,y: x|y,[np.isnan(synth[x]) for x in filts],False))
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )#& (obs['Foundationg']-obs['DESr'] > .2)& (obs['DESg']-obs['DESr'] <1)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['PS1SNg']).resample(1)[0][0],  stats.exponnorm(1e-2,loc=.2,scale=.3).rvs ,  
              (0,1),filts,obs[obscut], synth[cut&nans],ps1synth[cut&nans], survoffsets[name],survoffsets['PS1'] ,.2,True)
    
    
    elif name=='ZTFD':
        filts=[name+x for x in 'gri']
        synth,obs=getdata(name)
        nans=(~reduce(lambda x,y: x|y,[np.isnan(synth[x]) for x in filts],False))
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )#& (obs['Foundationg']-obs['DESr'] > .2)& (obs['DESg']-obs['DESr'] <1)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['ZTFDg']).resample(1)[0][0],  stats.uniform(.3,.5).rvs ,   
              (0,1),filts,obs[obscut], synth[cut&nans],ps1synth[cut&nans], survoffsets[name],survoffsets['PS1'],.2 )
        
    elif name=='ZTFS':
        filts=[name+x for x in 'gri']
        synth,obs=getdata(name)
        nans=(~reduce(lambda x,y: x|y,[np.isnan(synth[x]) for x in filts],False))
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )#& (obs['Foundationg']-obs['DESr'] > .2)& (obs['DESg']-obs['DESr'] <1)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['ZTFSg']).resample(1)[0][0],  stats.uniform(.3,.5).rvs ,   
              (0,1),filts,obs[obscut], synth[cut&nans],ps1synth[cut&nans], survoffsets[name],survoffsets['PS1'] ,.2)

    elif name=='SWIFT':
        filts=[name+x for x in 'BV']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['SWIFTB']-obs['SWIFTV'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['SWIFTB']).resample(1)[0][0],  stats.exponnorm(1e-4,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.4 )

    elif name=='KAIT1MO':
        filts=[name+x for x in 'BVRI']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['KAIT1MOR']-obs['KAIT1MOI'] > -.2)& (obs['KAIT1MOB']-obs['KAIT1MOV'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['KAIT1MOB']).resample(1)[0][0],  stats.exponnorm(1e-4,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.4 )

    elif name=='KAIT2MO':
        filts=[name+x for x in 'BVRI']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['KAIT2MOR']-obs['KAIT2MOI'] > -.2)& (obs['KAIT2MOB']-obs['KAIT2MOV'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['KAIT2MOB']).resample(1)[0][0],  stats.exponnorm(1e-4,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.4 )

    elif name=='KAIT3MO':
        filts=[name+x for x in 'BVRI']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['KAIT3MOR']-obs['KAIT3MOI'] > -.2)& (obs['KAIT3MOB']-obs['KAIT3MOV'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['KAIT3MOB']).resample(1)[0][0],  stats.exponnorm(1e-4,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.4 )

    elif name=='KAIT4MO':
        filts=[name+x for x in 'BVRI']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['KAIT4MOR']-obs['KAIT4MOI'] > -.2)& (obs['KAIT4MOB']-obs['KAIT4MOV'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['KAIT4MOB']).resample(1)[0][0],  stats.exponnorm(1e-4,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.4 )

    elif name=='NICKEL1MO':
        filts=[name+x for x in 'BVRI']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['NICKEL1MOR']-obs['NICKEL1MOI'] > -.2)& (obs['NICKEL1MOB']-obs['NICKEL1MOV'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['NICKEL1MOB']).resample(1)[0][0],  stats.exponnorm(1e-4,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.4 )

    elif name=='NICKEL2MO':
        filts=[name+x for x in 'BVRI']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['NICKEL2MOR']-obs['NICKEL2MOI'] > -.2)& (obs['NICKEL2MOB']-obs['NICKE2MOV'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['NICKEL2MOB']).resample(1)[0][0],  stats.exponnorm(1e-4,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.4 )

    elif name=='KAIT3':
        filts=[name+x for x in 'BVRI']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['KAIT3R']-obs['KAIT3I'] > -.2)& (obs['KAIT3B']-obs['KAIT3V'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['KAIT3B']).resample(1)[0][0],  stats.exponnorm(1e-4,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.4 )

    elif name=='KAIT4':
        filts=[name+x for x in 'BVRI']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['KAIT4R']-obs['KAIT4I'] > -.2)& (obs['KAIT4B']-obs['KAIT4V'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['KAIT4B']).resample(1)[0][0],  stats.exponnorm(1e-4,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.4 )

    elif name=='NICKEL1':
        filts=[name+x for x in 'BVRI']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['NICKEL1R']-obs['NICKEL1I'] > -.2)& (obs['NICKEL1B']-obs['NICKEL1V'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['NICKEL1B']).resample(1)[0][0],  stats.exponnorm(1e-4,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.4 )

    elif name=='NICKEL2':
        filts=[name+x for x in 'BVRI']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['NICKEL2R']-obs['NICKEL2I'] > -.2)& (obs['NICKEL2B']-obs['NICKEL2V'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['NICKEL2B']).resample(1)[0][0],  stats.exponnorm(1e-4,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.4 )

    elif name=='ASASSN1':
        filts=[name+x for x in 'BVri']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['ASASSN1r']-obs['ASASSN1i'] > -.2)& (obs['ASASSN1B']-obs['ASASSN1V'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['ASASSN1B']).resample(1)[0][0],  stats.exponnorm(1e-4,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.4 )

    elif name=='ASASSN2':
        filts=[name+x for x in 'BVri']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['ASASSN2r']-obs['ASASSN2i'] > -.2)& (obs['ASASSN2B']-obs['ASASSN2V'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['ASASSN2B']).resample(1)[0][0],  stats.exponnorm(1e-4,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.4 )

    elif name=='CFA4P1':
        filts=[name+x for x in 'BVri']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['CFA4P1r']-obs['CFA4P1i'] > -.2)& (obs['CFA4P1B']-obs['CFA4P1V'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['CFA4P1B']).resample(1)[0][0],  stats.exponnorm(1e-4,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.4 )

    elif name=='CFA4P2':
        filts=[name+x for x in 'BVri']
        synth,obs=getdata(name)
        obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['CFA4P2r']-obs['CFA4P2i'] > -.2)& (obs['CFA4P2B']-obs['CFA4P2V'] > .3)
        surv=survey(name,lambda obs=obs[obscut]: stats.gaussian_kde(obs['CFA4P2B']).resample(1)[0][0],  stats.exponnorm(1e-4,loc=.2,scale=.3).rvs ,   
              (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'],.4 )


    __surveycache__[name]=surv
    return surv


def generatewhitedwarfs(survoffsets):
    whitedwarf_seds=get_whitedwarf_synths(config['surveys_for_dovekie'])
    whitedwarf_obs = pd.read_csv('spectra/bboyd/DA_WD_DES-update.dat',index_col='Object')

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
    whitedwarftotal=pd.merge(whitedwarf_obs,grouped.sample().set_index('Object')[filts],left_index=True,right_index=True,suffixes=('_obs','_synth'))
    whitedwarftotal=whitedwarftotal.replace(-999.,np.nan)
    whitedwarf_generated=whitedwarf_obs.copy()
    for fullfilt in filts:
        surv,filtname=fullfilt.split('-')
        synth,obs,obserr=whitedwarftotal[fullfilt+'_synth'],whitedwarftotal[fullfilt+'_obs'], whitedwarftotal[fullfilt+'-err']
        isgood=(synth>0) &( obs>0)&(~np.isnan(obs))
        errscale=stats.gamma(2,scale=1).rvs()
        errfloor=stats.uniform(0,0.02).rvs()
        trueerr=np.hypot(obserr*errscale,errfloor)
        generated= synth[isgood]+np.random.normal(scale=trueerr.values[isgood])+survoffsets[surv]['griz'.index(filtname)]
        whitedwarf_generated.loc[generated.index,fullfilt]=generated
    return whitedwarf_generated


def getsurveygenerators(*args,**kwargs):

    names='SNLS','SDSS','CFA3K','CFA3S','CSP','DES','Foundation','PS1SN', "CFA4P1", "CFA4P2" ,'ZTFS', 'ZTFD', 'D3YR'
    #names ='SWIFT', 'KAIT1MO', 'KAIT2MO', 'KAIT3MO', 'KAIT4MO', 'NICKEL1MO', 'NICKEL2MO', 'KAIT3', 'KAIT4', 'NICKEL1', 'NICKEL2', 'ASASSN1', 'ASASSN2', 'PS1', 'PS1SN', 'DES', 'SNLS', 'SDSS', 'CSP', 'CFA3K', 'CFA3S', 'CFA4P2', 'CFA4P1'
    #names='ZTFD','ZTFS'
    for name in names:
        yield name,generatesurvey(name,*args,**kwargs)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speclibrary',type=str,default='calspec23')
    args = parser.parse_args()
    survoffsets=generatesurveyoffsets()
    surveys=getsurveygenerators(survoffsets,speclibrary=args.speclibrary)
###########################################################################
    
    surveys={name:surv for name,surv in surveys}
    for i in range(100):
        survoffsets=generatesurveyoffsets()
        for name,surv in surveys.items():
            surv.offsets=np.concatenate((survoffsets[name],survoffsets['PS1']))
        outputdir=f'output_simulated_apermags+AV/{args.speclibrary}_{i}'
        try:
            os.mkdir(outputdir)
        except FileExistsError:
            print(outputdir+' directory already exists, removing it')
            shutil.rmtree(outputdir)
            os.mkdir(outputdir)
        for name,surv in surveys.items():
            if 'ZTF' in name: continue
            simdata=surv.genstar(surv.nobs)
            with open(outputdir+f'/{name}_observed.csv', 'w') as csvfile:
                out = csv.writer(csvfile, delimiter=',')
                dashednames=[x[:-1]+'-'+x[-1] for x in simdata.dtype.names]
                out.writerow( ['survey']+dashednames+['RA','DEC']+[x+'_AV' for x in dashednames])
                for row in simdata:
                    out.writerow([name]+list(row)+[99,99]+ [0]*len(row))
        if 'ZTF' in surveys:
            double,single=surveys['ZTFD'],surveys['ZTFS']
            with open(outputdir+'/ZTF_observed.csv', 'w') as csvfile:
                out = csv.writer(csvfile, delimiter=',')
                dashednames=[(x[:-1]+'-'+x[-1]).replace('D-','-').replace('S-','-') for x in (single.filtnames+[x.upper() for x in double.filtnames] + ['PS1'+x for x in 'griz'])]
                out.writerow( ['survey']+dashednames+['RA','DEC']+[x+'_AV' for x in dashednames] + [])
                simdata=single.genstar(single.nobs)
                for row in simdata:
                    row=list(row)
                    out.writerow(['ZTF']+list(row[:3])+([-999]*3)+list(row[3:])+[99,99]+ [0]*(len(row)+3))
                simdata=double.genstar(double.nobs)
                for row in simdata:
                    row=list(row)
                    out.writerow(['ZTF']+([-999]*3)+list(row[:3])+list(row[3:])+[99,99]+ [0]*(len(row)+3))
        with open(outputdir+'/simmedoffsets.json','w') as file:
            file.write(json.dumps({name:list(survoffsets[name]) for name in survoffsets}))
    
        generatedwds=generatewhitedwarfs(survoffsets)
        generatedwds.to_csv(outputdir+'/WD_simmed.csv')
###########################################################################
    
    
    offsetdict={'PS1':dict(zip(['PS1'+x for x in 'griz'] ,survoffsets['PS1']))}
    for name in surveys:
        offsetdict[name]=dict(zip(surveys[name].filtnames,survoffsets[name]))
    


if __name__=='__main__': main()


