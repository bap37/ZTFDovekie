#!/usr/bin/env python


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

def loadsynthphot(fname):
    
    result=np.genfromtxt(fname,names=True,dtype=None,encoding='utf-8',delimiter=' ')
    for key,type in result.dtype.descr:
        if type=='<f8':
            result[key]*=1
    return result

def getdata(survname):
    synth=loadsynthphot(f'output_synthetic_magsaper/synth_{survname}_shift_0.000.txt')
    obs=np.genfromtxt(f'output_observed_apermags+AV/{survname}_observed.csv',names=True,dtype=None,encoding='utf-8',delimiter=',')
    return synth,obs


@jax.jit
def unpack(data,colorinds,pars):
    nfilts,nstars=data.shape
    
    mags,colors, slopes,noise,intrinsiccolor= pars[:nstars],pars[nstars:2*nstars], pars[2*nstars:2*nstars+nfilts],pars[2*nstars+nfilts:2*nstars+2*nfilts],pars[2*nstars+2*nfilts:]
    intrinsiccolor=intrinsiccolor.at[colorinds[0]].set(0)
    slopes=slopes.at[jnp.array(colorinds)].set([0,-1])
    noise=jnp.exp(noise)
    return mags,colors, slopes,noise,intrinsiccolor


def linearfit(data,colorinds,pars):
    mags,colors, slopes,noise,intrinsiccolor=unpack(data,colorinds,pars)
    modelled=mags+colors[np.newaxis,:]*slopes[:,np.newaxis]+intrinsiccolor[:,np.newaxis]
    return -jsci.stats.norm.logpdf(data.flatten(),modelled.flatten(),jnp.tile(noise,(1,data.shape[1])).flatten()).sum()

grad= jax.jit(jax.grad(linearfit,argnums=2),static_argnums=1)
linearfit=jax.jit(linearfit,static_argnums=1)



def decomposedata(data,colorinds,maxiter=100000):
    #Decompose each star into a sum of four components; a mean/intrinsic color, a color described by fixed linear relations in each band, a magnitude, and noise
    assert(np.isnan(data).sum()==0)
    nfilts,nstars=data.shape
    magind,colorind=colorinds
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
    parsinit=np.concatenate( [maginit,colorinit,slopesinit,noiseinit,meancolorinit
    ])
    
    #Minuits function call limit wasn't working, so I wrote my own
    class counter:
        def __init__(self):
            self.steps=0
            self.vals=0
            self.minval=np.inf
    tracker=counter()
    def passthrough(x):
        result=linearfit(data,colorinds,x)
        tracker.steps+=1
        if tracker.minval >result : tracker.vals=x
        if tracker.steps > maxiter: raise RuntimeError()
        return result
    
    
    isnoise=np.zeros(parsinit.size,dtype=bool)
    isnoise[2*nstars+nfilts:2*nstars+2*nfilts]=True
    #Fit all parameters other than noise first
    optimizer=Minuit( passthrough, parsinit,grad=partial(grad,data,colorinds)  )
    optimizer.fixed=isnoise
    try:
        result=optimizer.migrad()
        vals=np.array(result.values)
    except RuntimeError as e:
        vals=tracker.vals
    #Fit noise
    tracker.steps=0
    optimizer=Minuit( passthrough, vals,grad=partial(grad,data,colorinds)  )
    optimizer.fixed=~isnoise
    try:
        result=optimizer.migrad()
        vals=np.array(result.values)
    except RuntimeError as e:
        vals=tracker.vals

    return unpack(data,colorinds,np.array(vals))
    



class survey:
    
    def __init__(self, magdist, colordist, magcolorinds, filtnames,obs, survsynth,ps1synth,survoffsets,ps1offsets,isps1survey=False):
        """
        Initialize an object to produce simulated data from a given survey.
    
        Parameters:
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
        self.magind,self.colorind=magcolorinds
        assert(self.magind!=self.colorind)
        self.nobs=obs.size
        self.magdist=magdist
        self.colordist=colordist
        self.filtnames= filtnames
        self.isps1survey= isps1survey
        
        #Determine excess variance of observed populations
        data=np.array([obs[x] for x in filtnames]+ [obs['PS1'+x] for x in 'griz'])
        #Use factor analysis on observed data
        
        _,_,_,noise,_=decomposedata(data,magcolorinds)
        self.variance=noise**2
        
        
        #If it's a PS1 survey, the synthetic photometry will be identical to PS1, so don't regress on identical data
        if self.isps1survey: 
            mags=np.array([survsynth[x] for x in filtnames])
        else:
            mags=np.array([survsynth[x] for x in filtnames]+ [ps1synth['PS1'+x] for x in 'griz'])
        
        #Use factor analysis to reduce the dimensionality of the synthetic photometry
        _,_,self.slopes,_,self.intrinsic=decomposedata(mags,(0,1))
        
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
            #Draw magnitude and color
            mag=self.magdist()
            color=self.colordist()
            #Generate  magnitudes from factor analysis
            mags= self.slopes*color + self.intrinsic + mag
            if self.isps1survey:
                mags=np.concatenate((mags,mags))
            #Add noise
            if addvariance: mags+=(np.random.normal(0,np.sqrt(self.variance)))
            #Add zero-point offsets
            mags+=self.offsets
            return tuple(mags)
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
    filts=[name+x for x in 'UBVri']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))
    
    name='CFA3S'
    filts=[name+x for x in 'BVRI']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))
    
    name='CSPDR3nat'
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
    
    name='ZTF'
    filts=[name+x for x in 'gri']
    survoffsets[name]= np.random.normal(0,0.01,len(filts))
    return survoffsets

def getsurveygenerators(survoffsets):

    surveys={}
    ps1synth=loadsynthphot('output_synthetic_magsaper/synth_PS1_shift_0.000.txt')
    cut=(ps1synth['standard_catagory']=='calspec23')& ( ps1synth['PS1g']-ps1synth['PS1r'] > -1 ) &(ps1synth['PS1g']-ps1synth['PS1r'] <.8)

    name='SNLS'
    print(f'Preparing {name}')
    synth,obs=getdata(name)
    surv=survey(lambda obs=obs: stats.gaussian_kde(obs['SNLSg']).resample(1)[0][0],lambda : np.random.uniform(0,.5)+.2,
          (0,1),['SNLS'+x for x in 'griz'],obs, synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'] )
    surveys[name]=surv
    
    name='SDSS'
    print(f'Preparing {name}')
    synth,obs=getdata(name)
    surv=survey(lambda obs=obs: stats.gaussian_kde(obs['SDSSg']).resample(1)[0][0],  stats.exponnorm(1.8564479517780803, 0.4772301484843199, 0.07070238022618985).rvs ,
          (0,1),[name+x for x in 'griz'],obs, synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'] )
    surveys[name]=surv
    
    name='CFA3K'
    print(f'Preparing {name}')
    filts=[name+x for x in 'UBVri']
    synth,obs=getdata(name)
    obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['CFA3KU']-obs['CFA3KB'] > -.5) & (obs['CFA3KU']-obs['CFA3KB'] < 20)
    surv=survey(lambda obs=obs[obscut]: stats.gaussian_kde(obs['CFA3KB']).resample(1)[0][0],  stats.uniform(.25,.3).rvs ,
          (1,2),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'] )
    surveys[name]=surv
    
    name='CFA3S'
    print(f'Preparing {name}')
    filts=[name+x for x in 'BVRI']
    synth,obs=getdata(name)
    obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['CFA3SR']-obs['CFA3SI'] > -.2)& (obs['CFA3SB']-obs['CFA3SV'] > .3)
    surv=survey(lambda obs=obs[obscut]: stats.gaussian_kde(obs['CFA3SB']).resample(1)[0][0],  stats.uniform(.45,.4).rvs ,   
          (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'] )
    surveys[name]=surv
    
    name='CSPDR3nat'
    print(f'Preparing {name}')
    synth=loadsynthphot(f'output_synthetic_magsaper/synth_CSP_shift_0.000.txt')
    obs=np.genfromtxt(f'output_observed_apermags+AV/CSPDR3nat_observed.csv',names=True,dtype=None,encoding='utf-8',delimiter=',')
    synth=np.array(list(synth), dtype=[(x.replace('CSP_TAMU',name),synth.dtype.fields[x][0]) for x in synth.dtype.fields])
    filts=[name+x for x in 'BVgri']
    obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )#& (obs['CFA3SR']-obs['CFA3SI'] > -.2)& (obs['CFA3SB']-obs['CFA3SV'] > .3)
    surv=survey(lambda obs=obs[obscut]: stats.gaussian_kde(obs['CSPDR3natB']).resample(1)[0][0],  stats.uniform(.45,.4).rvs ,   
          (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'] )
    surveys[name]=surv
    
    name='DES'
    print(f'Preparing {name}')
    filts=[name+x for x in 'griz']
    synth,obs=getdata(name)
    obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['DESg']-obs['DESr'] > .2)& (obs['DESg']-obs['DESr'] <1)
    surv=survey(lambda obs=obs[obscut]: stats.gaussian_kde(obs['DESg']).resample(1)[0][0],  stats.uniform(.3,.3).rvs ,    
          (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],survoffsets['PS1'] )
    surveys[name]=surv
    
    name='Foundation'
    print(f'Preparing {name}')
    filts=[name+x for x in 'griz']
    synth,obs=getdata(name)
    nans=(~reduce(lambda x,y: x|y,[np.isnan(synth[x]) for x in filts],False))
    obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )#& (obs['Foundationg']-obs['DESr'] > .2)& (obs['DESg']-obs['DESr'] <1)
    surv=survey(lambda obs=obs[obscut]: stats.gaussian_kde(obs['Foundationg']).resample(1)[0][0],  stats.uniform(.25,.4).rvs ,      
          (0,1),filts,obs[obscut], synth[cut&nans],ps1synth[cut&nans], survoffsets[name],survoffsets['PS1'] ,True)
    surveys[name]=surv
    
    name='PS1SN'
    filts=[name+x for x in 'griz']
    synth,obs=getdata(name)
    synth=np.array(list(synth), dtype=[(x.replace('Foundation',name),synth.dtype.fields[x][0]) for x in synth.dtype.fields])
    nans=(~reduce(lambda x,y: x|y,[np.isnan(synth[x]) for x in filts],False))
    obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )#& (obs['Foundationg']-obs['DESr'] > .2)& (obs['DESg']-obs['DESr'] <1)
    surv=survey(lambda obs=obs[obscut]: stats.gaussian_kde(obs['PS1SNg']).resample(1)[0][0],  stats.uniform(.25,.6).rvs ,  
          (0,1),filts,obs[obscut], synth[cut&nans],ps1synth[cut&nans], survoffsets[name],survoffsets['PS1'] ,True)
    surveys[name]=surv
    
    
    name='ZTF'
    print(f'Preparing {name}')
    filts=[name+x for x in 'gri']
    synth,obs=getdata(name)
    nans=(~reduce(lambda x,y: x|y,[np.isnan(synth[x]) for x in filts],False))
    obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )#& (obs['Foundationg']-obs['DESr'] > .2)& (obs['DESg']-obs['DESr'] <1)
    surv=survey(lambda obs=obs[obscut]: stats.gaussian_kde(obs['ZTFg']).resample(1)[0][0],  stats.uniform(.3,.5).rvs ,   
          (0,1),filts,obs[obscut], synth[cut&nans],ps1synth[cut&nans], survoffsets[name],survoffsets['PS1'] )
    surveys[name]=surv

    return surveys
    
def main():
    survnames=['CFA3K','CFA3S','CSPDR3nat', 'DES','Foundation','PS1SN','SDSS','SNLS','ZTF']
    survoffsets=generatesurveyoffsets()
    surveys=getsurveygenerators(survoffsets)
    
###########################################################################
    
    
    for name in surveys:
        surv=surveys[name]
        simdata=surv.genstar(surv.nobs)
        with open(f'output_simulated_apermags+AV/{name}_simobserved.csv', 'w') as csvfile:
            out = csv.writer(csvfile, delimiter=',')
            dashednames=[x[:-1]+'-'+x[-1] for x in simdata.dtype.names]
            out.writerow( ['survey']+dashednames+['RA','DEC']+[x+'_AV' for x in dashednames])
            for row in simdata:
                out.writerow([name]+list(row)+[99,99]+ [0]*len(row))

###########################################################################
    
    
    offsetdict={'PS1':dict(zip(['PS1'+x for x in 'griz'] ,survoffsets['PS1']))}
    for name in surveys:
        offsetdict[name]=dict(zip(surveys[name].filtnames,survoffsets[name]))
    
    with open('output_fake_apermags+AV/simmedoffsets.json','w') as file:
        file.write(json.dumps(offsetdict))


if __name__=='__main__': main()


