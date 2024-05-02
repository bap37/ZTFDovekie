#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import csv
import os
import json



def loadsynthphot(fname):
    
    result=np.genfromtxt(fname,names=True,dtype=None,encoding='utf-8',delimiter=' ')
    for key,type in result.dtype.descr:
        if type=='<f8':
            result[key]*=1
    return result


#Function copied from  https://github.com/axiezai/factor_analysis
def bfa(y, num_fa, num_it, mixing_matrix = None, noise_precision = None):
    """
    Bayesian Factor analysis with expectation maximization algorithm
    
    Args:
        - y (array): Nsensor x Ntime data 
        - num_fa (int): number of factors in factor analysis
        - num_it (int): number of iterations for EM updates
        - mixing_matrix (array): The factor loading matrix A in the factor analysis model. If stays as default value of None, the function initializes with SVD of autocorrelation of y
        - noise_precision (array): the noise precision matrix, also defaults to None and can be initialized.
    
    Outputs: (everything gets plotted)
        - mixing_matrix
        - noise_precision
        - xbar - x after update rules are applied.
        - igamma - the variance matrix
    """
    num_y = y.shape[0] #data number of sensors
    num_t = y.shape[1] #data time samples
    # auto-correlation:
    ryy = np.matmul(y,np.transpose(y))
    
    # initialize SVD:
    if mixing_matrix is None:
        U, SV, V = np.linalg.svd(ryy/num_t)
        D = np.diag(SV)
        mixing_matrix = U * np.diag(np.sqrt(D))
        mixing_matrix = mixing_matrix[:,0:num_fa]
        noise_precision_matrix = np.diag(num_t/np.diag(ryy))
    elif noise_precision is None:
        mixing_matrix = mixing_matrix
        noise_precision_matrix = np.diag(num_t/np.diag(ryy))
    else:
        noise_precision_matrix=noise_precision
    # EM iteration:
    likelihood = np.zeros([num_it, 1])
    lambda_psi = np.matmul(np.matmul(np.transpose(mixing_matrix),noise_precision_matrix), mixing_matrix)

    for i in np.arange(0, num_it):
        gamma = lambda_psi + np.eye(num_fa)
        igamma = np.linalg.inv(gamma)
        xbar = np.matmul(np.matmul(np.matmul(igamma, np.transpose(mixing_matrix)),noise_precision_matrix), y)

        # update rules:
        ldnpm = np.sum(np.log(np.diag(noise_precision_matrix/(2*np.pi))))
        ldgam = np.sum(np.log(np.linalg.svd(gamma)[1]))
        likelihood[i] = 0.5*num_t*(ldnpm - ldgam) - 0.5*np.sum(np.sum(y*np.matmul(noise_precision_matrix,y))) + 0.5*np.sum(np.sum(xbar*np.matmul(gamma,xbar)))

        # update:
        ryx = np.matmul(y,xbar.transpose())
        rxx = np.matmul(xbar,xbar.transpose()) + num_t*igamma
        psi = np.linalg.inv(rxx)
        mixing_matrix = np.matmul(ryx,psi)
        noise_precision_matrix = np.diag(num_t/np.diag(ryy - np.matmul(mixing_matrix, ryx.transpose())))
        lambda_psi = np.matmul(mixing_matrix.transpose(), np.matmul(noise_precision_matrix, mixing_matrix))
        
    return mixing_matrix, noise_precision_matrix, xbar, igamma



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
        self.magdist=magdist
        self.colordist=colordist
        self.filtnames= filtnames
        self.isps1survey= isps1survey
        
        #Determine excess variance of observed populations
        data=np.array([obs[x] for x in filtnames]+ [obs['PS1'+x] for x in 'griz'])
        data-=np.mean(data,axis=1)[:,np.newaxis]
        #Use bayesian factor analysis on observed data
        factors,precision,_,_=bfa(data, 2,10000)
        self.factors=factors
        self.variance=1/(np.diag(precision))
        
        
        self.magind,self.colorind=magcolorinds
        #If it's a PS1 survey, the synthetic photometry will be identical to PS1, so don't regress on identical data
        if self.isps1survey: 
            mags=np.array([survsynth[x] for x in filtnames])
        else:
            mags=np.array([survsynth[x] for x in filtnames]+ [ps1synth['PS1'+x] for x in 'griz'])
        
        #Use bayesian factor analysis to reduce the dimensionality of the synthetic photometry
        self.means=np.mean(mags,axis=1)
        mags-=self.means[:,np.newaxis]
        synth_factors,_,_,_=bfa(mags, 2, 10000,)
        #Renormalize factors
        synth_factors[:,1]=(synth_factors[:,1]-synth_factors[self.magind,1]/synth_factors[self.magind,0] *synth_factors[:,0])
        synth_factors[:,1]=-synth_factors[:,1]/synth_factors[self.colorind,1]
        synth_factors[:,self.magind]*=np.std(coords,axis=1)[self.magind]
        self.synth_factors=synth_factors
        self.offsets=np.concatenate([survoffsets,ps1offsets] )
        
    def genstar(self,size=1):
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
            coords=(np.random.normal(), color)
            #Generate  magnitudes from factor analysis
            mags=self.synth_factors@ np.array(coords)+self.means
            mags+= mag-mags[self.magind]
            
            if self.isps1survey:
                mags=np.concatenate((mags,mags))
            #Add noise
            mags+=(np.random.normal(0,np.sqrt(self.variance)))
            #Add zero-point offsets
            mags+=self.offsets
            return tuple(mags)
        else:
            return np.array([(self.genstar()) for i in range(size)], dtype=list(zip(list(self.filtnames)+['PS1'+x for x in 'griz'],

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






survnames=['CFA3K','CFA3S','CSPDR3nat', 'DES','Foundation','PS1SN','SDSS','SNLS','ZTF']

surveys={}

ps1offsets=np.random.normal(0,0.01,4)
survoffsets={'PS1':ps1offsets }
ps1synth=loadsynthphot('output_synthetic_magsaper/synth_PS1_shift_0.000.txt')
cut=(ps1synth['standard_catagory']=='calspec23')& ( ps1synth['PS1g']-ps1synth['PS1r'] > -1 ) &(ps1synth['PS1g']-ps1synth['PS1r'] <.8)


def getdata(survname):
    
    synth=loadsynthphot(f'output_synthetic_magsaper/synth_{survname}_shift_0.000.txt')
    obs=np.genfromtxt(f'output_observed_apermags+AV/{survname}_observed.csv',names=True,dtype=None,encoding='utf-8',delimiter=',')
    return synth,obs


###########################################################################

#Initialize survey objects
name='SNLS'
synth,obs=getdata(name)
survoffsets[name]= np.random.normal(0,0.01,4)
surv=survey(lambda obs=obs: stats.gaussian_kde(obs['SNLSg']).resample(1)[0][0],lambda : np.random.uniform(0,.5)+.2,
      
      (0,1),['SNLS'+x for x in 'griz'],obs, synth[cut],ps1synth[cut], survoffsets[name],ps1offsets )

surveys[name]=surv

name='SDSS'
synth,obs=getdata(name)
survoffsets[name]= np.random.normal(0,0.01,4)
surv=survey(lambda obs=obs: stats.gaussian_kde(obs['SDSSg']).resample(1)[0][0],  stats.exponnorm(1.8564479517780803, 0.4772301484843199, 0.07070238022618985).rvs ,
      
      (0,1),[name+x for x in 'griz'],obs, synth[cut],ps1synth[cut], survoffsets[name],ps1offsets )
surveys[name]=surv

name='CFA3K'
filts=[name+x for x in 'UBVri']

synth,obs=getdata(name)
obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['CFA3KU']-obs['CFA3KB'] > -.5) & (obs['CFA3KU']-obs['CFA3KB'] < 20)

survoffsets[name]= np.random.normal(0,0.01,len(filts))
surv=survey(lambda obs=obs[obscut]: stats.gaussian_kde(obs['CFA3KB']).resample(1)[0][0],  stats.uniform(.25,.3).rvs ,
      
      (1,2),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],ps1offsets )
surveys[name]=surv

name='CFA3S'
filts=[name+x for x in 'BVRI']

synth,obs=getdata(name)
obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['CFA3SR']-obs['CFA3SI'] > -.2)& (obs['CFA3SB']-obs['CFA3SV'] > .3)


survoffsets[name]= np.random.normal(0,0.01,len(filts))
surv=survey(lambda obs=obs[obscut]: stats.gaussian_kde(obs['CFA3SB']).resample(1)[0][0],  stats.uniform(.45,.4).rvs ,
      
      (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],ps1offsets )
surveys[name]=surv

name='CSPDR3nat'


synth=loadsynthphot(f'output_synthetic_magsaper/synth_CSP_shift_0.000.txt')
obs=np.genfromtxt(f'output_observed_apermags+AV/CSPDR3nat_observed.csv',names=True,dtype=None,encoding='utf-8',delimiter=',')
synth=np.array(list(synth), dtype=[(x.replace('CSP_TAMU',name),synth.dtype.fields[x][0]) for x in synth.dtype.fields])
filts=[name+x for x in 'BVgri']

obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )#& (obs['CFA3SR']-obs['CFA3SI'] > -.2)& (obs['CFA3SB']-obs['CFA3SV'] > .3)


survoffsets[name]= np.random.normal(0,0.01,len(filts))
surv=survey(lambda obs=obs[obscut]: stats.gaussian_kde(obs['CSPDR3natB']).resample(1)[0][0],  stats.uniform(.45,.4).rvs ,
      
      (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],ps1offsets )
surveys[name]=surv

name='DES'
filts=[name+x for x in 'griz']

synth,obs=getdata(name)
obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )& (obs['DESg']-obs['DESr'] > .2)& (obs['DESg']-obs['DESr'] <1)


survoffsets[name]= np.random.normal(0,0.01,len(filts))
surv=survey(lambda obs=obs[obscut]: stats.gaussian_kde(obs['DESg']).resample(1)[0][0],  stats.uniform(.45,.4).rvs ,
      
      (0,1),filts,obs[obscut], synth[cut],ps1synth[cut], survoffsets[name],ps1offsets )
surveys[name]=surv

name='Foundation'
filts=[name+x for x in 'griz']
synth,obs=getdata(name)
nans=(~reduce(lambda x,y: x|y,[np.isnan(synth[x]) for x in filts],False))
obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )#& (obs['Foundationg']-obs['DESr'] > .2)& (obs['DESg']-obs['DESr'] <1)


survoffsets[name]= np.random.normal(0,0.01,len(filts))
surv=survey(lambda obs=obs[obscut]: stats.gaussian_kde(obs['Foundationg']).resample(1)[0][0],  stats.uniform(.25,.4).rvs ,
      
      (0,1),filts,obs[obscut], synth[cut&nans],ps1synth[cut&nans], survoffsets[name],ps1offsets ,True)
surveys[name]=surv

name='PS1SN'
filts=[name+x for x in 'griz']
synth,obs=getdata(name)
synth=np.array(list(synth), dtype=[(x.replace('Foundation',name),synth.dtype.fields[x][0]) for x in synth.dtype.fields])
nans=(~reduce(lambda x,y: x|y,[np.isnan(synth[x]) for x in filts],False))
obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )#& (obs['Foundationg']-obs['DESr'] > .2)& (obs['DESg']-obs['DESr'] <1)

survoffsets[name]= np.random.normal(0,0.01,len(filts))
surv=survey(lambda obs=obs[obscut]: stats.gaussian_kde(obs['PS1SNg']).resample(1)[0][0],  stats.uniform(.25,.6).rvs ,
      
      (0,1),filts,obs[obscut], synth[cut&nans],ps1synth[cut&nans], survoffsets[name],ps1offsets ,True)
surveys[name]=surv

name='DES'
filts=[name+x for x in 'griz']
synth,obs=getdata(name)
nans=(~reduce(lambda x,y: x|y,[np.isnan(synth[x]) for x in filts],False))
obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )#& (obs['Foundationg']-obs['DESr'] > .2)& (obs['DESg']-obs['DESr'] <1)
survoffsets[name]= np.random.normal(0,0.01,len(filts))
surv=survey(lambda obs=obs[obscut]: stats.gaussian_kde(obs['DESg']).resample(1)[0][0],  stats.uniform(.4,.5).rvs ,
      (0,1),filts,obs[obscut], synth[cut&nans],ps1synth[cut&nans], survoffsets[name],ps1offsets )
surveys[name]=surv

name='ZTF'
filts=[name+x for x in 'gri']
synth,obs=getdata(name)
nans=(~reduce(lambda x,y: x|y,[np.isnan(synth[x]) for x in filts],False))

obscut=(~reduce(lambda x,y: x|y,[np.abs(obs[x])>30 for x in filts],False) )#& (obs['Foundationg']-obs['DESr'] > .2)& (obs['DESg']-obs['DESr'] <1)


survoffsets[name]= np.random.normal(0,0.01,len(filts))
surv=survey(lambda obs=obs[obscut]: stats.gaussian_kde(obs['ZTFg']).resample(1)[0][0],  stats.uniform(.3,.5).rvs ,
      
      (0,1),filts,obs[obscut], synth[cut&nans],ps1synth[cut&nans], survoffsets[name],ps1offsets )
surveys[name]=surv

###########################################################################


for name in surveys:
    surv=surveys[name]
    simdata=surv.genstar(obs.size)
    with open(f'output_simulated_apermags+AV/{name}_simobserved.csv', 'w') as csvfile:
        out = csv.writer(csvfile, delimiter=',')
        dashednames=[x[:-1]+'-'+x[-1] for x in simdata.dtype.names]
        out.writerow( ['survey']+dashednames+['RA','DEC']+[x+'_AV' for x in dashednames])
        for row in simdata:
            out.writerow([name]+list(row)+[99,99]+list(row))

###########################################################################


offsetdict={'PS1':dict(zip(['PS1'+x for x in 'griz'] ,survoffsets['PS1']))}
for surv in surveys:
    offsetdict[surv]=dict(zip(surveys[surv].filtnames,survoffsets[surv]))

with open('output_simulated_apermags+AV/simmedoffsets.json','w') as file:
    file.write(json.dumps(offsetdict))





