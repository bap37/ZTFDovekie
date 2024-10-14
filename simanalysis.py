#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import glob
from os import path
import json
from scipy import stats
import matplotlib.pyplot as plt 


# In[3]:


dirs=glob.glob('plots/fakes_*/')


# In[5]:


simmap={ 'PS1':'griz', 
    'SNLS':'griz',
    'SDSS':'griz',
    'CFA3K':'UBVri',
    'CFA3S': 'BVRI'  ,  
    'CSP':'BVgri',
    'PS1SN':'griz',
    
    'DES': 'griz',

    'ZTFS':'gri',
    
    'ZTFD':'GRI',
}
outdir=f'plots/fakes_0'
result=np.genfromtxt(path.join(outdir,'preprocess_dovekie.dat'),dtype=None,names=True,encoding='utf-8')
result=result[result['SPECLIB']=='calspec23']
outfile=np.load(path.join(outdir,'results.npz'))
    
simmap_indexes={}
label=outfile['labels'][0]
for label in outfile['labels']:
    surv,filt=label.split('-')[0],label.split('-')[1].split('_')[0]
    if 'ZTF' in surv: 
        if filt.isupper():
            surv='ZTFD'
        else:
            surv='ZTFS'
    index=simmap[surv].index('V' if surv=='CSP' and filt in 'omn' else filt  )
    simmap_indexes[label]=(surv,index)


# In[6]:
#outdirfstring='plots/fakes_calspeconly_fitstis_{i}'
outdirfstring='plots/fakes_stisonly_fitboth_{i}'
inputlib=('calspec23' if 'calspeconly' in outdirfstring else 'stis_ngsl_v2')
fitlib=('calspec23' if 'fitcalspec' in outdirfstring else 'stis_ngsl_v2')


results=[]
def splittuple(arr):
    return map( lambda x: np.array(list(map(float,x))),list(zip(*np.char.split(arr,'+-'))))

for i in range(100):
    outdir=outdirfstring.format(i=i)
    result=np.genfromtxt(path.join(outdir,'preprocess_dovekie.dat'),dtype=None,names=True,encoding='utf-8')
    result=result[result['SPECLIB']==fitlib]
    outfile=np.load(path.join(outdir,'DOVEKIE.V3.npz'))

    offsetsamples=outfile['samples']
    offsets,offseterrs=np.mean(offsetsamples,axis=0),np.std(offsetsamples,axis=0)
    with open(f'output_simulated_apermags+AV/{inputlib}_{i}/simmedoffsets.json') as file: 
        simmedoffsets=(json.loads(file.read()))
    trueoffsets=np.array([simmedoffsets[(idx:=simmap_indexes[label])[0]][idx[1]] for label in outfile['labels']])
    pval=np.sum(offsetsamples>trueoffsets[np.newaxis,:],axis=0)/offsetsamples.shape[0]
    slopes,errs=list((splittuple(result['D_SLOPE'])))
    synth_slopes,_=list((splittuple(result['S_SLOPE'])))
    results+=[(slopes,errs,synth_slopes,offsets,offseterrs,trueoffsets,pval)]
    
slopes,errs,synth_slopes,offsets,offseterrs,trueoffsets,pval=list(map(np.stack,zip(*results)))


# In[8]:


slopelabels=np.char.add(np.char.add(result['OFFSETSURV'],'-'),result['OFFSETFILT2']) 
offsetlabels=np.char.replace(outfile['labels'],'_offset','')


# In[11]:


plt.hist(stats.kstest(pval,lambda x: x).pvalue,bins=np.linspace(0,1,11,True))
plt.xlabel('KS test p-value')
plt.savefig('offsetpvals.pdf')


# In[12]:




a=(np.abs(np.mean(synth_slopes,axis=0)-np.mean(slopes,axis=0))<.25)
plt.errorbar(np.mean(synth_slopes,axis=0)[a],(np.mean(slopes,axis=0) - np.mean(synth_slopes,axis=0))[a],yerr=np.std(slopes,axis=0)[a],fmt='.')
plt.axhline(0,linestyle='--',color='black')
plt.xlabel('Synthetic slope')
plt.ylabel('Data slope bias (recovered-input)')

plt.savefig('plots/slopebias.pdf')


# In[13]:


plus,minus=result['COLORFILT1']== result['OFFSETFILT1'],result['COLORFILT2']== result['OFFSETFILT1']
zero=~(plus | minus)

bias=np.mean(slopes,axis=0)-np.mean(synth_slopes,axis=0)
bins=np.linspace(-.05,.05,10,True)
plt.hist([np.clip(bias[x],bins[1]-1e-3,bins[-2]+1e-3) for x in [plus,zero,minus]],label=['Resid matches first index','Resid matches none','Resid matches second index'],bins=bins)
plt.legend()
plt.xlabel('Bias in slope')
plt.xticks( list((bins[1:]+bins[:-1])/2) ,[('Outliers')]+ [f'{x:.3f}' for x in list((bins[1:]+bins[:-1])/2)[1:-1]]+ [('Outliers')],rotation=90)
plt.savefig('plots/slopebiashist.pdf')


# In[14]:


plt.errorbar(np.arange(offsets.shape[1]),np.mean((offsets-trueoffsets)/offseterrs,axis=0) ,fmt='k.')
plt.xticks(np.arange(offsets.shape[1]),[x.split('_')[0] for x in (outfile['labels'])],rotation=90);
plt.ylabel('')
zvals=np.abs(np.mean(offsets-trueoffsets,axis=0)/(np.std(offsets,axis=0)/np.sqrt(offsets.shape[0])))
plt.ylabel('Mean $z$-score (true-measured/err)')
plt.savefig('meanzscore.pdf')


# In[15]:


plt.hist( np.clip(((trueoffsets-offsets)/offseterrs).flatten(),-4.9,4.9),bins=np.linspace(-5,5,21),density=True)
plt.plot(plotxs:=np.linspace(-4,4,101), stats.norm.pdf(plotxs))
plt.xlabel('$z$-score')
plt.savefig('offsetzscoreshist.pdf')


# In[16]:


plt.hist([np.corrcoef((slopes-synth_slopes).T)[~np.eye(38,dtype=bool)],np.corrcoef((offsets-trueoffsets).T)[~np.eye(42,dtype=bool)]],
        label=['Slopes','Offsets'])

plt.xlabel('Correlation coefficient (off-diagonal)')

plt.savefig('plots/corrcoefs.pdf')
plt.savefig('plots/corrcoefs.png',dpi=288)


# In[ ]:


plt.plot(np.sqrt(np.mean(offseterrs**2,axis=0)), np.std((offsets-trueoffsets),axis=0),'k.')
plt.ylim(0,0.02)
plt.xlim(0,0.02)
plt.plot([0,.02],[0,.02],'k--')
plt.xlabel('Estimated offset errors')
plt.ylabel('Observed offset scatter')
plt.savefig('plots/simscatter.pdf')


# In[25]:


with open('simbiases.txt','w') as file:
    defcolumnwidth=9
    header=['FILTER']+ ['OFFSETBIAS' , 'OFFSETZSCORE','OFFSETERRMEAN','OFFSETSCATTER', 'OFFSETPVAL','SLOPEBIAS','SLOPEERROR','SLOPEZSCORE','SLOPEPVAL']
    def formatline(arr):
        return' '.join([('{: >'+(str(max(defcolumnwidth,len(header)+3)))+'.4f}').format(x) if type(x) is np.float64 else ('{: >'+(str(max(defcolumnwidth,len(header)+3)))+'}').format(x) for x,head in zip(arr,header)])
    file.write(formatline(header)+'\n')    
    for filt in np.unique(np.concatenate((offsetlabels,slopelabels))):
        idx=np.where(filt==offsetlabels)[0]
        if len(idx):
            meanbias=np.mean((offsets[:,idx]-trueoffsets[:,idx]))
            meanzscore= np.mean((offsets[:,idx]-trueoffsets[:,idx])/offseterrs[:,idx])
            offerrest,offerrobs=np.sqrt(np.mean(offseterrs[:,idx]**2)), np.std((offsets[:,idx]-trueoffsets[:,idx]))
            kspval=stats.kstest(pval[:,idx],lambda x: x).pvalue[0]
        else: meanbias , meanzscore,offerrest,offerrobs, kspval = ['NA']*3
        idx=np.where(filt==slopelabels)[0]
        if len(idx):
            slopebias=np.mean(slopes[:,idx]-synth_slopes[:,idx])
            slopeerr=np.std(slopes[:,idx]-synth_slopes[:,idx])
            slopebiaszscore=np.mean(slopes[:,idx]-synth_slopes[:,idx])/(slopeerr)
            slopebiaspval= (1-stats.norm.cdf(np.abs(slopebiaszscore)))*2
        else:
            slopebias,slopeerr,slopebiaszscore,slopebiaspval= ['NA']*4
        allres=[meanbias , meanzscore,offerrest,offerrobs, kspval,slopebias,slopeerr,slopebiaszscore,slopebiaspval]
        file.write(formatline([filt]+allres)+'\n')


# In[20]:





# In[36]:





# In[ ]:





# In[ ]:




