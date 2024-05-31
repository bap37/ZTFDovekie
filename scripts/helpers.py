import pandas as pd
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
from scipy.optimize import curve_fit
from glob import glob
import time, sys, os
import pickle, yaml
from chainconsumer import ChainConsumer
import emcee
import corner
from astroquery.irsa_dust import IrsaDust
import astropy.coordinates as coord
import astropy.units as u
from scipy import interpolate
from jax.scipy.optimize import minimize as jmin
from jax import numpy as jnp


def load_config(config_path):
    with open(config_path, "r") as cfgfile:
        config = yaml.load(cfgfile, Loader=yaml.FullLoader)
    return config

def prep_config(config):
    survmap = config['survmap'] 
    survmap4shift = config['survmap4shift']
    survfiltmap = config['survfiltmap']
    obssurvmap = config['obssurvmap']
    revobssurvmap = config['revobssurvmap']
    revobssurvmapforsnana = config['revobssurvmapforsnana']
    survcolormin = config['survcolormin']
    survcolormax = config['survcolormax']
    synth_gi_range = config['synth_gi_range']
    obsfilts = config['obsfilts']
    snanafilts = config['snanafilts']
    snanafiltsr = config['snanafiltsr']
    relativeweights = config['relativeweights']
    errfloors = config['errfloors']
    return survmap, survmap4shift, survfiltmap, obssurvmap, revobssurvmap, revobssurvmapforsnana, survcolormin, survcolormax, synth_gi_range, obsfilts, snanafilts, snanafiltsr, relativeweights, errfloors,  config['target_acceptance'] , config['n_burnin']


jsonload = 'DOVEKIE_DEFS.yml' #where all the important but unwieldy dictionaries live
config = load_config(jsonload)
survmap, survmap4shift, survfiltmap, obssurvmap, revobssurvmap, revobssurvmapforsnana, survcolormin, survcolormax, synth_gi_range, obsfilts, snanafilts, snanafiltsr, relativeweights, errfloors,target_acceptance , n_burnin = prep_config(config)

filter_means = pd.read_csv('filter_means.csv') 

filter_means = filter_means.set_index(['SURVEYFILTER']).to_dict()['MEANLAMBDA ']

def line(x, a, b):
    return a * x + b

def getoffsetforuniversalslope(surv,filt,slope,meanlambda):
    filtmean=filter_means[surv+filt]
    offset = (filtmean-meanlambda)*slope/1000.
    return offset

def query_irsa(row,col='NA'):
    newrow = row
    coo = coord.SkyCoord(row['RA']*u.deg,row['DEC']*u.deg, frame='icrs')
    table = IrsaDust.get_extinction_table(coo)
    aa = np.argsort(table['LamEff'])
    avinterp = interpolate.interp1d(table['LamEff'][aa]*10000,table['A_SandF'][aa])
    rs = col.replace('_4shooter','S').replace('_keplercam','K').split('-')[0]
    if "CSP" in rs: rs = "CSP" ;
    sys.stdout.flush() 
    return avinterp(filter_means[revobssurvmap[rs]+snanafiltsr[revobssurvmapforsnana[rs]][col[-1]]])

def get_extinction(survdict):
    correctedmags = survdict #bigdfdict[surv]
    for col in correctedmags.columns[1:-2]:
        if col[-1] != 'U':
            correctedmags[col+'_AV'] = survdict.apply(query_irsa,col=col,axis=1)
    return correctedmags

def myround(x, base=5):
    return base * round(x/base)
    
    
def itersigmacut_linefit_jax(x,y,cut,niter=3,nsigma=4):
    returnx = x
    returny = y
    
    for i in range(niter):
        result = jmin(lambda pars: (((line(returnx,pars[0],pars[1],)-returny)*cut)**2).sum(), 
                               x0=jnp.array([0.,0.]), method='BFGS'
                            )
        popt, pcov = result.x, result.hess_inv
        yres = returny-line(returnx,popt[0],popt[1])
        stdev = jnp.std(yres,where=cut) 
        cut= (jnp.abs(yres)<nsigma*stdev ) & cut
        returnx = returnx*(cut)
        returny = returny*(cut)
    return returnx[cut],returny[cut],stdev,yres,popt,pcov


def itersigmacut_linefit(x,y,niter=3,nsigma=4):
    returnx = x
    returny = y
    
    for i in range(niter):
    
        popt, pcov = curve_fit(line, 
                               returnx,
                               returny,
                               p0=[0,0], 
                               ftol=0.1, xtol=0.1)
        yres = returny-line(returnx,popt[0],popt[1])
        stdev = np.std(yres)
        returnx = returnx[np.abs(yres)<nsigma*stdev]
        returny = returny[np.abs(yres)<nsigma*stdev]
    return returnx,returny,stdev,yres,popt,pcov

def walker_maker(nparams, prepos, walkfactor=2):  
    pos = (0.01 * np.random.randn(nparams*walkfactor, nparams))
    for entry in range(nparams):                              
        prepos = list(prepos)
        pos[:,entry] = np.random.normal(np.array(prepos*walkfactor), 0.001) 
    return pos
    #END input_cleaner



################# plotoffsets.py lives below

def create_chains(labels, samples, ndim=10):
    fig, axes = plt.subplots(ndim, figsize=(10, 10), sharex=True)
    
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, -1*i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    plt.tight_layout()
    plt.savefig('testchains.png')
    print('upload testchains.png')
    return "Done" 

def create_labels(filename):
    labels = np.load(filename,allow_pickle=True)['labels']
    for i in range(len(labels)):
        labels[i] = labels[i].replace('_',' ')
    for i,label in enumerate(labels):
        labels[i] = labels[i].replace('offset','O').replace('lamshift','L')
    labels = list(labels)
    return labels

def create_cov(labels, flat_samples):
    plt.clf()
    c = ChainConsumer()
    c.add_chain(flat_samples, parameters=labels)
    _,cov = c.analysis.get_covariance()
    np.savez('EXCALIBUR_COVARIANCE_v1.0.npz',cov=cov,labels=labels)
    fig, ax = plt.subplots(figsize=(14, 12))

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False

    im = ax.matshow(cov, cmap='cividis')
    fig.colorbar(im, cmap='cividis')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels,fontsize=9)
    ax.set_yticklabels(labels,fontsize=10)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_title('Cov')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")#rotation_mode="anchor")
    fig.tight_layout()

    plt.savefig('covmat.png')
    print('upload covmat.png')
    return c

def create_corr(c):
    plt.clf()
    _,corr = c.analysis.get_correlations()

    fig, ax = plt.subplots(figsize=(14, 12))

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False

    im = ax.matshow(corr, cmap='cividis')
    fig.colorbar(im, cmap='cividis')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels,fontsize=9)
    ax.set_yticklabels(labels,fontsize=10)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_title('Correlation')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")#rotation_mode="anchor")                                                                                                                         
    fig.tight_layout()

    plt.savefig('corrmat.png')
    print('upload corrmat.png')
    return "Done"


def create_corner(labels, flat_samples):    
    plt.clf()
    fig = corner.corner(
        flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84],show_titles=True, title_kwargs={"fontsize": 12},
        labelpad=.15, title_fmt='.3f',smooth1d=1.,smooth=1.
    );
    plt.tight_layout()
    plt.savefig('cornerslopespectraslope.png')
    print('upload cornerslopespectraslope.png')
    return "Done"

def create_postoffsets_summary(c):
    summary = c.analysis.get_summary()
    f = open('postoffsets.dat','w')
    f.write('SURVEYFILT OFFSET OFFSETERR\n')
    poss = []
    for key,val in summary.items():
        try:
            print(key,'%.4f %.4f'%(val[1],np.mean([val[1]-val[0],val[2]-val[1]])))
            f.write('%s %.4f %.4f\n'%(key.replace(' ','_')[:-2],val[1],np.mean([val[1]-val[0],val[2]-val[1]])))
            poss.append(val[1])
        except:
            print(key,'did not converge')
            f.write('%s didnt converge\n'%key.replace(' ','_')[:-2])
            poss.append(0.0)

    print('\nwrote postoffsets.dat')
    f.close()
    return poss

def create_likelihoodhistory(fullsamples, poss, ll, surveys_for_chisq, fixsurveynames, surveydata, obsdfs):
    flat_fullsamples = fullsamples.reshape(-1, fullsamples.shape[-1])
    chi2s = []
    steps = []
    print(int(len(flat_fullsamples)/10000)-4)
    for i in range(int(len(flat_fullsamples)/10000)-4):
        this_samples = flat_fullsamples[:(i+2)*10000,:]
        tposs = np.mean(this_samples,axis=0)
        chi2 = ll.remote_full_likelihood(tposs,surveys_for_chisqin=surveys_for_chisq,fixsurveynamesin=fixsurveynames,surveydatain=surveydata,obsdfin=obsdfs,subscript='after_v6',doplot=False,first=False,outputdir='postmcmc')
        chi2s.append(chi2[0])
        steps.append((i+2)*10000)

    plt.figure(figsize=(8,6))
    plt.plot(steps,chi2s,lw=3)
    plt.xlabel('Step - arbitrary')
    plt.ylabel('Log Likelihood')
    plt.savefig('likelihoodhistory.png')
    print('upload likelihoodhistory.png')
    return "Done" 

def create_latex(infile, outfile):
    header = """\\begin{table}
    \centering
    \caption{}
    \label{tab:model_params}
    \\begin{tabular}{cc}
        \hline \n"""
    df = pd.read_csv(infile, sep=' ')
    with open(outfile, "w") as f2:
        f2.write(header)
        for i, row in df.iterrows():
            f2.write(f"{row.SURVEYFILT} & ${row.OFFSET} \pm {row.OFFSETERR}$ \\\\ \n")

        f2.write("\end{tabular}\n")
        f2.write("\end{table}")
