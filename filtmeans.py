import pandas as pd
import numpy as np
import sys
sys.path.insert(1, 'scripts/')
from helpers import load_config

#need to do this again but just for PS1 regular

jsonload = "DOVEKIE_DEFS.yml"
config = load_config(jsonload)

out = open('filter_means.csv','w')
out.write('SURVEYFILTER,MEANLAMBDA \n')
for fp,fts,ofs,surv in zip(config['filtpaths'],config['filttranss'],config['obsfiltss'],config['survs']):
    for ft,of in zip(fts,ofs):
        d = pd.read_csv(fp+'/'+ft,names=['wavelength', 'trans'],delim_whitespace=True,comment='#')
        #print(d['wavelength'],d['trans'])
        print(ft,round(np.average(d['wavelength'],weights=d['trans'])))
        out.write(surv+str(of)+','+str(round(np.average(d['wavelength'],weights=d['trans'])))+'\n')
        if surv == "PS1SN":
            out.write(surv.strip("SN")+str(of)+','+str(round(np.average(d['wavelength'],weights=d['trans'])))+'\n')

out.close()
