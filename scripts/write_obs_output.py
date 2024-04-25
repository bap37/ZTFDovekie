import pandas as pd
import numpy as np

def write(survey,colnames,collists,outfile):
    d = {'survey':[survey for x in collists[0]]}
    for col,dat in zip(colnames,collists):
        d[col] = np.array(dat,dtype='float')
    df = pd.DataFrame(d)
    df.to_csv(outfile,index=False)
