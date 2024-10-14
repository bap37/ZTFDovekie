import pandas as pd

filename = 'ztfg2.dat'
# #wave(A) trans

df = pd.read_csv(filename, delim_whitespace=True)

shiftval = +20

df['#wave(A)'] += shiftval

df.to_csv(filename+f"+{shiftval}", float_format="%g", index=False, sep=' ')
