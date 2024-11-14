import pandas as pd

filename = 'effMEGACAM-z.dat'
# #wave(A) trans

df = pd.read_csv(filename, sep=r'\s+', names=['#wave(A)', 'trans'], comment="#")

shiftval = +30

df['#wave(A)'] += shiftval

df.to_csv(filename+f"+{shiftval}", float_format="%g", index=False, sep=' ')
