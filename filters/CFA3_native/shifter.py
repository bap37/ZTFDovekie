import pandas as pd

filename = 'SNLS3_4shooter2_B.dat'
# #wave(A) trans

df = pd.read_csv(filename, sep=r'\s+', names=['#wave(A)', 'trans'])

shiftval = -20

df['#wave(A)'] += shiftval

df.to_csv(filename+f"+{shiftval}", float_format="%g", index=False, sep=' ')
