import pandas as pd

filename = 'B_tel_ccd_atm_ext_1.2.dat'
# #wave(A) trans

df = pd.read_csv(filename, delim_whitespace=True)

shiftval = 70

df['#wave(A)'] += shiftval

df.to_csv(filename+f"+{shiftval}", float_format="%g", index=False, sep=' ')
