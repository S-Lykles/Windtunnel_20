#%%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
from matplotlib import pyplot as plt
# %%
p = Path('Numerical data 2d')

# %%
# data is sepperated by tabs, of skipinitialspace is not set to True all the collumns will have 
# spaces in them, for example '     Alpha' instead of 'Alpha', same for the indices.
# indices are strings, so if you want to select by runnumber corr_data.loc['1']
corr_data = pd.read_csv(p/'corr_test.txt', sep='\t', index_col=0, skipinitialspace=True)
corr_data.rename({'/': 'info'}, inplace=True)
columns = corr_data.columns

Cpu_idx = [column for column in columns if 'Cpu' in column]
Cpl_idx = [column for column in columns if 'Cpl' in column]
# Cpu_idx = columns.map(lambda idx: True if 'Cpu' in idx else False)
# Clu_idx = columns.map(lambda idx: True if 'Clu' in idx else False)

# %%
Cpu = corr_data[['Alpha']+Cpu_idx]
Clu = corr_data[['Alpha']+Cpl_idx]

# %%
