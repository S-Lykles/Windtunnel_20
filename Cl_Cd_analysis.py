
#%%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

# %%
p = Path('Numerical data 3d balance')
# data is sepperated by tabs, of skipinitialspace is not set to True all the collumns will have 
# spaces in them, for example '     Alpha' instead of 'Alpha', same for the indices.
# indices are strings, so if you want to select by runnumber corr_data.loc['1']
data_3d = pd.read_csv(p/'corr_test(1).txt', sep='\t', index_col=0, skipinitialspace=True)
data_3d.rename({'/': 'info'}, inplace=True)
data_3d.loc['info', 'Alpha'] = np.nan
data_3d['Alpha'] = data_3d['Alpha'].astype('float64')

p = Path('Numerical data 2d')
data_2d = pd.read_csv(p/'corr_test.txt', sep='\t', index_col=0, skipinitialspace=True)
data_2d.rename({'/': 'info'}, inplace=True)
data_2d.loc['info', 'Alpha'] = np.nan
data_2d['Alpha'] = data_2d['Alpha'].astype('float64')

# %%
def plot_Cl_a(df_1, df_2=None, name=None):
    if name==None:
        name=['']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_dpi(120)
    ax.grid(True, alpha=1, lw=0.5, ls='--')
    ax.axhline(0, c='k', ls='-', lw=0.5)
    ax.axvline(0, c='k', ls='-', lw=0.5)
    c_idx = df_1.columns.map(lambda c: 'cl' in c.lower())
    y = df_1.loc[1:, c_idx].astype('float64')
    alpha = df_1.Alpha[1:]
    ax.plot(alpha, y, label=name[0], lw=0.8, ls='-')
    ax.scatter(alpha, y, label=name[0], s=4)
    if df_2 is not None:
        c_idx = df_2.columns.map(lambda c: 'cl' in c.lower())
        y = df_2.loc[1:, c_idx].astype('float64')
        alpha = df_2.Alpha[1:]
        ax.plot(alpha, y, label=name[0], lw=0.8, ls='-')
        ax.scatter(alpha, y, label=name[0], s=4)





# %%
