
#%%
import numpy as np
from numpy.lib import polynomial
import pandas as pd
from pathlib import Path
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from numpy.polynomial.polynomial import polyval, polyfit

# %%
p = Path('Experimental data 3d balance')
# data is sepperated by tabs, of skipinitialspace is not set to True all the collumns will have 
# spaces in them, for example '     Alpha' instead of 'Alpha', same for the indices.
# indices are strings, so if you want to select by runnumber corr_data.loc['1']
data_3d = pd.read_csv(p/'corr_test(1).txt', sep='\t', index_col=0, skipinitialspace=True)
data_3d.rename({'/': 'info'}, inplace=True)
data_3d.loc['info', 'Alpha'] = np.nan
data_3d['Alpha'] = data_3d['Alpha'].astype('float64')

p = Path('Experimental data 2d')
data_2d = pd.read_csv(p/'corr_test.txt', sep='\t', index_col=0, skipinitialspace=True)
data_2d.rename({'/': 'info'}, inplace=True)
data_2d.loc['info', 'Alpha'] = np.nan
data_2d['Alpha'] = data_2d['Alpha'].astype('float64')

#%%
style_2d = dict(marker='o', linestyle='-', markersize=6,
                           color='black',
                           markerfacecolor='tab:green',
                           markerfacecoloralt='red',
                           markeredgecolor='black')
style_3d = dict(marker='v', linestyle='-', markersize=6,
                           color='black',
                           markerfacecolor='tab:blue',
                           markerfacecoloralt='red',
                           markeredgecolor='black')


# %%
def plot_Cl_a():
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    fig.set_dpi(120)
    ax.grid(True, alpha=1, lw=0.5, ls='--')
    ax.axhline(0, c='k', ls='-', lw=0.5)
    ax.axvline(0, c='k', ls='-', lw=0.5)
    CL= data_3d.CL[1:].astype('float64')
    alpha = data_3d.Alpha[1:]
    ax.plot(alpha, CL, **style_3d, label='3d')
    Cl = data_2d.Cl[1:].astype('float64')
    alpha = data_2d.Alpha[1:]
    ax.plot(alpha, Cl, **style_2d, label='2d')
    ax.legend()

def plot_Cd_a():
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    fig.set_dpi(120)
    ax.grid(True, alpha=1, lw=0.5, ls='--')
    ax.axhline(0, c='k', ls='-', lw=0.5)
    ax.axvline(0, c='k', ls='-', lw=0.5)
    CD = data_3d.CD[1:].astype('float64')
    alpha = data_3d.Alpha[1:]
    ax.plot(alpha, CD, **style_3d, label='3d')
    Cd = data_2d.Cd[1:].astype('float64')
    alpha = data_2d.Alpha[1:]
    ax.plot(alpha, Cd, **style_2d, label='2d')
    ax.legend()

def plot_polar():
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    fig.set_dpi(120)
    ax.grid(True, alpha=1, lw=0.5, ls='--')
    ax.axhline(0, c='k', ls='-', lw=0.5)
    ax.axvline(0, c='k', ls='-', lw=0.5)
    CL = data_3d.CL[1:11].astype('float64')
    CD = data_3d.CD[1:11].astype('float64')
    p = polyfit(CL[:10], CD[:10], [2, 0])
    r2 = r2_score(CD[:10], polyval(CL[:10], p))
    CL_fit = np.linspace(CL.min(), CL.max())
    CD_fit = p[2]*CL_fit**2 + p[0]
    fit_label = f'$C_D = {p[2]:.2}C_L^2 + {p[0]:.2}$ \n$r^2 = ${r2:.3}'
    ax.plot(CD, CL, **style_3d, label='3d')
    ax.plot(CD_fit, CL_fit, ls='--', c='tab:blue', label=fit_label)
    Cl = data_2d.Cl[1:-16].astype('float64')
    Cd = data_2d.Cd[1:-16].astype('float64')
    ax.plot(Cd, Cl, **style_2d, label='2d')
    ax.legend()
    AR = 5.345
    print(f'e = {1/(p[2]*np.pi*AR):.2}')


# %%
plot_polar()
# %%
1/(np.pi * 5.345)
# %%
