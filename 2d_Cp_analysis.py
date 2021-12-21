#%%
import numpy as np
from numpy.lib.npyio import save
import pandas as pd
from pathlib import Path
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
# %%
p = Path('Experimental data 2d')

# %%
# data is sepperated by tabs, of skipinitialspace is not set to True all the collumns will have 
# spaces in them, for example '     Alpha' instead of 'Alpha', same for the indices.
# indices are strings, so if you want to select by runnumber corr_data.loc['1']
data_2d = pd.read_csv(p/'corr_test.txt', sep='\t', index_col=0, skipinitialspace=True)
data_2d.rename({'/': 'info'}, inplace=True)
data_2d.loc['info', 'Alpha'] = np.nan
data_2d['Alpha'] = data_2d['Alpha'].astype('float64')
columns = data_2d.columns

Cpu_idx = [column for column in columns if 'Cpu' in column]
Cpl_idx = [column for column in columns if 'Cpl' in column]
# Cpu_idx = columns.map(lambda idx: True if 'Cpu' in idx else False)
# Clu_idx = columns.map(lambda idx: True if 'Clu' in idx else False)

# %%
Cpu = data_2d[['Alpha']+Cpu_idx]
Cpl = data_2d[['Alpha']+Cpl_idx]

style_2d = dict(marker='o', linestyle='-', markersize=5,
                           color='black',
                           markerfacecolor='tab:green',
                           markerfacecoloralt='red',
                           markeredgecolor='black')
# %%
def Cp_3d(Cp_pd):
    x = Cp_pd.loc['info', Cp_pd.columns[1:]]
    y = Cp_pd.Alpha[1:]
    z = Cp_pd.iloc[1:, 1:]
    x, y = np.meshgrid(x, y)
    return x, y, z

def plot_3d_cp(Cp_pd):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(*Cp_3d(Cp_pd), cmap='plasma_r')
    ax.invert_zaxis()
    fig.set_dpi(200)

def Cp_2d(Cp_pd, alpha):
    x = Cp_pd.loc['info', Cp_pd.columns[1:]]
    y = Cp_pd[Cp_pd.Alpha==alpha].iloc[0, 1:]
    return x, y

def highlight_region(ax, x, y1, y2, color1, color2):
    positive = (y1 >= y2)
    p_begin = positive[0]
    x_lst = [x[0]]
    y1_lst = [y1[0]]
    y2_lst = [y2[0]]
    for b_i, x_i, y1_i, y2_i in zip(positive, x, y1, y2):
        x_lst.append(x_i)
        y1_lst.append(y1_i)
        y2_lst.append(y2_i)
        if b_i != p_begin:
            if p_begin == True:
                color = color1
            else:
                color = color2
            ax.fill_between(x_lst, y1_lst, y2_lst, color=color, alpha=0.3)
            p_begin = b_i
            x_lst = [x_i]
            y1_lst = [y1_i]
            y2_lst = [y2_i]
            
    if p_begin == True:
        color = color1
    else:
        color = color2
    ax.fill_between(x_lst, y1_lst, y2_lst, color=color, alpha=0.3)

def plot_2d_cp(Cpu, Cpl, alpha, CL, CD, V):
    """
    Cpu and Cpl are tuples of (x, Cp) where x and Cp are some kind of list/array,
    x should be in percentage of chord 
    alpha is angle of attack in degrees.
    """
    p_color = 'coral'
    s_color = 'deepskyblue'
    x = np.linspace(0, 100, 1000)
    y_up = interp1d(*Cpu)(x)
    y_low = interp1d(*Cpl)(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.fill_between(x, y_up, y_low, alpha=0.3)
    highlight_region(ax, x, y_up, y_low, p_color, s_color)
    # ax.scatter(*Cpu, c=s_color, s=4)
    ax.plot(*Cpu, c=s_color, lw=1, ls='-', label='upper surface', marker='o', markersize=2)
    # ax.scatter(*Cpl, c=p_color, s=4)
    ax.plot(*Cpl, c=p_color, lw=1, ls='-', label='lower surface', marker='o', markersize=2)
    ax.axhline(0, c='k', ls='-', lw=0.5)
    ax.axvline(0, c='k', ls='-', lw=0.5)
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, which='major', alpha=1, lw=0.5, ls='-')
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.5, lw=0.3, ls='-')
    ax.set_xlabel('x/c [%]')
    ax.set_ylabel('$C_p$ [-]')
    # ax.set_title(f'$\\alpha = {alpha} \degree$, $C_l = {CL}$, $C_d = {CD:2.2e}$, $V = {V}$ [m/s]')
    # ax.set_title(f'$\\alpha = {alpha} \degree$')
    fig.set_dpi(120)

# plot_3d_cp(Cpu)


# %%
for idx in data_2d.Alpha.index[4:17:4]:
    alpha = data_2d.loc[idx, 'Alpha']
    V = data_2d.loc[idx, 'V']
    Cl = data_2d.loc[idx, 'Cl']
    Cd = data_2d.loc[idx, 'Cd']
    plot_2d_cp(Cp_2d(Cpu, alpha), Cp_2d(Cpl, alpha), alpha, float(Cl), float(Cd), float(V))
    plt.savefig(f'PLots/2d_Cp_plots/2d_Cp_{alpha:.3}.pdf')
    

# %%
def plot_Cl_a(savefig=False):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    fig.set_dpi(120)
    ax.grid(True, which='major', alpha=1, lw=0.5, ls='-')
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.5, lw=0.3, ls='-')
    ax.axhline(0, c='k', ls='-', lw=0.7)
    ax.axvline(0, c='k', ls='-', lw=0.7)
    ax.set_xlabel('$\\alpha$ [deg]')
    ax.set_ylabel('$C_l$ [-]')
    
    alpha = data_2d.Alpha[1:].astype('float64')
    Cl = data_2d.Cl[1:].astype('float64')
    ax.plot(alpha, Cl, **style_2d, label='2D experimental')
    ax.legend()
    if savefig:
        fig.savefig('Plots/lift_curve_2d.pdf')

def plot_polar(savefig=False):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    fig.set_dpi(120)
    ax.grid(True, which='major', alpha=1, lw=0.5, ls='-')
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.5, lw=0.3, ls='-')
    ax.axhline(0, c='k', ls='-', lw=0.7)
    ax.axvline(0, c='k', ls='-', lw=0.7)
    ax.set_xlabel('$C_d$ [-]')
    ax.set_ylabel('$C_l$ [-]')
    ax.set_xlim([0, 0.07])
    ax.set_ylim([-0.3, 1.05])
    
    Cd = data_2d.Cd[1:].astype('float64')
    Cl = data_2d.Cl[1:].astype('float64')
    ax.plot(Cd, Cl, **style_2d, label='2D experimental')
    ax.legend(loc='lower right')
    if savefig:
        fig.savefig('Plots/drag_polar_2d.pdf')

# %%
plot_Cl_a(True)
# %%
plot_polar(True)

# %%
