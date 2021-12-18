#%%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
# %%
p = Path('Numerical data 2d')

# %%
# data is sepperated by tabs, of skipinitialspace is not set to True all the collumns will have 
# spaces in them, for example '     Alpha' instead of 'Alpha', same for the indices.
# indices are strings, so if you want to select by runnumber corr_data.loc['1']
corr_data = pd.read_csv(p/'corr_test.txt', sep='\t', index_col=0, skipinitialspace=True)
corr_data.rename({'/': 'info'}, inplace=True)
corr_data.loc['info', 'Alpha'] = np.nan
corr_data['Alpha'] = corr_data['Alpha'].astype('float64')
columns = corr_data.columns

Cpu_idx = [column for column in columns if 'Cpu' in column]
Cpl_idx = [column for column in columns if 'Cpl' in column]
# Cpu_idx = columns.map(lambda idx: True if 'Cpu' in idx else False)
# Clu_idx = columns.map(lambda idx: True if 'Clu' in idx else False)

# %%
Cpu = corr_data[['Alpha']+Cpu_idx]
Cpl = corr_data[['Alpha']+Cpl_idx]

# %%
z, y, x = Cpu.loc['1', Cpu.columns[1:]], Cpu.loc['1', 'Alpha'], Cpu.loc['info', Cpu.columns[1:]]
# %%
y = np.broadcast_to(y, np.shape(z))
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

def plot_2d_cp(Cpu, Cpl, alpha):
    """
    Cpu and Cpl are tuples of (x, Cp), alpha is angle of attack in degrees.
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
    ax.scatter(*Cpu, c=s_color, s=4)
    ax.plot(*Cpu, c=s_color, lw=0.8, ls='-', label='upper surface')
    ax.scatter(*Cpl, c=p_color, s=4)
    ax.plot(*Cpl, c=p_color, lw=0.8, ls='-', label='lower surface')
    ax.axhline(0, c='k', ls='-', lw=0.5)
    ax.axvline(0, c='k', ls='-', lw=0.5)
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=1, lw=0.5, ls='--')
    ax.set_xlabel('% of chord')
    ax.set_ylabel('$C_p$')
    ax.set_title(f'$\\alpha = {alpha} \degree$')
    fig.set_dpi(120)

# plot_3d_cp(Cpu)


# %%
for alpha in corr_data.Alpha[1:2]:
    # print(Cp_2d(Cpu, alpha)[1][0],Cp_2d(Cpl, alpha)[1][0])
    plot_2d_cp(Cp_2d(Cpu, alpha), Cp_2d(Cpl, alpha), alpha)
    
# %%
