
#%%
from matplotlib.figure import Figure
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
from scipy.stats import linregress

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
style_2d = dict(marker='o', linestyle='-', markersize=5,
                           color='black',
                           markerfacecolor='tab:green',
                           markerfacecoloralt='red',
                           markeredgecolor='black')
style_3d = dict(marker='v', linestyle='-', markersize=5,
                           color='black',
                           markerfacecolor='tab:blue',
                           markerfacecoloralt='red',
                           markeredgecolor='black')

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
    ax.set_ylabel('$C_L$, $C_l$ [-]')
    # --3d--
    CL = data_3d.CL[1:].astype('float64')
    alpha = data_3d.Alpha[1:]
    ax.plot(alpha, CL, **style_3d, label='3d experimental')

    n_terms = 10
    slope, y0, r, _, _ = linregress(alpha[:n_terms], CL[:n_terms])
    line_terms = 20
    CL_fit = slope * alpha[:line_terms] + y0
    ax.plot(alpha[:line_terms], CL_fit, c='tab:blue', ls='--', label='3d regression line')
    ax.annotate(f'$C_L = {slope:.3} \cdot \\alpha {y0:+.3} $\n $r^2 = {r**2:.4}$', (7.5, 0.35))
    print(f'slope is {slope*180/(np.pi**2):.3} pi [1/rad]')
    # --2d--
    Cl = data_2d.Cl[1:].astype('float64')
    alpha = data_2d.Alpha[1:]
    ax.plot(alpha, Cl, **style_2d, label='2d experimental')

    n_terms = 10
    slope, y0, r, _, _ = linregress(alpha[:n_terms], Cl[:n_terms])
    line_terms = 20
    CL_fit = slope * alpha[:line_terms] + y0
    ax.plot(alpha[:line_terms], CL_fit, c='tab:green', ls='--', label='2d regression line')
    ax.annotate(f'$C_l = {slope:.3} \cdot \\alpha {y0:+.3} $\n $r^2 = {r**2:.4}$', (0.53, 0.61))
    print(f'slope is {slope*180/(np.pi**2):.3} pi [1/rad]')
    ax.legend(loc='lower right')
    if savefig:
        fig.savefig('Plots/lift_curve_2d_3d.pdf')

def plot_Cd_a(savefig=False):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    fig.set_dpi(120)
    ax.grid(True, which='major', alpha=1, lw=0.5, ls='-')
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.5, lw=0.3, ls='-')
    ax.axhline(0, c='k', ls='-', lw=0.7)
    ax.axvline(0, c='k', ls='-', lw=0.7)
    ax.set_xlabel('$\\alpha$ [deg]')
    ax.set_ylabel('$C_D$, $C_d$ [-]')
    # --3d--
    CD = data_3d.CD[1:].astype('float64')
    alpha = data_3d.Alpha[1:]
    ax.plot(alpha, CD, **style_3d, label='3d experimental')
    # --2d--
    Cd = data_2d.Cd[1:].astype('float64')
    alpha = data_2d.Alpha[1:]
    ax.plot(alpha, Cd, **style_2d, label='2d experimental')
    ax.legend()
    if savefig:
        fig.savefig('Plots/drag_curve_2d_3d.pdf')

def plot_Cm_a(savefig=False):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    fig.set_dpi(120)
    ax.grid(True, which='major', alpha=1, lw=0.5, ls='-')
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.5, lw=0.3, ls='-')
    ax.axhline(0, c='k', ls='-', lw=0.7)
    ax.axvline(0, c='k', ls='-', lw=0.7)
    ax.set_xlabel('$\\alpha$ [deg]')
    ax.set_ylabel('$C_M$, $C_m$ [-]')
    ax.set_ylim([-0.02, 0.042])
    # --3d--
    CM = data_3d.Cm_p_qc[1:].astype('float64')
    alpha = data_3d.Alpha[1:]
    ax.plot(alpha, CM, **style_3d, label='3d experimental')
    # --2d--
    Cm = data_2d.Cm[1:].astype('float64')
    alpha = data_2d.Alpha[1:]
    ax.plot(alpha, Cm, **style_2d, label='2d experimental')
    ax.legend()
    if savefig:
        fig.savefig('Plots/moment_curve_2d_3d.pdf')

def plot_polar(savefig=False):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    fig.set_dpi(120)
    ax.grid(True, which='major', alpha=1, lw=0.5, ls='-')
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.5, lw=0.3, ls='-')
    ax.axhline(0, c='k', ls='-', lw=0.7)
    ax.axvline(0, c='k', ls='-', lw=0.7)
    ax.set_ylabel('$C_D$, $C_d$ [-]')
    ax.set_xlabel('$C_L$, $C_l$ [-]')
    ax.set_xlim([0, 0.07])
    # --3d--
    CL = data_3d.CL[1:].astype('float64')
    max_alpha_3d = data_3d.Alpha[24]
    CD = data_3d.CD[1:].astype('float64')
    ax.plot(CD, CL, **style_3d, label='3d experimental')

    n_terms = 10
    p = polyfit(CL[:n_terms], CD[:n_terms], [2, 0])
    r2 = r2_score(CD[:n_terms], polyval(CL[:n_terms], p))
    CL_fit = np.linspace(CL.min(), CL.max())
    CD_fit = p[2]*CL_fit**2 + p[0]
    fit_label = f'3d regression curve'
    ax.plot(CD_fit, CL_fit, ls='--', c='tab:blue', label= fit_label)
    ax.annotate(f'$C_D = {p[2]:.3} \cdot C_L^2  {p[0]:+.3} $\n $r^2 = {r2:.4}$', (0.02, 0.61))
    # --2d--
    Cl = data_2d.Cl[1:].astype('float64')
    max_alpha_2d = data_2d.Alpha[24]
    # print(max_alpha_2d)
    Cd = data_2d.Cd[1:].astype('float64')
    ax.plot(Cd, Cl, **style_2d, label='2d experimental')
    ax.legend()
    AR = 5.345
    print(f'e = {1/(p[2]*np.pi*AR):.2}')
    if savefig:
        fig.savefig('Plots/drag_polar_2d_3d.pdf')

#%%
plot_polar()
#%%
plot_Cl_a()
#%%
def savefigs():
    plot_Cl_a(True)
    plot_Cd_a(True)
    plot_Cm_a(True)
    plot_polar(True)
#%%
savefigs()
# %%