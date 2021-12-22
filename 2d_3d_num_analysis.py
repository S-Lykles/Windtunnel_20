
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
p_num = Path('Numerical data 3d')
data_3d_num = pd.read_csv(p_num/'3D data_LLT/T1-54.4 m s-LLT-x0.000mm.txt', sep=' ', header=4, skipinitialspace=True)
data_3d_num = data_3d_num.drop(0).apply(pd.to_numeric)

p_exp = Path('Experimental data 3d balance')
data_3d_exp = pd.read_csv(p_exp/'corr_test(1).txt', sep='\t', index_col=0, skipinitialspace=True)
data_3d_exp = data_3d_exp.drop('/').apply(pd.to_numeric)
# %%

style_3d_num = dict(marker='>', linestyle='-', markersize=5,
                           color='black',
                           markerfacecolor='skyblue',
                           markerfacecoloralt='red',
                           markeredgecolor='black')

style_3d_exp = dict(marker='v', linestyle='-', markersize=5,
                           color='black',
                           markerfacecolor='tab:red',
                           markerfacecoloralt='red',
                           markeredgecolor='black')
# %%
def plot_Cl_a(regression=False, savefig=False):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    fig.set_dpi(120)
    ax.grid(True, which='major', alpha=1, lw=0.5, ls='-')
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.5, lw=0.3, ls='-')
    ax.axhline(0, c='k', ls='-', lw=0.7)
    ax.axvline(0, c='k', ls='-', lw=0.7)
    ax.set_xlabel('$\\alpha$ [deg]')
    ax.set_ylabel('$C_L$ [-]')
    # --3d num--
    CL = data_3d_num.CL
    alpha = data_3d_num.alpha
    ax.plot(alpha, CL, **style_3d_num, label='3D Numerical')
    if regression:
        n_terms = 13
        slope_3d, y0, r, _, _ = linregress(alpha[:n_terms], CL[:n_terms])
        line_terms = 30
        CL_fit = slope_3d * alpha[:line_terms] + y0
        ax.plot(alpha[:line_terms], CL_fit, c='skyblue', ls='-.', label='3D numerical regression line')
        # This is to annotate the regression lines, needs to be placed manually
        ax.annotate(f'$C_L = {slope_3d:.3} \cdot \\alpha {y0:+.3} $\n $r^2 = {r**2:.4}$', (0.53, 0.61))
        print(f'3d slope is {slope_3d*180/(np.pi**2):.3} pi [1/rad]')
    # --3d exp--
    CL = data_3d_exp.CL
    alpha = data_3d_exp.Alpha
    ax.plot(alpha, CL, **style_3d_exp, label='3D experimental')
    
    if regression:
        n_terms = 10
        slope_2d, y0, r, _, _ = linregress(alpha[:n_terms], CL[:n_terms])
        line_terms = 30
        CL_fit = slope_2d * alpha[:line_terms] + y0
        ax.plot(alpha[:line_terms], CL_fit, c='tab:red', ls='--', label='3D experimental regression line')
        ax.annotate(f'$C_L = {slope_2d:.3} \cdot \\alpha {y0:+.3} $\n $r^2 = {r**2:.4}$', (7.5, 0.35))
    # print(f'2d slope is {slope_2d*180/(np.pi**2):.3} pi [1/rad]')
    # slope_2d = np.rad2deg(slope_2d)
    # slope_3d = np.rad2deg(slope_3d)
    # slope_3d_theory = slope_2d/(1+slope_2d/(np.pi*AR))
    # print(f'lifting line theory {slope_3d_theory/np.pi:.4} pi [1/rad]')
    # slope_3d_theory = slope_2d/(np.sqrt(1+(slope_2d/(np.pi*AR))**2)+slope_2d/(np.pi*AR))
    # print(f'hemold = {slope_3d_theory/np.pi:.4} pi [1/rad]')
    # print(slope_3d/slope_2d)
    ax.legend(loc='lower right')

    file_label = ''
    if regression:
        file_label = '_regression'
    if savefig:
        fig.savefig('Plots/lift_curve_3d_exp_num'+file_label+'.pdf')

# %%
plot_Cl_a(regression=True)
# %%
