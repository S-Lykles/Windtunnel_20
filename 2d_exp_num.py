#%%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
from matplotlib import pyplot as plt
# %%
p = Path('Experimental_data')

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

# %%
Cpu = corr_data[['Alpha']+Cpu_idx]
Clu = corr_data[['Alpha']+Cpl_idx]

x_u = Cpu.loc['info', Cpu.columns[1:]]
x_l = Clu.loc['info', Clu.columns[1:]]


filled_blue_circ_marker= dict(marker='o', linestyle='-', markersize=6,
                           color='black',
                           markerfacecolor='tab:blue',
                           markerfacecoloralt='red',
                           markeredgecolor='black')

filled_green_trig_marker= dict(marker='^', linestyle='-', markersize=6,
                           color='black',
                           markerfacecolor='tab:green',
                           markerfacecoloralt='red',
                           markeredgecolor='black')

z_u = Cpu.iloc[1:, 1:]
z_l = Clu.iloc[1:, 1:]

alfa = "3"


plt.plot(x_u, z_u.loc['6'],  **filled_blue_circ_marker, label = "Experiment, upper side")
plt.plot(x_l, z_l.loc['6'],  **filled_green_trig_marker, label = "Experiment, lower side")
plt.grid()
plt.title("Pressure distribution")
plt.xlabel("x/c [%]")
plt.ylabel("Cp [-]")
plt.gca().invert_yaxis()
# %%
cp_v = pd.read_csv("cp_"+alfa+"_v.dat", header = None, sep = " ", decimal = ".", names = ["x", "y", "Cp"], skiprows = 3, skipinitialspace= True)
plt.plot(cp_v.x*100, cp_v.Cp, color = "tab:red", label = "XFoil, viscous")
cp_i = pd.read_csv("cp_"+alfa+"_i.dat", header = None, sep = " ", decimal = ".", names = ["x", "y", "Cp"], skiprows = 3, skipinitialspace= True)
#plt.plot(cp_i.x*100, cp_i.Cp, linestyle = "--", label = "Inviscid")
plt.legend()
# %%
plt.savefig("Cp_"+alfa+"_v.pdf")
plt.show()

"""cf_u = pd.read_csv("cf_"+alfa+"_u.dat", header = None, sep = " ", decimal = ".", names = ["x", "cf"], skiprows = 8, skipinitialspace= True)
cf_l = pd.read_csv("cf_"+alfa+"_l.dat", header = None, sep = " ", decimal = ".", names = ["x", "cf"], skiprows = 8, skipinitialspace= True)
plt.grid()
plt.xlim(0, 101)
plt.title("Skin friction coefficient")
plt.xlabel("x/c [%]")
plt.ylabel("Cf [-]")
plt.plot(cf_u.x*100, cf_u.cf,  linestyle = "--", color = "tab:red", label = "Upper side")
plt.plot(cf_l.x*100, cf_l.cf,  linestyle = ":", color = "tab:blue", label = "Lower side")
plt.legend()
plt.savefig("Cf_"+alfa+".pdf")
plt.show()"""
