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
corr_data.loc['info', 'Alpha'] = np.nan
corr_data['Alpha'] = corr_data['Alpha'].astype('float64')
columns = corr_data.columns

Cpu_idx = [column for column in columns if 'Cpu' in column]
Cpl_idx = [column for column in columns if 'Cpl' in column]
# Cpu_idx = columns.map(lambda idx: True if 'Cpu' in idx else False)
# Clu_idx = columns.map(lambda idx: True if 'Clu' in idx else False)

# %%
Cpu = corr_data[['Alpha']+Cpu_idx]
Clu = corr_data[['Alpha']+Cpl_idx]

# %%
z, y, x = Cpu.loc['1', Cpu.columns[1:]], Cpu.loc['1', 'Alpha'], Cpu.loc['info', Cpu.columns[1:]]
# %%
y = np.broadcast_to(y, np.shape(z))
# %%
from mpl_toolkits.mplot3d import Axes3D
# %%
from mpl_toolkits.mplot3d import Axes3D
# %%
def Cp_3d(row, info):
    z, y, x = row[1:], row[0], info[1:]
    y = np.broadcast_to(y, np.shape(z))
    return x, y, z

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for index in Cpu.index[1:37]:
    row = Cpu.loc[index]
    ax.plot(*Cp_3d(row, Cpu.loc['info']))

ax.set_zlim([1, -5])
fig.set_dpi(200)
# %%
# %%
# x = Cpu.loc['info', Cpu.columns[1:]]
# y = Cpu.Alpha[1:]
# z = Cpu.iloc[1:, 1:]
# x, y = np.meshgrid(x, y)

# fig = plt.figure()
# fig.set_dpi(200)
# ax = fig.add_subplot(111, projection='3d')

# # ax.plot_surface(x, y, z, cmap='rainbow_r')
# ax.set_zlim([-5, 3])

x = Cpu.loc['info', Cpu.columns[1:]]
y = Cpu.Alpha[1:]
z = Cpu.iloc[1:, 1:]
# x, y = np.meshgrid(x, y)
# ax.plot_surface(x, y, z, cmap='plasma')
# %%
plt.plot(x, z.loc['20'])
# %%
z
# %%
