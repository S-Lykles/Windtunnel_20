#%%
from pathlib import Path
import re

#%%
p = Path('Numerical data 3d balance')
#%%
with open(p/'corr_test.txt', 'r') as file:
    a = file.read()
    b = re.sub('\t\s*\t', '', a)
    with open(p/'corr_test(1).txt', 'w') as file:
        file.write(b)

# %%
