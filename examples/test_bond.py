import sys
sys.path.append('..')
import numpy as np

import pysaft

v = 4.*np.pi/3.
bond = pysaft.bond.ContactBond(num_patch=[6,2], bond_volume=[[0,v],[v,0]], G=lambda n,i,j : 1.0)
X = bond.X([0.1,0.6])
print(X)
fex = bond.f_ex([0.1,0.6])
print(fex)
