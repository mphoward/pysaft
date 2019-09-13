import sys
sys.path.append('..')
import numpy as np

import pysaft

# test hard spheres
deta = 0.01
etas = np.arange(deta, 0.45 + 0.5*deta, deta)
with open('hs.dat','w') as f:
    for eta in etas:
        rho = 6.*eta/np.pi
        hs = pysaft.eos.HardSphere(1.)
        f.write("{:.2f} {:.5f}\n".format(eta,hs.z(rho)))

# test hard chains
with open('hc.dat','w') as f:
    Ms = (1,2,4,8,16)
    for eta in etas:
        row = [eta]
        for M in Ms:
            rho = 6.*eta/(M*np.pi)
            hc = pysaft.eos.HardChain(1.,M)
            row.append(hc.z(rho))

        f.write(("{:.2f}" + " {:.5f}"*len(Ms) + "\n").format(*row))

# test hard chains with amos
with open('hc_poly.dat','w') as f:
    Ms = (1,2,4,8,16)
    for eta in etas:
        row = [eta]
        for M in Ms:
            # polymer topology
            idx = np.arange(M)
            p = pysaft.Polymer(types=[0]*M, bonds=np.column_stack((idx[:-1],idx[1:])))

            rho = 6.*eta/(M*np.pi)
            hc = pysaft.eos.PolyatomicHardChain(d=[1.], polymers=p)
            row.append(hc.z(rho))

        f.write(("{:.2f}" + " {:.5f}"*len(Ms) + "\n").format(*row))
