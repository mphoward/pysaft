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

        # estimate compressibility from free energy
        eps = 1.e-6
        rho_eps = rho+eps
        z_f = rho * (hs.f(rho_eps)/rho_eps - hs.f(rho)/rho)/eps

        f.write("{:.2f} {:.5f} {:.5f}\n".format(eta,hs.z(rho),z_f))

# test hard chains
with open('hc.dat','w') as f:
    Ms = (1,2,4,8,16)
    for eta in etas:
        row = [eta]
        for M in Ms:
            rho = 6.*eta/(M*np.pi)
            hc = pysaft.eos.HardChain(M, 1.)

            # estimate compressibility from free energy
            eps = 1.e-6
            rho_eps = rho+eps
            z_f = rho * (hc.f(rho_eps)/rho_eps - hc.f(rho)/rho)/eps
            row.append(z_f)

        f.write(("{:.2f}" + " {:.5f}"*len(Ms) + "\n").format(*row))

# test hard chains with amos
with open('hc_poly.dat','w') as f:
    Ms = (1,2,4,8,16)
    for eta in etas:
        row = [eta]
        for M in Ms:
            # polymer topology
            idx = np.arange(M)
            p = pysaft.eos.Polymer(types=[0]*M, bonds=np.column_stack((idx[:-1],idx[1:])))

            rho = 6.*eta/(M*np.pi)
            hc = pysaft.eos.PolyatomicHardChain(polymers=p, d=[1.])

            # estimate compressibility from free energy
            eps = 1.e-6
            rho_eps = rho+eps
            z_f = rho * (hc.f(rho_eps)/rho_eps - hc.f(rho)/rho)/eps
            row.append(z_f)

        f.write(("{:.2f}" + " {:.5f}"*len(Ms) + "\n").format(*row))
