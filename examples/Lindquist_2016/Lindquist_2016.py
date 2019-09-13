import sys
sys.path.append('../..')

import numpy as np
import numdifftools as nd

import pysaft

# state point
gamma = float(sys.argv[1])

# grid of eta and kT to sweep
deta = 0.0025
dkT = 0.0025
etas = np.arange(0.005,0.15+0.5*deta,deta)
kTs = np.arange(0.06,0.15+0.5*dkT,dkT)

# bond volume from Jackson, Chapman, and Gubbins
# (note -2 should be +2 in first term of eq. (9) to get eq. (11))
rc = 0.3
eAB = 1.0
K = (6.*(2.*rc-1.)*(1.+rc)**2*np.log(1.+rc)+rc*(6.-3.*rc-16.*rc**2))/36.

# tabulate the stability of the states (> 0 is stable, < 0 is unstable)
file_ = open('stability_{:.1f}.dat'.format(gamma),'w')
file_.write('# eta kT sign(det(H))\n')
for i,eta in enumerate(etas):
    # eta is the total packing fraction, so split between primary and linker particles
    ntot = 6.*eta/np.pi
    n = [ntot/(1.+gamma), ntot*gamma/(1.+gamma)]

    for j,kT in enumerate(kTs):
        # hard sphere + bonding
        hs = pysaft.eos.HardSphere(d=[1.,1.])
        v = 4.*np.pi*(np.exp(eAB/kT)-1.)*K
        bond = pysaft.bond.ContactBond(num_patch=[6,2], bond_volume=[[0,v],[v,0]], G=hs.G)
        f = hs + bond

        # get sign of the determinant of the stability matrix
        step = nd.MinStepGenerator(base_step=1.e-6, step_ratio=2, num_steps=4)
        h = nd.Hessian(f, step=step)(n)
        s,logdet = np.linalg.slogdet(h)

        file_.write('{:.5f} {:.5f} {}\n'.format(eta, kT, int(s)))
file_.close()
