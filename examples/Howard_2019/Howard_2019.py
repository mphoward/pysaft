import sys
sys.path.append('../..')

import numpy as np
import numdifftools as nd
import scipy.interpolate

import pysaft

# state point
gamma = 1.5
M = 8
dp = 1.0
dc = 5.0

# grid of eta and epsilon to sweep
deta = 0.005
deb = 0.25
etas = np.arange(0.005,0.15+0.5*deta,deta)
ebs = np.arange(10.,20.+0.5*deb,deb)

# bond volume data (already computed for this state point)
bv_ = np.loadtxt('bond_integral.dat')
bv = scipy.interpolate.InterpolatedUnivariateSpline(bv_[:,0], bv_[:,1])

# tabulate the stability of the states (> 0 is stable, < 0 is unstable)
file_ = open('stability_{}.dat'.format(M),'w')
file_.write('# eta epsilon sign(det(H))\n')
for i,eta in enumerate(etas):
    nc = 6.*eta/(np.pi*dc**3)
    n = [nc, nc*gamma]

    for j,eb in enumerate(ebs):
        # hard chain + bonding
        hc = pysaft.eos.HardChain(d=[dc,dp],M=[1,M])
        def G(n,i,j):
            hs,nhs = hc._as_hs(n)
            return hs.G(nhs,i,j)
        v = bv(eb)
        bond = pysaft.bond.ContactBond(num_patch=[6,2], bond_volume=[[0,v],[v,0]], G=G)
        f = hc + bond

        # get sign of the determinant of the stability matrix
        step = nd.MinStepGenerator(base_step=1.e-7, step_ratio=2, num_steps=4)
        h = nd.Hessian(f, step=step)(n)
        s,logdet = np.linalg.slogdet(h)

        file_.write('{:.3f} {:.3f} {}\n'.format(eta, eb, int(s)))
file_.close()
