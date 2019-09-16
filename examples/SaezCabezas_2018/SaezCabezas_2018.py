import sys
sys.path.append('../..')

import numpy as np
import numdifftools as nd

import pysaft

# experimental parameters
dc = 2*2.83 # nm
dp = 2*0.98 # nm
v = 0.181*dc**3 # nm^3
num_ads = 30 # number of binding sites on a NC
num_bind = 2 # number of NCs a polymer can bridge

# grid of eta and gamma to sweep
deta = 0.005
etas = np.arange(0.005,0.3+0.5*deta,deta)
gammas = np.logspace(start=0,stop=4,num=50,base=10)

# tabulate the stability of the states (> 0 is stable, < 0 is unstable)
file_ = open('stability.dat','w')
file_.write('# eta gamma sign(det(H))\n')
for i,eta in enumerate(etas):
    nc = 6.*eta/(np.pi*dc**3)
    for j,gamma in enumerate(gammas):
        n = [nc, gamma*nc]

        # AO mixture + bonding
        ao = pysaft.fluid.AsakuraOosawa(d=[dc,dp])
        bond = pysaft.bond.ContactBond(num_patch=[num_ads,num_bind], bond_volume=[[0,v],[v,0.]], G=ao.G)
        f = ao + bond

        # get sign of the determinant of the stability matrix
        step = nd.MaxStepGenerator(base_step=1.e-5, step_ratio=2, num_steps=4)
        h = nd.Hessian(f, step=step)(n)
        s,logdet = np.linalg.slogdet(h)

        file_.write('{:.5f} {:.5f} {}\n'.format(eta, gamma, int(s)))
file_.close()
