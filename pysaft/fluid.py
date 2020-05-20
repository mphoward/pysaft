"""Standard reference equations of state."""

import numpy as np

from . import core

class IdealGas(core.FreeEnergy):
    def __init__(self):
        super().__init__()

    def f(self, n):
        r"""Free-energy density (per kT).

        Args:
            n (float or list): Number densities of each type.

        The ideal-gas free energy is:

        ..math::

            \beta f = \sum n_i (\ln n_i - 1)

        where :math:`\beta = k_{\rm B} T`. The thermal de Broglie wavelengths
        (quantum contribution to the classical free energy) are implicitly set
        equal to 1.

        """
        n = np.atleast_1d(n).astype(np.float64)
        return np.sum(n*(np.log(n)-1.))

    def z(self, n):
        return 1.0

    def mu(self, n, i):
        return np.log(n)

    def G(self, n, i, j):
        return 1.0

    def lnG(self, n, i, j):
        return 0.0

    def dG(self, n, i, j, k):
        return 0.0

    def dlnG(self, n, i, j, k):
        return 0.0

class HardSphere(core.FreeEnergy):
    """Boublik's equation of state for a hard-sphere mixture.

    Args:
        d (list): Diameters of each sphere type.

    """
    def __init__(self, d):
        super().__init__()
        self.d = d

    @property
    def d(self):
        """Diameters."""
        return self._d

    @d.setter
    def d(self, d):
        self._d = np.atleast_1d(d).astype(np.float64)

    def f(self, n):
        return IdealGas().f(n) + self.fex(n)

    def f_ex(self, n):
        xi = self._xi(n)
        return (6./np.pi)*((xi[2]**3/xi[3]**2-xi[0])*np.log(1.-xi[3])
                          + 3.*xi[1]*xi[2]/(1.-xi[3])
                          + xi[2]**3/(xi[3]*(1.-xi[3])**2))

    def z(self, n):
        xi = self._xi(n)
        ntot = np.sum(n)
        return (6./(np.pi*ntot))*(xi[0]/(1.-xi[3]) + 3.*xi[1]*xi[2]/(1.-xi[3])**2
                                 + (3.-xi[3])*xi[2]**3/(1.-xi[3])**3)

    def mu(self, n, i):
        return IdealGas().mu(n,i) + self.muex(n,i)

    def mu_ex(self, n, i):
        xi = self._xi(n)
        _13 = 1.-xi[3]
        _log13 = np.log(_13)

        f0 = -_log13
        f1 = 3.*xi[2]/_13
        f2 = 3.*(xi[1]/_13 + (xi[2]/xi[3])**2*_log13 + xi[2]**2/(_13**2*xi[3]))
        f3 = (3.*xi[1]*xi[2]/_13**2 - (xi[2]**3/xi[3]**2-xi[0])/_13 - 2.*(xi[2]/xi[3])**3*_log13
             -xi[2]**3/(xi[3]**2*_13**2) + 2.*xi[2]**3/(xi[3]*_13**3))

        di = self.d[i]
        return f0 + f1*di + f2*di**2 + f3*di**3

    def G(self, n, i, j):
        ri = 0.5*self.d[i]
        rj = 0.5*self.d[j]
        rij = ri*rj/(ri+rj)

        xi = self._xi(n)

        return (1./(1.-xi[3]) + 6.*(rij*xi[2])/(1.-xi[3])**2
               + 8.*(rij*xi[2])**2/(1.-xi[3])**3)

    def lnG(self, n, i, j):
        return np.log(self.G(n,i,j))

    def dG(self, n, i, j, k):
        return self.G(n,i,j)*self.dlnG(n,i,j,k)

    def dlnG(self, n, i, j, k):
        di = self.d[i]
        dj = self.d[j]

        xi = self._xi(n)
        dxi = self._dxi(n)

        A = di+dj+di*dj*xi[2]-(di+dj)*xi[3]
        B = di+dj+2.*di*dj*xi[2]-(di+dj)*xi[3]
        f2 = di*dj*(1./A+2./B)
        f3 = 3./(1.-xi[3])-(di+dj)*(1./A+1./B)

        return f2*dxi[2] + f3*dxi[3]

    def _xi(self, n):
        r"""Fundamental measures of the sphere mixture.

        Args:
            n (float or list): Number densities of each sphere type.

        The auxilliary values for this free energy are related to fundamental
        measures of the sphere for :math:`0 \le m \le 3`.

        .. math::

            \xi_m = \sum \frac{n_i \pi d_i^m}{6}

        For example, :math:`m=3` is the volume fraction of the mixture.
        The dimensions of :math:`\xi_m` are :math:`d^(m-3)` since *n*
        is a density.

        """
        n = np.atleast_1d(n).astype(np.float64)
        return np.array([np.sum(n*np.pi*self.d**m/6.) for m in range(4)], dtype=np.float64)

    def _dxi(self, n, i):
        return np.array([np.pi*self.d[i]**m/6. for m in range(4)], dtype=np.float64)

class HardChain(core.FreeEnergy):
    """Chapman-Jackson-Gubbins equation of state for a homonuclear hard-chain mixture.

    Args:
        d (float or list): Diameters of beads in each chain type.
        M (int or list): Number of beads in each chain type.

    """
    def __init__(self, d, M):
        super().__init__()
        self.d = d
        self.M = M

    @property
    def d(self):
        """Bead diameters for each chain."""
        return self._d

    @d.setter
    def d(self, d):
        self._d = np.atleast_1d(d).astype(np.float64)

    @property
    def M(self):
        """Number of beads per chain."""
        return self._M

    @M.setter
    def M(self, M):
        self._M = np.atleast_1d(M)

    def f(self, n):
        return IdealGas().f(n) + self.f_ex(n)

    def f_ex(self, n):
        # excess hard-sphere free energy
        hs,nhs = self.as_hs(n)
        f_hs = hs.f_ex(nhs)

        # free energy to bond hard beads in a chain
        f_bond = np.sum([-n[i]*(self.M[i]-1.)*hs.lnG(nhs,i,i) for i in range(len(n))])

        return f_hs + f_bond

    def mu(self, n, i):
        return IdealGas().mu(n,i) + self.mu_ex(n,i)

    def mu_ex(self, n, i):
        # excess hard-sphere chemical potential
        Mi = self.M[i]
        hs,nhs = self.as_hs(n)
        mu_hs = Mi*hs.mu_ex(nhs,i)

        # chain
        mu_bond = (-(Mi-1.)*hs.lnG(nhs,i,i) +
                   Mi*np.sum([-n[j]*(self.M[j]-1.)*hs.dlnG(nhs,j,j,i) for j in range(len(n))]))

        return mu_hs + mu_bond

    def as_hs(self, n):
        n = np.atleast_1d(n).astype(np.float64)
        return HardSphere(self.d), n*self.M

class PolyatomicHardChain(core.FreeEnergy):
    """Amos and Jackson equation of state for polyatomic hard chains.

    Args:
        d (float or list): Diameters of each **bead** type.
        polymers (Polymer or list): Polymer chains in mixture.

    Unlike the HardChain, this free energy does not assume that all beads
    are the same within a chain. Hence, the polymer bead types and
    bonding topology need to be defined for each chain as a Polymer
    object. The types in the Polymer should map to the indexes of the
    bead diameters *d*.

    """
    def __init__(self, d, polymers):
        super().__init__()
        self.d = d
        self.polymers = polymers

    @property
    def d(self):
        """Bead diameters for each type."""
        return self._d

    @d.setter
    def d(self, d):
        self._d = np.atleast_1d(d).astype(np.float64)

    @property
    def polymers(self):
        return self._polymers

    @polymers.setter
    def polymers(self, polymers):
        try:
            self._polymers = tuple(polymers)
        except TypeError:
            self._polymers = (polymers,)

    def f(self, n):
        return IdealGas().f(n) + self.f_ex(n)

    def f_ex(self, n):
        # excess hard-sphere free energy
        hs,nhs = self.as_hs(n)
        f_hs = hs.f_ex(nhs)

        # bonding contributions
        f_bond = 0.
        for n_,p in zip(n,self.polymers):
            for (i,j),N_b in p.count_bonds().items():
                f_bond -= n_ * N_b * hs.lnG(nhs,i,j)

        return f_hs + f_bond

    def mu(self, n, i):
        return IdealGas().mu(n,i) + self.mu_ex(n,i)

    def mu_ex(self, n, i):
        # excess hard-sphere chemical potential for this polymer
        hs,nhs = self.as_hs(n)
        pi = self.polymers[i]
        mu_hs = 0.
        for t,N_t in pi.count_types().items():
            mu_hs += N_t*hs.mu_ex(nhs,t)

        # bond terms within this polymer
        mu_bond = 0.
        for (j,k), N_b in pi.count_bounds().items():
            mu_bond -= N_b*hs.lnG(nhs,j,k)

        # other bond terms from all polymers
        for n_,p in zip(n,self.polymers):
            for (j,k),N_b in p.count_bonds().items():
                for t,N_t in pi.count_types().items():
                    mu_bond -= N_t*n_*N_b*hs.dlnG(nhs,j,k,t)

        return mu_hs + mu_bond

    def as_hs(self, n):
        num_types = len(self.d)
        nhs = np.zeros(num_types, dtype=np.float64)
        for n_,p in zip(n,self.polymers):
            for i,N_i in p.count_types().items():
                if i >= num_types:
                    raise KeyError('Bead index {} outside range [0,{}]'.format(i, num_types))
                nhs[i] += n_*N_i
        return HardSphere(self.d),nhs

class AsakuraOosawa(core.FreeEnergy):
    """Asakura-Oosawa model for a colloid--polymer mixture.

    Args:
        d (list): Diameters of colloids (0) and polymers (1).

    The colloids are modeled as a hard sphere fluid, while the
    the polymers are an ideal solution that is excluded from (and
    does not adsorb onto) the colloids.

    """
    def __init__(self, d):
        super().__init__()
        self.d = d

    @property
    def d(self):
        """Diameters."""
        return self._d

    @d.setter
    def d(self, d):
        if len(d) != 2:
            raise TypeError('Diameters must have exactly 2 components')
        self._d = np.atleast_1d(d).astype(np.float64)

    def f(self, n):
        return IdealGas().f(n) + self.f_ex(n)

    def f_ex(self, n):
        n = np.atleast_1d(n)
        if len(n) != 2:
            raise TypeError('Densities must have exactly 2 components')

        # excess for colloids
        hs = HardSphere(self.d[0])
        fhs = hs.f_ex(n[0])

        # excess for polymers
        lam_ = self.d[1]/self.d[0]
        A = 3*lam_ + 3*lam_**2 + lam_**3
        B = 9*lam_**2/2 + 3*lam_**3
        C = 3*lam_**3
        eta = n[0]*np.pi*self.d[0]**3/6.
        gamma = eta/(1.-eta)
        fao = n[1]*(A*gamma + B*gamma**2 + C*gamma**3 - np.log(1.-eta))

        return fhs + fao

    def G(self, n, i, j):
        """Radial distribution function.

        Only the cross-term is implemented using the approximation of Santos.

        """
        n = np.atleast_1d(n)
        if len(n) != 2:
            raise TypeError('Densities must have exactly 2 components')

        if i == j:
            raise NotImplementedError('Only cross-term is implemented')

        eta = n[0]*np.pi*self.d[0]**3/6.
        lam_ = self.d[1]/self.d[0]
        return 1./(1.-eta) + 2.*lam_/(1.+lam_)*((1.-eta/2.)/(1.-eta)**3-1./(1.-eta))
