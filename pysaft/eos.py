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
        return np.sum(n*(np.log(n)-1.))

    def f_ex(self, n):
        return 0.

    def z(self, n):
        return 1.

    def z_ex(self, n):
        return 0.

    def mu(self, n, i):
        return np.log(n[i])

    def mu_ex(self, n, i):
        return 0.

    def G(self, n, i, j):
        return 1.

class HardSphere(core.FreeEnergy):
    """Boublik's equation of state for a hard-sphere mixture.

    Args:
        d (list): Diameters of each sphere type.

    """
    def __init__(self, d):
        super().__init__()
        self._d = np.atleast_1d(d)

    @property
    def d(self):
        """Diameters."""
        return self._d

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
        n = np.atleast_1d(n)
        return np.array([np.sum(n*np.pi*self.d**m/6.) for m in range(4)])

    def f(self, n):
        return IdealGas().f(n) + self.f_ex(n)

    def f_ex(self, n):
        xi = self._xi(n)

        return (6./np.pi)*((xi[2]**3/xi[3]**2-xi[0])*np.log(1.-xi[3])
                          + 3.*xi[1]*xi[2]/(1.-xi[3])
                          + xi[2]**3/(xi[3]*(1.-xi[3])**2))

    def z(self, n):
        ntot = np.sum(n)
        xi = self._xi(n)

        return (6./(np.pi*ntot))*(xi[0]/(1.-xi[3])
                                 + 3.*xi[1]*xi[2]/(1.-xi[3])**2
                                 + (3.-xi[3])*xi[2]**3/(1.-xi[3])**3)

    def z_ex(self, n):
        return self.z(n) - 1.

    def G(self, n, i, j):
        ri = 0.5*self.d[i]
        rj = 0.5*self.d[j]
        rij = ri*rj/(ri+rj)

        xi = self._xi(n)

        return (1./(1.-xi[3]) + 6.*(rij*xi[2])/(1.-xi[3])**2
               + 8.*(rij*xi[2])**2/(1.-xi[3])**3)


class HardChain(core.FreeEnergy):
    """Chapman-Jackson-Gubbins equation of state for a homonuclear hard-chain mixture.

    Args:
        M (int or list): Number of beads in each chain type.
        d (float or list): Diameters of each beads in each chain type.

    """
    def __init__(self, M, d):
        super().__init__()
        self._M = np.atleast_1d(M)
        self._d = np.atleast_1d(d)

    @property
    def M(self):
        """Number of beads per chain."""
        return self._M

    @property
    def d(self):
        """Bead diameters for each chain."""
        return self._d

    def f(self, n):
        return IdealGas().f(n) + self.f_ex(n)

    def f_ex(self, n):
        n = np.atleast_1d(n)

        # reference hard-sphere free energy
        hs = HardSphere(self.d)
        # the total number of spheres of each type is n*M
        nhs = n*self.M
        fhs = hs.f_ex(nhs)

        # free energy to bond beads in a chain
        fbond = np.sum([-n[i]*(self.M[i]-1.)*np.log(hs.G(nhs,i,i)) for i in range(len(n))])

        return fhs + fbond

class PolyatomicHardChain(core.FreeEnergy):
    """Amos and Jackson equation of state for polyatomic hard chains.

    Args:
        polymers (Polymer or list): Polymer chains in mixture.
        d (float or list): Diameters of each **bead** type.

    Unlike the HardChain, this free energy does not assume that all beads
    are the same within a chain. Hence, the polymer bead types and
    bonding topology need to be defined for each chain as a Polymer
    object. The types in the Polymer should map to the indexes of the
    bead diameters *d*.

    """
    def __init__(self, polymers, d):
        super().__init__()
        self._d = np.atleast_1d(d)

        try:
            self._polymers = tuple(polymers)
        except TypeError:
            self._polymers = (polymers,)

    @property
    def polymers(self):
        return self._polymers

    @property
    def d(self):
        return self._d

    def f(self, n):
        return IdealGas().f(n) + self.f_ex(n)

    def f_ex(self, n):
        n = np.atleast_1d(n)

        # sum up the densities of each hs type
        num_types = len(self.d)
        nhs = np.zeros(num_types, dtype=np.float64)
        for n_,p in zip(n,self.polymers):
            for i,N_i in p.count_types().items():
                if i >= num_types:
                    raise KeyError('Bead index {} outside range [0,{}]'.format(i, num_types))
                nhs[i] += n_*N_i
        hs = HardSphere(self.d)
        fhs = hs.f_ex(nhs)

        # bonding contributions
        fbond = 0.
        for n_,p in zip(n,self.polymers):
            for (i,j),N_b in p.count_bonds().items():
                fbond -= n_ * N_b * np.log(hs.G(n,i,j))

        return fhs + fbond
