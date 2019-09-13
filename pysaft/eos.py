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

    def G(self, n, i, j):
        return 1.0

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
        f_id = IdealGas()(n)

        xi = self._xi(n)
        f_ex = (6./np.pi)*((xi[2]**3/xi[3]**2-xi[0])*np.log(1.-xi[3])
                          + 3.*xi[1]*xi[2]/(1.-xi[3])
                          + xi[2]**3/(xi[3]*(1.-xi[3])**2))

        return f_id + f_ex

    def G(self, n, i, j):
        ri = 0.5*self.d[i]
        rj = 0.5*self.d[j]
        rij = ri*rj/(ri+rj)

        xi = self._xi(n)

        return (1./(1.-xi[3]) + 6.*(rij*xi[2])/(1.-xi[3])**2
               + 8.*(rij*xi[2])**2/(1.-xi[3])**3)

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
        n = np.atleast_1d(n).astype(np.float64)

        # ideal gas of chains
        f_id = IdealGas()(n)

        # excess hard-sphere free energy
        # the total number of spheres of each type is n*M
        hs,nhs = self.as_hs(n)
        f_hs = hs(nhs) - IdealGas()(nhs)

        # free energy to bond hard beads in a chain
        f_bond = np.sum([-n[i]*(self.M[i]-1.)*np.log(hs.G(nhs,i,i)) for i in range(len(n))])

        return f_id + f_hs + f_bond

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
        n = np.atleast_1d(n).astype(np.float64)
        f_id = IdealGas()(n)

        # sum up the densities of each hs type
        num_types = len(self.d)
        hs,nhs = self.as_hs(n)
        f_hs = hs(nhs) - IdealGas()(nhs)

        # bonding contributions
        f_bond = 0.
        for n_,p in zip(n,self.polymers):
            for (i,j),N_b in p.count_bonds().items():
                f_bond -= n_ * N_b * np.log(hs.G(nhs,i,j))

        return f_id + f_hs + f_bond

    def as_hs(self, n):
        num_types = len(self.d)
        nhs = np.zeros(num_types, dtype=np.float64)
        for n_,p in zip(n,self.polymers):
            for i,N_i in p.count_types().items():
                if i >= num_types:
                    raise KeyError('Bead index {} outside range [0,{}]'.format(i, num_types))
                nhs[i] += n_*N_i
        return HardSphere(self.d),nhs
