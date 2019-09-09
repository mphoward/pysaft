"""Reference equations of state."""

import numpy as np

class EOS:
    """Equation of state."""
    def __init__(self):
        pass

    def f(self, n):
        r"""Free-energy density (per kT).

        Args:
            n (float or list): Number densities of each type.

        The total free energy is the sum of the ideal and excess parts.

        ..math::

            \beta f = \sum n_i (\ln n_i - 1) + \beta f^{\rm ex}

        where :math:`\beta = k_{\rm B} T`. The thermal de Broglie wavelengths
        (quantum contribution to the classical ideal gas free energy) are
        implicitly set equal to 1.

        """
        return np.sum(n*(np.log(n)-1.)) + self.fex(n)

    def fex(self, n):
        """Excess free-energy density (per kT).

        Args:
            n (float or list): Number densities of each type.

        """
        raise NotImplementedError("Excess free energy not defined")

    def z(self, n):
        r"""Compressibility factor.

        Args:
            n (float or list): Number densities of each type.

        The compressibility factor is defined relative to the ideal
        gas of the component densities.

        .. math::

            Z = \frac{p}{\rho k_{\rm B} T}

        where *p* is the pressure and :math:`\rho = \sum n_i`.

        The compressibility factor can also be computed from the density
        derivative of the free energy.

        .. math::

            Z = \rho \left( \frac{\partial (\beta f/\rho)}{\partial \rho} \right)_{N_i,T}

        """
        raise NotImplementedError("Compressibility factor not defined")

    def mu(self, n, i):
        r"""Chemical potential.

        Args:
            n (float or list): Number densities of each type.
            i (int): Component to evaluate chemical potential for.

        The chemical potential is defined from the partial density derivative
        of the free energy.

        .. math::

            \beta \mu_i = \left( \frac{\beta f}{n_i} \right)_{N_{j \ne i}, V,T}

        """
        raise NotImplementedError("Chemical potential not defined")

    def G(self, n, i, j):
        """Radial distribution function at contact.

        Args:
            n (float or list): Number densities of each type.
            i (int): First component in pair.
            j (int): Second component in pair.

        """
        raise NotImplementedError("Radial distribution function not defined")

class HardSphere(EOS):
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

        The auxilliary values for this EOS are related to fundamental
        measures of the sphere for :math:`0 \le m \le 3`.

        .. math::

            \xi_m = \sum \frac{n_i \pi d_i^m}{6}

        For example, :math:`m=3` is the volume fraction of the mixture.
        The dimensions of :math:`\xi_m` are :math:`d^(m-3)` since *n*
        is a density.

        """
        n = np.atleast_1d(n)
        return np.array([np.sum(n*np.pi*self.d**m/6.) for m in range(4)])

    def fex(self, n):
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

    def G(self, n, i, j):
        ri = 0.5*self.d[i]
        rj = 0.5*self.d[j]
        rij = ri*rj/(ri+rj)

        xi = self._xi(n)

        return (1./(1.-xi[3]) + 6.*(rij*xi[2])/(1.-xi[3])**2
               + 8.*(rij*xi[2])**2/(1.-xi[3])**3)


class HardChain(EOS):
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

    def fex(self, n):
        n = np.atleast_1d(n)

        # reference hard-sphere free energy
        hs = HardSphere(self.d)
        # the total number of spheres of each type is n*M
        nhs = n*self.M
        fhs = hs.fex(nhs)

        # free energy to bond beads in a chain
        fbond = np.sum([-n[i]*(self.M[i]-1.)*np.log(hs.G(nhs,i,i)) for i in range(len(n))])

        return fhs + fbond

class Polymer:
    """Polymer chain topology.

    Args:
        particles (list): Type of each sphere in the chain.
        bonds (list): Bonds between particles in the chain.

    """
    def __init__(self, types, bonds):
        self._types = types
        self._bonds = bonds

        self._type_counts = None
        self._bond_counts = None

    @property
    def types(self):
        return self._types

    @property
    def bonds(self):
        return self._bonds

    def count_types(self):
        """Count the number of each bead type.

        Returns:
            A dict where each key is a bead type and each value
            is the number of beads of that type in the polymer (int).
        """
        if self._type_counts is None:
            counts = {}
            for i in self.types:
                if i not in counts:
                    counts[i] = 0
                counts[i] += 1
            self._type_counts = counts
        return self._type_counts

    def count_bonds(self):
        """Count the number of each bond pair.

        Returns:
            A dict where each key is a bond pair and each
            value is the number of those pairs in the polymer (int).

        A bond pair is defined by the types of the beads in the pair.
        For simplicity, in counting a pair is defined so that the
        first type in the pair has an index less than or equal to the
        second type in the pair.

        """
        if self._bond_counts is None:
            counts = {}
            for i,j in self.bonds:
                a = self.types[i]
                b = self.types[j]
                pair = (a,b) if a < b else (b,a)
                if pair not in counts:
                    counts[pair] = 0
                counts[pair] += 1
            self._bond_counts = counts
        return self._bond_counts

class PolyatomicHardChain(EOS):
    """Amos and Jackson equation of state for polyatomic hard chains.

    Args:
        polymers (Polymer or list): Polymer chains in mixture.
        d (float or list): Diameters of each **bead** type.

    Unlike the HardChainEOS, this EOS does not assume that all beads
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

    def fex(self, n):
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
        fhs = hs.fex(nhs)

        # bonding contributions
        fbond = 0.
        for n_,p in zip(n,self.polymers):
            for (i,j),N_b in p.count_bonds().items():
                fbond -= n_ * N_b * np.log(hs.G(n,i,j))

        return fhs + fbond

if __name__ == '__main__':
    # test hard spheres
    deta = 0.01
    etas = np.arange(deta, 0.45 + 0.5*deta, deta)
    with open('hs.dat','w') as f:
        for eta in etas:
            rho = 6.*eta/np.pi
            hs = HardSphere(1.)

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
                hc = HardChain(M, 1.)

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
                p = Polymer(types=[0]*M, bonds=np.column_stack((idx[:-1],idx[1:])))

                rho = 6.*eta/(M*np.pi)
                hc = PolyatomicHardChain(polymers=p, d=[1.])

                # estimate compressibility from free energy
                eps = 1.e-6
                rho_eps = rho+eps
                z_f = rho * (hc.f(rho_eps)/rho_eps - hc.f(rho)/rho)/eps
                row.append(z_f)

            f.write(("{:.2f}" + " {:.5f}"*len(Ms) + "\n").format(*row))
