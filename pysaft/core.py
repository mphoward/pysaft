"""Core data structures."""

__all__ = ['FreeEnergy','Polymer']

import numpy as np
import numdifftools as nd

class FreeEnergy:
    def __init__(self):
        pass

    def __add__(self, term):
        if isinstance(term, CompositeFreeEnergy):
            return CompositeFreeEnergy([self] + term.terms)
        elif isinstance(term, FreeEnergy):
            return CompositeFreeEnergy([self,term])
        else:
            raise TypeError('Can only add FreeEnergy terms together')

    def __call__(self, n):
        r"""Alias for free-energy density (per kT).

        Args:
            n (float or list): Number densities of each type.

        """
        return self.f(n)

    def f(self, n):
        r"""Free-energy density (per kT).

        Args:
            n (float or list): Number densities of each type.

        """
        raise NotImplementedError("Free energy density not implemented.")

    def z(self, n):
        r"""Compressibility factor.

        Args:
            n (float or list): Number densities of each type.

        The compressibility factor is defined relative to the ideal
        gas of the component densities.

        .. math::

            Z = \frac{p}{\rho k_{\rm B} T}

        where *p* is the pressure and :math:`\rho = \sum n_i`.

        The compressibility factor can also be computed from the volume
        derivative of the free energy.

        .. math::

            Z = -\rho^{-1} \left( \frac{\partial [\beta f(N_i/V) V]}{\partial V} \right)_{N_i,T}

        """
        # nominal basis for holding N constant in differentiation
        V0 = 1.0
        N0 = V0*np.atleast_1d(n).astype(np.float64)
        dFdV = nd.Derivative(lambda V : V*self.f(N0/V), step=1.e-5)
        return -dFdV(V0)/np.sum(n)

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
        # partial derivative with respect to n[i], all others held constant
        def _f(x):
            _n = np.copy(n).astype(np.float64)
            _n[i] = x
            return self.f(_n)
        step = nd.MinStepGenerator(base_step=1.e-7, step_ratio=2, num_steps=4)
        return nd.Derivative(_f, step=step)(n[i])

class CompositeFreeEnergy(FreeEnergy):
    def __init__(self, terms=[]):
        super().__init__()

        # keep track of all terms in this composite free energy
        self.terms = []
        try:
            self.terms.extend(terms)
        except TypeError:
            self.terms.append(terms)

    def __add__(self, term):
        if isinstance(term, CompositeFreeEnergy):
            return CompositeFreeEnergy(self.terms + term.terms)
        elif isinstance(term, FreeEnergy):
            return CompositeFreeEnergy(self.terms + [term])
        else:
            raise TypeError('Can only add FreeEnergy terms together')

    def f(self, n):
        return np.sum([term.f(n) for term in self.terms])

    def z(self, n):
        return np.sum([term.z(n) for term in self.terms])

    def mu(self, n, i):
        return np.sum([term.mu(n,i) for term in self.terms])

class Polymer:
    """Polymer chain topology.

    Args:
        particles (list): Type of each sphere in the chain.
        bonds (list): Bonds between particles in the chain.

    """
    def __init__(self, types, bonds):
        self.types = types
        self.bonds = bonds

    @property
    def types(self):
        return self._types

    @types.setter
    def types(self, types):
        self._types = types
        self._type_counts = None

    @property
    def bonds(self):
        return self._bonds

    @bonds.setter
    def bonds(self, bonds):
        self._bonds = bonds
        self._bond_counts = None

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
