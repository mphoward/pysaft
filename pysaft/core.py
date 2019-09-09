"""Core data structures."""

__all__ = ['FreeEnergy','Polymer']

class FreeEnergy:
    """Free energy."""
    def __init__(self):
        pass

    def f(self, n):
        r"""Free-energy density (per kT).

        Args:
            n (float or list): Number densities of each type.

        """
        raise NotImplementedError("Free energy not defined")

    def f_ex(self, n):
        r"""Excess free-energy density (per kT).

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

    def z_ex(self, n):
        r"""Excess compressibility factor.

        Args:
            n (float or list): Number densities of each type.

        Excess is defined relative to the ideal gas at the same *n*.

        .. math::

            Z^{\rm ex} = Z - 1

        """
        raise NotImplementedError("Excess compressibility factor not defined")

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

    def mu_ex(self, n, i):
        r"""Excess chemical potential.

        Args:
            n (float or list): Number densities of each type.
            i (int): Component to evaluate chemical potential for.

        The excess chemical potential is defined relative to the ideal gas at
        the same *n*.
        """
        raise NotImplementedError("Excess chemical potential not defined")

    def G(self, n, i, j):
        """Radial distribution function at contact.

        Args:
            n (float or list): Number densities of each type.
            i (int): First component in pair.
            j (int): Second component in pair.

        """
        raise NotImplementedError("Radial distribution function not defined")

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
