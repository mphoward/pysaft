"""Bonding free energy."""

import numpy as np
import scipy.optimize

from . import core

class BondFreeEnergy(core.FreeEnergy):
    def __init__(self, num_patch):
        super().__init__()
        self.num_patch = num_patch

    @property
    def num_patch(self):
        return self._num_patch

    @num_patch.setter
    def num_patch(self, num_patch):
        self._num_patch = np.atleast_1d(num_patch)

    def bond_volume(self, n, i, j):
        raise NotImplementedError("Bond volume not defined.")

    def f(self, n):
        r"""Free-energy density (per kT).

        Args:
            n (float or list): Number densities of each type.

        Assuming that all *m* patches on a component are equal, the
        excess free energy of bonding is connected to the fraction of
        patches that are not bonded *X*.

        .. math::

            \beta f = \sum n_i m_i \left(\ln X_i - (1-X_i)/2 \right)

        """
        n = np.atleast_1d(n).astype(np.float64)
        X = self.X(n)
        return np.sum(n*self.num_patch*(np.log(X) + 0.5*(1.-X)))

    def X(self, n):
        r"""Fraction of patches not bonded.

        Args:
            n (float or list): Number densities of each type.

        Computes the fraction of patches *X* for each component that
        are **not** bonded. This amounts to solving the balances

        .. math::

            X_i = \left[ 1 + \sum n_j m_j X_j \Delta_{ij} \right]^{-1}

        where *m* is the number of patches on each component, and
        :math:`\Delta` is the bond-volume integral between components
        *i* and *j*. Either of these quantities can be zero to prevent
        bonding. Note that our use of this form of equations assumes that
        all patches on each component are equivalent to each other.

        The equations are solved using SciPy's nonlinear least-squares solver
        for the residual of these equations, subject to the physical
        constraint :math:`0 \le X \le 1`.

        """
        num_types = len(self.num_patch)

        # fill bond volume matrix
        d = np.zeros((num_types, num_types), dtype=np.float64)
        for i in range(num_types):
            for j in range(i,num_types):
                d[i,j] = d[j,i] = self.bond_volume(n,i,j)

        if num_types == 1:
            # quadratic formula for one component
            a = self.num_patch[0]*n[0]*d[0,0]
            return (-1. + np.sqrt(1.+4.*a))/(2.*a)
        elif num_types == 2 and np.allclose(np.diag(self._bond_volume),0):
            # quadratic formula for two components with no self-interactions
            a0 = self.num_patch[0]*n[0]*d[1,0]
            a1 = self.num_patch[1]*n[1]*d[0,1]

            x = np.zeros(2)
            x[0] = 2./(1.+a1-a0+np.sqrt(4.*a0+(1.+a1-a0)**2))
            x[1] = 2./(1.-a1+a0+np.sqrt(4.*a0+(1.+a1-a0)**2))
            return x
        else:
            # pose the system of equations as a constrained least-squares problem
            def residual(x):
                """Residual of the mass-action equations."""
                res = np.zeros(num_types, np.float64)
                for i in range(num_types):
                    res[i] = x[i]*(1.+np.sum(n*self.num_patch*x*d[i]))-1.
                return res
            def jacobian(x):
                """Jacobian of the mass-action equations."""
                jac = np.zeros((num_types, num_types), np.float64)
                for i in range(num_types):
                    for j in range(num_types):
                        if i == j:
                            # the diagonal gets all terms in sum, plus an extra bit to account for self terms
                            jac[i,i] = np.sum(n*self.num_patch*x*d[i]) + (1. + n[i]*self.num_patch[i]*x[i]*d[i,i])
                        else:
                            # off diagonal only picks up the cross interaction
                            jac[i,j] = n[j]*self.num_patch[j]*x[i]*d[i,j]
                return jac
            bounds = (np.zeros(num_types), np.ones(num_types))
            x0 = 0.5*np.ones(num_types)

            # solve system and ensure that all residuals are in fact close to zero
            result = scipy.optimize.least_squares(fun=residual, x0=x0, jac=jacobian, bounds=bounds)
            if not np.allclose(result.fun,0):
                raise RuntimeError('Unable to converge bond fraction solution')
            if np.any(result.x < 0) or np.any(result.x > 1):
                raise RuntimeError('Bond fraction solution outside range [0,1]')
            return result.x

class ContactBond(BondFreeEnergy):
    """Bond free-energy density using contact value.

    Args:
        num_patch (int or list): Number of patches on each component.
        bond_volume (float or list): Volume of each (*i*,*j*) bond pair
            as a square matrix (can be zero).
        G (callable): A callable function for g(r) at contact.

    """
    def __init__(self, num_patch, bond_volume, G):
        super().__init__(num_patch)
        self._bond_volume = np.atleast_2d(bond_volume).astype(np.float64)
        self._G = G

    def bond_volume(self, n, i, j):
        # only compute g if bond volume is **identically** zero
        if self._bond_volume[i,j] != 0:
            return self._bond_volume[i,j]*self._G(n,i,j)
        else:
            return 0.0
