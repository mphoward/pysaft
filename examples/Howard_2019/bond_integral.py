import numpy as np
import scipy.integrate

kT = 1.0
dp = 1.0
dc = 5.0

rp = dp*2.**(1./6.) + 0.5*(dp+dc) - dp
rc = 0.5

class GaussianPotential:
    def __init__(self, A, B, rc):
        self.A = A
        self.B = B
        self.rc = rc

    def __call__(self, r):
        if r <= self.rc:
            return -self.A*np.exp(-self.B*r**2)
        else:
            return 0.0

def volume(u, sigma, rp):
    r"""Bond volume integral.

    Args:
        u (callable): Potential energy function.
        sigma (float): Contact distance for particle 1 and particle 2.
        rp (float): Radial offset distance of patch from center of particle 1.

    The integral is for a patch offset some distance from particle 1 interacting
    with some patch at the center of particle 2.

    The bond volume :math:`\Delta` is decomposed into two terms:

    .. math::

        \Delta = g(\sigma^+) v

    where :math:`g(\sigma^+)` is the radial distribution function at contact.
    This makes use of the approximation that the patch attraction range is short,
    and so the radial distribution function can be separated from the integral
    averaging the Mayer *f*-function.

    The bond volume *v* under this separation is:

    .. math::

        v = 2 \pi \sigma^2 \int_\sigma^{r_p + r_c} dr
            \int_{\cos \phi^*}^1 d\cos\phi f[d(r_p, r, \cos\phi)]

    where f is the Mayer *f*-function and *d* is the distance between patches.

    .. math::

        d^2 = r_p^2 + r^2 - 2 r r_p \cos\phi

    Here, :math:`r_p` is the distance of the patch from the center of particle 1,
    and :math:`r_c` is the radial cutoff distance on *d* for the patch--patch interaction.

    The lower limit :math:`\cos\phi^*` (with :math:`\phi` being the polar angle)
    is set by geometric considerations of the cutoff distance (with :math:`r_c` being
    sufficiently small):

    .. math::

        \cos phi^* = \frac{r^2 + r_p^2 - r_c^2}{2 r r_p}

    The double integral is straightforward to compute by quadrature.

    """
    def f_(x,r):
        """Integrand.

        Args:
            x (float): cos(phi) for polar angle phi [0,pi].
            r (float): distance to patch.

        Expects arguments in order (y,x) from quadrature, which is why
        these are ordered (x,r).

        """
        d = np.sqrt(rp**2+r**2-2.*r*rp*x)
        return np.exp(-u(d)/kT)-1.

    # cos(phi) bounds depend on r, so it is the "y" variable
    integral = scipy.integrate.dblquad(f_,
                                       sigma,
                                       rp+u.rc,
                                       lambda r : (r**2+rp**2-u.rc**2)/(2.*r*rp),
                                       1.)
    return 2.*np.pi*sigma**2*integral[0]

dA = 0.02
As = np.arange(0.0, 20.0+0.5*dA, dA)
vs = np.zeros_like(As)
for i,A in enumerate(As):
    vs[i] = volume(GaussianPotential(A,1./0.2**2,rc), sigma=0.5*(dc+dp), rp=rp)
np.savetxt('bond_integral.dat', np.column_stack((As,vs)), header='epsilon/kT volume')
