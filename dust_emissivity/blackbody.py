"""
============================
Simple black-body calculator
============================

Includes both wavelength and frequency blackbody functions.  Has flexible
units.  Also allows for a few varieties of modified blackbody.

"""
from numpy import exp
from astropy import units as u
from astropy import constants

# Declare global constants with numeric values to allow for relatively
# high-performance (low-overhead) use of astropy units
_h = constants.h.cgs.value
_c = constants.c.cgs.value
_k_B = constants.k_B.cgs.value
_m_p = constants.m_p.cgs.value

# Globally define the unit of the blackbody function in CGS
_bbunit_nu_cgs = u.erg/u.s/u.cm**2/u.Hz
_bbunit_lam_cgs = u.erg/u.s/u.cm**2/u.cm


def _blackbody_hz(nu, temperature):
    """
    Compute the Planck function given nu in Hz and temperature in K with output
    in cgs
    """
    I = (2*_h*nu**3 / _c**2 * (exp(_h*nu/(_k_B*temperature)) - 1)**-1)

    return I

def blackbody(nu, temperature, outunit=u.erg/u.s/u.cm**2/u.Hz):
    """
    Planck's Law Blackbody (Frequency units)
    """
    I = _blackbody_hz(nu.to(u.Hz).value,
                      temperature.to(u.K).value) * _bbunit_nu_cgs

    return I.to(outunit)

def _blackbody_wavelength_cm(lam, temperature):
    """
    Compute the Planck function given wavelength in cm and temperature in K
    with output in cgs
    """
    I = (2*_h*_c**2 / lam**5 * (exp(_h*_c/(_k_B*temperature*lam)) - 1)**-1)
    return I

def blackbody_wavelength(lam, temperature, outunit=u.erg/u.s/u.cm**2/u.AA):
    I = _blackbody_wavelength_cm(lam.to(u.cm).value,
                                 temperature.to(u.K).value) * _bbunit_lam_cgs

    return I.to(outunit)

def _modified_blackbody_hz(nu, temperature, beta, column, muh2=2.8, kappanu=None,
                           kappa0=4.0, nu0=505e9, dusttogas=100.):
    """
    Numpy-only computation of the modified blackbody function.  Intended for
    use during fitting and in other cases of high-performance needs
    """
    if kappanu is None:
        kappanu = kappa0 / dusttogas * (nu/nu0)**beta
    
    # numpy apparently can't multiply floats and longs
    tau = muh2 * _m_p * kappanu * column

    modification = (1.0 - exp(-1.0 * tau))

    I = _blackbody_hz(nu, temperature)*modification

    return I

def modified_blackbody(nu, temperature, beta=1.75, column=1e22*u.cm**-2,
                       muh2=2.8,
                       kappanu=None,
                       kappa0=4.0*u.cm**2*u.g**-1,
                       nu0=505*u.GHz, dusttogas=100.,
                       outunit=u.erg/u.s/u.cm**2/u.Hz):
    """
    Snu =  2hnu^3 c^-2  (e^(hnu/kT) - 1)^-1  (1 - e^(-tau_nu) )
    Kappa0 and Nu0 are set as per http://arxiv.org/abs/1101.4654 which uses OH94 values.
    beta = 1.75 is a reasonable default for Herschel data
    N = 1e22 is the column density in cm^-2

    nu0 and nu must have same units!

    Parameters
    ----------
    nu : float
        Frequency in units of `frequency_units`
    temperature : float
        Temperature in Kelvins
    beta : float
        The blackbody modification value; the blackbody function is multiplied
        by :math:`(1-exp(-(\\nu/\\nu_0)**\\beta))`
    logN : float
        The log column density to be fit
    logscale : float
        An arbitrary logarithmic scale to apply to the blackbody function
        before passing it to mpfit; this is meant to prevent numerical
        instability when attempting to fit very small numbers.
        Can also be used to represent, e.g., steradians
    muh2 : float
        The mass (in amu) per molecule of H2.  Defaults to 2.8.
    units : 'cgs' or 'mks'
        The unit system to use
    frequency_units : string
        Hz or some variant (GHz, kHz, etc)
    kappa0 : float
        The opacity in cm^2/g *for gas* at nu0 (see dusttogas)
    nu0 : float
        The frequency at which the opacity power law is locked
    normalize : function or None
        A normalization function for the blackbody.  Set to None if you're
        interested in the amplitude of the blackbody
    dusttogas : float
        The dust to gas ratio.  The opacity kappa0 is divided by this number to
        get the opacity of the dust
    """
    if kappanu is not None:
        kappanu = kappanu.to(u.cm**2/u.g).value

    I = _modified_blackbody_hz(nu.to(u.Hz).value,
                               temperature.to(u.K).value,
                               beta,
                               column.to(u.cm**-2).value,
                               muh2,
                               kappanu,
                               kappa0.to(u.cm**2/u.g).value,
                               nu0.to(u.Hz).value,
                               dusttogas)*_bbunit_nu_cgs
    return I.to(outunit)

def integrate_sed(vmin, vmax, function=blackbody, **kwargs):
    """
    Integrate one of the SED functions over *frequency*

    Parameters
    ----------
    vmin, vmax : astropy.Quantity
        Quantities with frequency equivalents: can be wavelength or frequency
    function : function
        One of the above blackbody functions.  The temperature etc. can be specified with
        kwargs

    """
    from scipy.integrate import quad

    fmin = vmin.to(u.Hz, u.spectral())
    fmax = vmax.to(u.Hz, u.spectral())

    bbunit = u.erg/u.s/u.cm**2/u.Hz

    intfunc = lambda nu: function(nu*u.Hz, **kwargs).to(bbunit).value

    result = quad(intfunc, fmin.to(u.Hz).value, fmax.to(u.Hz).value,
                  full_output=True)

    if len(result) == 3:
        integral,err,infodict = result
    else:
        raise ValueError("Integral did not converge or had some other problem.")

    return integral*bbunit*u.Hz
