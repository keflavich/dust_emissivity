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

def blackbody(nu,temperature):
    """
    Planck's Law Blackbody (Frequency units)
    """
    I = 2*constants.h*nu**3 / constants.c**2 * (exp(constants.h*nu/(constants.k_B*temperature)) - 1)**-1

    return I


def blackbody_wavelength(lam,temperature):
    I = 2*constants.h*constants.c**2 / lam**5 * (exp(constants.h*constants.c/(constants.k_B*temperature*lam)) - 1)**-1

    return I

def modified_blackbody(nu, temperature, beta=1.75, column=1e22*u.cm**-2,
                       muh2=2.8, 
                       kappanu=None,
                       kappa0=4.0*u.cm**2*u.g**-1,
                       nu0=505*u.GHz, dusttogas=100.):
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

    if kappanu is None:
        kappanu = kappa0 / dusttogas * (nu/nu0)**beta
    
    # numpy apparently can't multiply floats and longs
    tau = muh2 * constants.m_p * kappanu * column

    modification = (1.0 - exp(-1.0 * tau))

    I = blackbody(nu, temperature)*modification

    return I

def modified_blackbody_wavelength(lam, temperature, beta=1.75, column=1e22*u.cm**-2,
                                  muh2=2.8,
                                  kappa0=4.0*u.cm**2*u.g**-1,
                                  nu0=505*u.GHz,
                                  dusttogas=100.):
    """
    Snu =  2hnu^3 c^-2  (e^(hnu/kT) - 1)^-1  (1 - e^(-tau_nu) )
    Kappa0 and Nu0 are set as per http://arxiv.org/abs/1101.4654 which uses OH94 values.
    beta = 1.75 is a reasonable default for Herschel data
    N = 1e22 is the column density in cm^-2

    nu0 and nu must have same units!  But wavelength is converted to frequency
    of the right unit anyway

    Parameters
    ----------
    lam : float
        Wavelength in units of `wavelength_units`
    temperature : float
        Temperature in Kelvins
    beta : float
        The blackbody modification value; the blackbody function is multiplied
        by :math:`(1-exp(-(\\nu/\\nu_0)**\\beta))`
    logN : float
        The log column denisty to be fit
    logscale : float
        An arbitrary logarithmic scale to apply to the blackbody function
        before passing it to mpfit; this is meant to prevent numerical
        instability when attempting to fit very small numbers.
        Can also be used to represent, e.g., steradians
    muh2 : float
        The mass (in amu) per molecule of H2.  Defaults to 2.8.
    units : 'cgs' or 'mks'
        The unit system to use
    wavelength_units : string
        A valid wavelength (e.g., 'angstroms', 'cm','m')
    kappa0 : float
        The opacity in cm^2/g *for gas* at nu0 (see dusttogas)
    nu0 : float
        The frequency at which the opacity power law is locked.
        kappa(nu) = kappa0/dusttogas * (nu/nu0)**beta
    normalize : function or None
        A normalization function for the blackbody.  Set to None if you're
        interested in the amplitude of the blackbody
    dusttogas : float
        The dust to gas ratio.  The opacity kappa0 is divided by this number to
        get the opacity of the dust
    """

    nu = constants.c/lam

    kappanu = kappa0/dusttogas * (nu/nu0)**beta
    tau = muh2 * constants.m_p * kappanu * column

    modification = (1.0 - exp(-1.0 * tau))

    I = blackbody(nu,temperature)*modification

    return I
