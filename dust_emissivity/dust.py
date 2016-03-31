"""
===============
Dust emissivity
===============
"""
from . import blackbody
from astropy import constants
from astropy import units as u
from numpy import exp,log

def kappa(nu, nu0=271.1*u.GHz, kappa0=0.0114*u.cm**2*u.g**-1, beta=1.75):
    """
    Compute the opacity $\kappa$ given a reference frequency (or wavelength)
    and a power law governing the opacity as a fuction of frequency:

    $$ \kappa = \kappa_0 \left(\\frac{\\nu}{\\nu_0}\\right)^{\\beta} $$

    Parameters
    ----------
    nu: astropy.Quantity [u.spectral() equivalent]
        The frequency at which to evaluate kappa
    nu0: astropy.Quantity [u.spectral() equivalent]
        The reference frequency at which $\kappa$ is defined
    kappa0: astropy.Quantity [cm^2/g]
        The dust opacity per gram of H2 along the line of sight.  Because of
        the H2 conversion, this factor implicitly includes a dust to gas ratio
        (usually assumed 100)
    beta: float
        The power-law index governing kappa as a function of nu
    """
    return (kappa0*(nu.to(u.GHz,u.spectral())/nu0.to(u.GHz,u.spectral()))**(beta)).to(u.cm**2/u.g)

def snu(nu, column, kappa, temperature):
    """
    Compute the flux density for a given column of gas assuming some opacity kappa
    """
    snu = blackbody.modified_blackbody(nu, temperature, kappanu=kappa, column=column)
    return snu

def snudnu(nu, column, kappa, temperature, bandwidth):
    return snu(nu, column, kappa, temperature) * bandwidth

def snuofmass(nu, mass, beamomega, distance=1*u.kpc, temperature=20*u.K, **kwargs):
    """
    nu in Hz
    snu in Jy
    """
    column = mass.to(u.M_sun) / (beamomega * (distance**2))
    tau = kappa(nu, **kwargs) * column * beamomega
    bnu = blackbody.blackbody(nu, temperature)
    snu = bnu * (1.0-exp(-tau))
    return snu.to(u.Jy)

def tauofsnu(nu, snu, temperature=20*u.K):
    """
    nu in GHz
    snu in Jy
    """
    bnu = blackbody.blackbody(nu, temperature)
    tau = -log(1-snu / bnu)
    return tau

def colofsnu(nu, snu, beamomega, temperature=20*u.K, muh2=2.8, **kwargs):
    tau = tauofsnu(nu=nu, snu=snu, temperature=temperature)
    column = tau / kappa(nu=nu,**kwargs) / constants.m_p / muh2 / beamomega
    return column

def massofsnu(nu, snu, distance=1*u.kpc, temperature=20*u.K, muh2=2.8, beta=1.75):
    # beamomega divides out: set it to 1
    col = colofsnu(nu=nu, snu=snu, beamomega=1, temperature=temperature, beta=beta)
    mass = col * constants.m_p * muh2 * (distance)**2
    return mass.to(u.M_sun)
