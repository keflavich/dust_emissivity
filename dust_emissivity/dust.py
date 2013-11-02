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
    return kappa0*(nu.to(u.GHz)/nu0)**(beta)

def snu(nu, column, kappa, temperature):
    snu = blackbody.blackbody(nu, temperature, normalize=False)
    return snu

def snudnu(nu, column, kappa, temperature, bandwidth):
    return snu(nu, column, kappa, temperature) * bandwidth

def snuofmass(nu, mass, beamomega, distance=1*u.kpc, temperature=20*u.K):
    """
    nu in Hz
    snu in Jy
    """
    column = mass.to(u.M_sun) / (beamomega * (distance**2))
    tau = kappa(nu) * column * beamomega
    bnu = blackbody.blackbody(nu, temperature)
    snu = bnu * (1.0-exp(-tau))
    return snu.to(u.Jy)

def tauofsnu(nu, snu, beamomega, temperature=20*u.K):
    """
    nu in GHz
    snu in Jy
    """
    bnu = blackbody.blackbody(nu, temperature)
    tau = -log(1-snu / bnu)
    return tau

def colofsnu(nu, snu, beamomega, temperature=20*u.K, muh2=2.8, **kwargs):
    tau = tauofsnu(nu,snu,beamomega,temperature=temperature)
    column = tau / kappa(nu,**kwargs) / constants.m_p / muh2 / beamomega
    return column

def massofsnu(nu, snu, beamomega, distance=1*u.kpc, temperature=20*u.K, muh2=2.8, beta=1.75):
    col = colofsnu(nu, snu, beamomega, temperature, beta=beta)
    mass = col * constants.m_p * muh2 * beamomega * (distance)**2 
    return mass.to(u.M_sun)
