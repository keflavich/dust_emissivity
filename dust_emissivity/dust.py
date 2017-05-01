"""
===============
Dust emissivity
===============
"""
from . import blackbody
import numpy as np
import os
import requests
from astropy.table import Table
from astropy import constants
from astropy import units as u
from numpy import exp,log
from scipy.integrate import quad

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

def kappa_table(nu, url='http://www2.mpia-hd.mpg.de/homes/henning/Dust_opacities/Opacities/Ralf/Pol/0__comp.Gofn'):
    """
    Use an opacity table to compute kappa

    (does not work well with integrals-  results in an error)
    """
    if os.path.exists(os.path.basename(url)):
        tbl = Table.read(os.path.basename(url), format='ascii.basic')
    else:
        response = requests.get(url)
        with open(os.path.basename(url),'w') as fh:
            fh.write(response.text)
        tbl = Table.read(response.text, format='ascii.basic')

    wavelengths = tbl["'lambda[um]"]*u.um
    frequencies = wavelengths.to(u.Hz, u.spectral())
    argsort = np.argsort(frequencies)

    absorption = tbl["<Kabs>"]*u.cm**2/u.g

    return np.interp(nu.to(u.Hz).value, frequencies[argsort].value,
                     absorption[argsort].value)*u.cm**2/u.g

def planck_average_kappa_table(temperature, weighting=blackbody.blackbody,
                               normalization='blackbody',
                               url='http://www2.mpia-hd.mpg.de/homes/henning/Dust_opacities/Opacities/Ralf/Pol/0__comp.Gofn',):

    if os.path.exists(os.path.basename(url)):
        tbl = Table.read(os.path.basename(url), format='ascii.basic')
    else:
        response = requests.get(url)
        with open(os.path.basename(url),'w') as fh:
            fh.write(response.text)
        tbl = Table.read(response.text, format='ascii.basic')

    wavelengths = tbl["'lambda[um]"]*u.um
    frequencies = wavelengths.to(u.Hz, u.spectral())
    argsort = np.argsort(frequencies)

    absorption = tbl["<Kabs>"]*u.cm**2/u.g

    dnu = np.diff(frequencies[argsort])

    bb = weighting(frequencies[argsort][:-1], temperature)

    # integrate over solid angle first, which is irrelevant as it's only a
    # weight, then integrate over frequency, which matters.
    integral = (4*np.pi*u.sr * bb * absorption[argsort][:-1] * dnu).sum()

    if normalization == 'blackbody':
        norm = constants.sigma_sb.cgs * temperature**4
    else:
        raise ValueError("Non-blackbody normalization not yet implemented")

    return (integral / norm).to(u.cm**2/u.g)



def planck_average_kappa(temperature, kappa=kappa, **kwargs):
    """
    Compute the Blackbody-weighted opacity assuming a power law dust
    absorptivity
    
    THIS IS NOT PRESENTLY SANE: the opacity doesn't continue as a power law past ~100um
    """
    def kappabb(nu):
        return (blackbody.blackbody(nu*u.Hz, temperature) * kappa(nu*u.Hz, **kwargs)).value

    result = quad(kappabb, 1e9, 1e15, full_output=True)

    if len(result) == 3:
        integral,err,infodict = result
    else:
        raise ValueError("Integral did not converge or had some other problem."
                         + str(result))

    integral_unit = (blackbody.blackbody(1e5*u.Hz, temperature)*u.Hz * kappa(1e5*u.Hz)).decompose().unit

    return (integral*integral_unit / (constants.sigma_sb.cgs * temperature**4)).to(u.cm**2/u.g)



def snu(nu, column, kappa, temperature):
    """
    Compute the flux density for a given column of gas assuming some opacity kappa
    """
    snu = blackbody.modified_blackbody(nu, temperature, kappanu=kappa, column=column)
    return snu

def snudnu(nu, column, kappa, temperature, bandwidth):
    return snu(nu, column, kappa, temperature) * bandwidth

def snuofmass(nu, mass, beamomega=1*u.sr, distance=1*u.kpc, temperature=20*u.K,
              **kwargs):
    """
    Flux density for a given mass and beam area
    (is generally independent of beam area, so it is set to a default value of
    1 sr)

    nu in Hz
    snu in Jy
    """
    beamomega = u.Quantity(beamomega, u.sr)
    effective_area = (beamomega * (distance**2)).to(u.cm**2, u.dimensionless_angles())
    column = (mass / effective_area).to(u.cm**-2*u.g)
    tau = kappa(nu, **kwargs) * column
    bnu = blackbody.blackbody(nu, temperature)
    snu = bnu * (1.0-exp(-tau)) * beamomega
    return snu.to(u.Jy)

def tauofsnu(nu, snu_per_beam, temperature=20*u.K):
    """
    nu in GHz
    snu in Jy/sr
    """
    bnu = blackbody.blackbody(nu, temperature)
    tau = -log(1-snu / bnu)
    return tau

def colofsnu(nu, snu_per_beam, temperature=20*u.K, muh2=2.8, **kwargs):
    tau = tauofsnu(nu=nu, snu_per_beam=snu_per_beam, temperature=temperature)
    column = tau / kappa(nu=nu,**kwargs) / constants.m_p / muh2
    return column.to(u.cm**-2)

def massofsnu(nu, snu, distance=1*u.kpc, temperature=20*u.K, muh2=2.8,
              beta=1.75, beamomega=1*u.sr):
    # beamomega matters if the optical depth is high.  Bigger beam = lower
    # optical depth.
    # However, we actually want to assume optically thin, so we use 1 sr by
    # default
    beamomega = u.Quantity(beamomega, u.sr)
    col = colofsnu(nu=nu, snu_per_beam=snu/beamomega, temperature=temperature,
                   beta=beta)
    effective_area = ((distance)**2 * beamomega).to(u.cm**2, u.dimensionless_angles())
    mass = col * constants.m_p * muh2 * effective_area
    return mass.to(u.M_sun)
