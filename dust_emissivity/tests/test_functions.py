from .. import dust
from astropy import units as u
import numpy as np

def test_kappa():
    assert dust.kappa(271.1*u.GHz) == 0.0114*u.cm**2/u.g
    np.testing.assert_almost_equal(dust.kappa(505*u.GHz).value, 0.03386008993377792)
    np.testing.assert_almost_equal(dust.kappa(505*u.GHz, beta=2).value, 0.03955747592258053)

def test_snuofmass():
    
    bm =  (2*np.pi*(40/2.35*u.arcsec)**2)
    snu = dust.snuofmass(600*u.GHz, 1e4*u.M_sun, bm, distance=5.1*u.kpc)

    np.testing.assert_almost_equal(snu.value, 363.7356579179746)
