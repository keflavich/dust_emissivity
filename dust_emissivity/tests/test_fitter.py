from astropy import units as u
import numpy as np
from astropy.tests.helper import pytest
from ..blackbody import modified_blackbody
from ..fit_sed import fit_modified_bb

@pytest.mark.parametrize(('fitter',),('lmfit','mpfit'))
def test_fit_modified_bb(fitter):
    wavelengths = np.array([20,70,160,250,350,500,850,1100]) * u.um
    frequencies = wavelengths.to(u.Hz, u.spectral())
    temperature = 15 * u.K
    column = 1e22 * u.cm**-2
    flux = modified_blackbody(frequencies, temperature, beta=1.75,
                              column=column)
    err = 0.1 * flux
    np.random.seed(0)
    noise = np.random.randn(frequencies.size) * err
    tguess, bguess, nguess = 20.*u.K,2.,10**21.5*u.cm**-2

    pars = fit_modified_bb(frequencies, flux+noise, err,
                           guesses=(tguess, bguess, nguess),
                           fitter=fitter)

    answer = np.array([1.50147255e+01,   1.77886818e+00,   1.05578825e+22])
    
    np.testing.assert_approx_equal(answer[0], pars[0].value, significant=8)
    np.testing.assert_approx_equal(answer[1], pars[1], significant=8)
    np.testing.assert_approx_equal(answer[2], pars[2].value, significant=8)
