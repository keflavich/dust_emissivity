from __future__ import print_function
import numpy as np
from astropy import units as u

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from .blackbody import modified_blackbody, blackbody, _modified_blackbody_hz, _blackbody_hz

def fit_modified_bb(xdata, flux, error, guesses, fitter='lmfit',
                    return_error=False, **kwargs):
    """
    Wrapper SED fitter that uses astropy's units and returns the fitted
    parameters & errors

    Parameters
    ----------
    xdata : u.Quantity array
        Array of the frequencies of the data; must be equivalent to u.Hz
    flux : u.Quantity array
        The fluxes corresponding to the xdata values.  Must be equivalent to
        erg/s/cm^2/Hz
    guesses : (Temperature,Beta,Column)
        The input guesses.  Temperature and Column should have units of K and
        cm**-2 respectively
    fitter: 'lmfit' or 'mpfit'
        The fitter backend to use
    return_error: bool
        Optional argument whether the errors should be returned


    Returns
    -------
    pars : list
        A list of the fitted parameters with appropriate units
    errors : list
        (optional: see return_error)
        The errors corresponding to the fitted parameters
    """

    bbunit = u.erg/u.cm**2/u.s/u.Hz
    x = xdata.to(u.Hz).value
    fx = flux.to(bbunit).value
    err = error.to(bbunit).value if error is not None else None

    guesses = [guesses[0].to(u.K).value,
               guesses[1],
               guesses[2].to(u.cm**-2).value]

    if fitter == 'lmfit':
        lm = fit_sed_lmfit_hz(xdata=x, flux=fx, guesses=guesses, err=err,
                              blackbody_function='modified', **kwargs)

        pars = [lm.params[0].value * u.K,
                lm.params[1].value,
                lm.params[2].value * u.cm**-2]
        if return_error:
            perrs = [lm.params[0].stderr * u.K,
                     lm.params[1].stderr,
                     lm.params[2].stderr * u.cm**-2]

            return pars, perrs
        else:
            return pars
    elif fitter == 'mpfit':
        mp = fit_sed_mpfit_hz(xdata=x, flux=fx, guesses=guesses, err=err,
                              blackbody_function='modified', **kwargs)

        pars = [mp.params[0] * u.K,
                mp.params[1],
                mp.params[2] * u.cm**-2]
        if return_error:
            perrs = [mp.perror[0] * u.K,
                     mp.perror[1],
                     mp.perror[2] * u.cm**-2]

            return pars, perrs
        else:
            return pars
    else:
        raise ValueError("Fitter type must be one of mpfit, lmfit")




def fit_sed_mpfit_hz(xdata, flux, guesses=(0,0), err=None,
                     blackbody_function='blackbody', quiet=True, **kwargs):
    """
    Parameters
    ----------
    xdata : array
        Array of the frequencies of the data
    flux : array
        The fluxes corresponding to the xdata values.  Should be in
        erg/s/cm^2/Hz
    guesses : (Temperature,Scale) or (Temperature,Beta,Scale)
        The input guesses.  3 parameters are used for modified blackbody
        fitting, two for temperature fitting.
    blackbody_function: str
        The blackbody function to fit, either 'blackbody', 'modified', or
        'modified_blackbody'
    quiet : bool
        quiet flag passed to mpfit

    Returns
    -------
    mp : mpfit structure
        An mpfit structure.  Access parameters and errors via
        `mp.params` and `mp.perror`.  The covariance matrix
        is in mp.covar.

    Examples
    --------
    >>> from astropy import units as u
    >>> import numpy as np
    >>> wavelengths = np.array([20,70,160,250,350,500,850,1100]) * u.um
    >>> frequencies = wavelengths.to(u.Hz, u.spectral())
    >>> temperature = 15 * u.K
    >>> column = 1e22 * u.cm**-2
    >>> flux = modified_blackbody(frequencies, temperature, beta=1.75,
    ...                           column=column)
    >>> err = 0.1 * flux
    >>> np.random.seed(0)
    >>> noise = np.random.randn(frequencies.size) * err
    >>> tguess, bguess, nguess = 20.,2.,21.5
    >>> bbunit = u.erg/u.s/u.cm**2/u.Hz
    >>> mp = fit_sed_mpfit_hz(frequencies.to(u.Hz).value,
    ...                       flux.to(bbunit).value, err=err.to(bbunit).value,
    ...                       blackbody_function='modified',
    ...                       guesses=(tguess, bguess, nguess))
    >>> print(mp.params)
    [ 14.99095224   1.78620237  22.05271119]
    >>> # T~14.9 K, beta ~1.79, column ~10^22
    """
    try:
        import mpfit
    except ImportError:
        print("Cannot import mpfit: cannot use mpfit-based fitter.")


    bbfd = {'blackbody': _blackbody_hz,
            'modified': _modified_blackbody_hz,
            'modified_blackbody': _modified_blackbody_hz}

    bbf = bbfd[blackbody_function]

    def mpfitfun(x,y,err):
        if err is None:
            def f(p,fjac=None):
                return [0,(y-bbf(x, *p, normalize=False, **kwargs))]
        else:
            def f(p,fjac=None):
                return [0,(y-bbf(x, *p, normalize=False, **kwargs))/err]
        return f

    err = err if err is not None else flux*0.0 + 1.0

    mp = mpfit.mpfit(mpfitfun(xdata,flux,err), guesses, quiet=quiet)

    return mp


def fit_sed_lmfit_hz(xdata, flux, guesses=(0,0), err=None,
                     blackbody_function='blackbody', quiet=True, **kwargs):
    """
    Parameters
    ----------
    xdata : array
        Array of the frequencies of the data
    flux : array
        The fluxes corresponding to the xdata values.  Should be in
        erg/s/cm^2/Hz
    guesses : (Temperature,Scale) or (Temperature,Beta,Scale)
        The input guesses.  3 parameters are used for modified blackbody
        fitting, two for temperature fitting.
    blackbody_function: str
        The blackbody function to fit, either 'blackbody', 'modified', or
        'modified_blackbody'
    quiet : bool
        quiet flag passed to mpfit

    Returns
    -------
    lm : lmfit parameters
        The lmfit-py result structure.  Each parameter has many properties.

    Examples
    --------
    >>> from astropy import units as u
    >>> import numpy as np
    >>> wavelengths = np.array([20,70,160,250,350,500,850,1100]) * u.um
    >>> frequencies = wavelengths.to(u.Hz, u.spectral())
    >>> temperature = 15 * u.K
    >>> column = 1e22 * u.cm**-2
    >>> flux = modified_blackbody(frequencies, temperature, beta=1.75,
    ...                           column=column)
    >>> err = 0.1 * flux
    >>> np.random.seed(0)
    >>> noise = np.random.randn(frequencies.size) * err
    >>> tguess, bguess, nguess = 20.,2.,21.5
    >>> bbunit = u.erg/u.s/u.cm**2/u.Hz
    >>> lm = fit_sed_lmfit_hz(frequencies.to(u.Hz),
                              flux.to(bbunit).value,
                              err=err.to(bbunit).value,
                              blackbody_function='modified',
                              guesses=(tguess,bguess,nguess))
    >>> print(lm.params)
    
    >>> # If you want to fit for a fixed beta, do this:
    >>> import lmfit
    >>> parlist = [(n,lmfit.Parameter(x))
    ...            for n,x in zip(('T','beta','N'),(20.,2.,21.5))]
    >>> parameters = lmfit.Parameters(OrderedDict(parlist))
    >>> parameters['beta'].vary = False
    >>> lm = fit_sed_lmfit_hz(frequencies.to(u.Hz),
    ...                       flux.to(bbunit).value,
    ...                       err=err.to(bbunit).value,
    ...                       blackbody_function='modified',
    ...                       guesses=parameters)
    >>> print(lm.params)
    """
    try:
        import lmfit
    except ImportError:
        print("Cannot import lmfit: cannot use lmfit-based fitter.")

    bbfd = {'blackbody': _blackbody_hz,
            'modified': _modified_blackbody_hz,
            'modified_blackbody': _modified_blackbody_hz}

    bbf = bbfd[blackbody_function]


    def lmfitfun(x,y,err):
        if err is None:
            def f(p):
                return (y-bbf(x, *[p[par].value for par in p], **kwargs))
        else:
            def f(p):
                return (y-bbf(x, *[p[par].value for par in p], **kwargs))/err
        return f

    if not isinstance(guesses,lmfit.Parameters):
        parlist = [(n,lmfit.Parameter(value=x,name=n))
                   for n,x in zip(('T','beta','N'),guesses)
                   ]
        guesspars = lmfit.Parameters(OrderedDict(parlist))
    else:
        guesspars = guesses

    minimizer = lmfit.minimize(lmfitfun(xdata,np.array(flux),err), guesspars)

    return minimizer


def fit_blackbody_montecarlo(frequency, flux, err=None,
                             temperature_guess=10, beta_guess=None,
                             scale_guess=None,
                             blackbody_function=blackbody, quiet=True,
                             return_MC=False, nsamples=5000, burn=1000,
                             min_temperature=0, max_temperature=100,
                             scale_keyword='scale', max_scale=1e60,
                             multivariate=False, **kwargs):
    """
    Parameters
    ----------
    frequency : array
        Array of frequency values
    flux : array
        array of flux values
    err : array (optional)
        Array of error values (1-sigma, normal)
    temperature_guess : float
        Input / starting point for temperature
    min_temperature : float
    max_temperature : float
        Lower/Upper limits on fitted temperature
    beta_guess : float (optional)
        Opacity beta value
    scale_guess : float
        Arbitrary scale value to apply to model to get correct answer
    blackbody_function: function
        Must take x-axis (e.g. frequency), temperature, then scale and beta
        keywords (dependence on beta can be none)
    return_MC : bool
        Return the pymc.MCMC object?
    nsamples : int
        Number of samples to use in determining the posterior distribution
        (the answer)
    burn : int
        number of initial samples to ignore
    scale_keyword : ['scale','logscale','logN']
        What scale keyword to pass to the blackbody function to determine
        the amplitude
    kwargs : kwargs
        passed to blackbody function
    """

    try:
        import pymc
    except ImportError:
        print("Cannot import pymc: cannot use pymc-based fitter.")


    d = {}
    d['temperature'] = pymc.distributions.Uniform('temperature',
            min_temperature, max_temperature, value=temperature_guess)
    d['scale'] = pymc.distributions.Uniform('scale',0,max_scale,
            value=scale_guess)
    if beta_guess is not None:
        d['beta'] = pymc.distributions.Uniform('beta',0,10,
                value=beta_guess)
    else:
        d['beta'] = pymc.distributions.Uniform('beta',0,0,
                value=0)


    @pymc.deterministic
    def luminosity(temperature=d['temperature'], beta=d['beta'],
                   scale=d['scale']):

        f = lambda nu: blackbody_function(nu, temperature, logN=scale,
                                          beta=beta, normalize=False)
        # integrate from 0.1 to 10,000 microns (100 angstroms to 1 cm)
        # some care should be taken; going from 0 to inf results in failure
        return quad(f, 1e4, 1e17)[0]

    d['luminosity'] = luminosity

    @pymc.deterministic
    def bb_model(temperature=d['temperature'], scale=d['scale'],
                 beta=d['beta']):
        kwargs[scale_keyword] = scale
        y = blackbody_function(frequency, temperature, beta=beta,
                               normalize=False, **kwargs)
        #print kwargs,beta,temperature,(-((y-flux)**2)).sum()
        return y

    d['bb_model'] = bb_model

    if err is None:
        d['err'] = pymc.distributions.Uninformative('error',value=1.)
    else:
        d['err'] = pymc.distributions.Uninformative('error',value=err,observed=True)

    d['flux'] = pymc.distributions.Normal('flux', mu=d['bb_model'],
                                          tau=1./d['err']**2, value=flux,
                                          observed=True)

    #print d.keys()
    MC = pymc.MCMC(d)
    
    if nsamples > 0:
        MC.sample(nsamples, burn=burn)
        if return_MC:
            return MC

        MCfit = pymc.MAP(MC)
        MCfit.fit()
        T = MCfit.temperature.value
        scale = MCfit.scale.value

        if beta_guess is not None:
            beta = MCfit.beta.value
            return T,scale,beta
        else:
            return T,scale

    return MC
