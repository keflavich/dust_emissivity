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
    fitter: 'lmfit', 'mpfit', or 'montecarlo'
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

    Example
    -------
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
    >>> tguess, bguess, nguess = 20.*u.K,2.,21.5*u.cm**-2
    >>> bbunit = u.erg/u.s/u.cm**2/u.Hz
    >>> pars = fit_modified_bb(frequencies, flux+noise, err,
    ...                        guesses=(tguess, bguess, nguess))
    """

    bbunit = u.erg/u.cm**2/u.s/u.Hz/u.sr
    x = xdata.to(u.Hz).value
    fx = flux.to(bbunit).value
    err = error.to(bbunit).value if error is not None else None

    guesses = [guesses[0].to(u.K).value,
               guesses[1],
               guesses[2].to(u.cm**-2).value]

    if fitter == 'lmfit':
        lm = fit_sed_lmfit_hz(xdata=x, flux=fx, guesses=guesses, err=err,
                              blackbody_function='modified', **kwargs)

        pars = [lm.params['T'].value * u.K,
                lm.params['beta'].value,
                lm.params['N'].value * u.cm**-2]
        if return_error:
            perrs = [lm.params['T'].stderr * u.K,
                     lm.params['beta'].stderr,
                     lm.params['N'].stderr * u.cm**-2]

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
    elif fitter == 'montecarlo':
        mc = fit_modifiedbb_montecarlo(x, fx, err=err,
                                       temperature_guess=guesses[0],
                                       beta_guess=guesses[1],
                                       column_guess=guesses[2],
                                       return_MC=True)
        stats = [mc.temperature.stats(),
                 mc.beta.stats(),
                 mc.column.stats()]
        pars = [stats[0]['mean']*u.K,
                stats[1]['mean'],
                stats[2]['mean']*u.cm**-2]
        if return_error:
            perrs = [stats[0]['standard deviation'] * u.K,
                     stats[1]['standard deviation'],
                     stats[2]['standard deviation']*u.cm**-2]

            return pars, perrs
        else:
            return pars
    else:
        raise ValueError("Fitter type must be one of mpfit, lmfit, montecarlo")




def fit_sed_mpfit_hz(xdata, flux, guesses=(0,0), err=None,
                     blackbody_function='blackbody', quiet=True, sc=1e20,
                     **kwargs):
    """
    Parameters
    ----------
    xdata : array
        Array of the frequencies of the data
    flux : array
        The fluxes corresponding to the xdata values.  Should be in
        erg/s/cm^2/Hz
    guesses : (Temperature,Column) or (Temperature,Beta,Column)
        The input guesses.  3 parameters are used for modified blackbody
        fitting, two for temperature fitting.
    blackbody_function: str
        The blackbody function to fit, either 'blackbody', 'modified', or
        'modified_blackbody'
    quiet : bool
        quiet flag passed to mpfit
    sc : float
        A numerical parameter to enable the fitter to function properly.
        It is unclear what values this needs to take, 1e20 seems to work
        by bringing the units from erg/s/cm^2/Hz to Jy, i.e. bringing them
        into the "of order 1" regime.  This does NOT affect the output *units*,
        though it may affect the quality of the fit.

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
    ...                       (flux+noise).to(bbunit).value, err=err.to(bbunit).value,
    ...                       blackbody_function='modified',
    ...                       guesses=(tguess, bguess, nguess))
    >>> print(mp.params)
    [ 14.99095224   1.78620237  22.05271119]
    >>> # T~14.9 K, beta ~1.79, column ~10^22
    """
    try:
        from agpy import mpfit
    except ImportError:
        print("Cannot import mpfit: cannot use mpfit-based fitter.")


    bbfd = {'blackbody': _blackbody_hz,
            'modified': _modified_blackbody_hz,
            'modified_blackbody': _modified_blackbody_hz}

    bbf = bbfd[blackbody_function]

    def mpfitfun(x,y,err):
        if err is None:
            def f(p,fjac=None):
                return [0,(y*sc-bbf(x, *p, **kwargs)*sc)]
        else:
            def f(p,fjac=None):
                return [0,(y*sc-bbf(x, *p, **kwargs)*sc)/(err*sc)]
        return f

    err = err if err is not None else flux*0.0 + 1.0

    mp = mpfit.mpfit(mpfitfun(xdata,flux,err), guesses, quiet=quiet)

    return mp


def fit_sed_lmfit_hz(xdata, flux, guesses=(0,0), err=None,
                     blackbody_function='blackbody', quiet=True, sc=1e20,
                     **kwargs):
    """
    Parameters
    ----------
    xdata : array
        Array of the frequencies of the data
    flux : array
        The fluxes corresponding to the xdata values.  Should be in
        erg/s/cm^2/Hz
    guesses : (Temperature,Column) or (Temperature,Beta,Column)
        The input guesses.  3 parameters are used for modified blackbody
        fitting, two for temperature fitting.
    blackbody_function: str
        The blackbody function to fit, either 'blackbody', 'modified', or
        'modified_blackbody'
    quiet : bool
        quiet flag passed to mpfit
    sc : float
        A numerical parameter to enable the fitter to function properly.
        It is unclear what values this needs to take, 1e20 seems to work
        by bringing the units from erg/s/cm^2/Hz to Jy, i.e. bringing them
        into the "of order 1" regime.  This does NOT affect the output *units*,
        though it may affect the quality of the fit.

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
    >>> lm = fit_sed_lmfit_hz(frequencies.to(u.Hz).value,
    ...                       (flux+noise).to(bbunit).value,
    ...                       err=err.to(bbunit).value,
    ...                       blackbody_function='modified',
    ...                       guesses=(tguess,bguess,nguess))
    >>> print(lm.params)
    
    >>> # If you want to fit for a fixed beta, do this:
    >>> import lmfit
    >>> parlist = [(n,lmfit.Parameter(x))
    ...            for n,x in zip(('T','beta','N'),(20.,2.,21.5))]
    >>> parameters = lmfit.Parameters(OrderedDict(parlist))
    >>> parameters['beta'].vary = False
    >>> lm = fit_sed_lmfit_hz(frequencies.to(u.Hz).value,
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
                return (y*sc-bbf(x, *[p[par].value for par in p], **kwargs)*sc)
        else:
            def f(p):
                return (y*sc-bbf(x, *[p[par].value for par in p], **kwargs)*sc)/(err*sc)
        return f

    if not isinstance(guesses,lmfit.Parameters):
        guesspars = lmfit.Parameters()

        for n,x in zip(('T','beta','N'),guesses):
            guesspars.add(value=x,name=n)
    else:
        guesspars = guesses

    minimizer = lmfit.minimize(lmfitfun(xdata,np.array(flux),err), guesspars)

    return minimizer


def fit_modifiedbb_montecarlo(frequency, flux, err=None,
                              temperature_guess=10, beta_guess=None,
                              column_guess=None,
                              quiet=True,
                              return_MC=False, nsamples=5000, burn=1000,
                              min_temperature=0, max_temperature=100,
                              max_column=1e30,
                              multivariate=False, **kwargs):
    """
    An MCMC version of the fitter.

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
    column_guess : float
        guess for the column density (cm^-2)
    return_MC : bool
        Return the pymc.MCMC object?
    nsamples : int
        Number of samples to use in determining the posterior distribution
        (the answer)
    burn : int
        number of initial samples to ignore
    kwargs : kwargs
        passed to blackbody function
    """

    try:
        import pymc
    except ImportError:
        print("Cannot import pymc: cannot use pymc-based fitter.")

    blackbody_function = _modified_blackbody_hz

    d = {}
    d['temperature'] = pymc.distributions.Uniform('temperature',
                                                  min_temperature,
                                                  max_temperature,
                                                  value=temperature_guess)
    d['column'] = pymc.distributions.Uniform('column',0,max_column,
                                             value=column_guess)
    if beta_guess is not None:
        d['beta'] = pymc.distributions.Uniform('beta',0,10, value=beta_guess)
    else:
        d['beta'] = pymc.distributions.Uniform('beta',0,0, value=0)


    @pymc.deterministic
    def luminosity(temperature=d['temperature'], beta=d['beta'],
                   column=d['column']):

        from scipy.integrate import quad
        f = lambda nu: blackbody_function(nu, temperature, beta=beta,
                                          column=column, **kwargs)
        # integrate from 0.1 to 10,000 microns (100 angstroms to 1 cm)
        # some care should be taken; going from 0 to inf results in failure
        return quad(f, 1e4, 1e17)[0]

    d['luminosity'] = luminosity

    @pymc.deterministic
    def bb_model(temperature=d['temperature'], column=d['column'],
                 beta=d['beta']):
        y = blackbody_function(frequency, temperature, beta=beta,
                               column=column, **kwargs)
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

    MC = pymc.MCMC(d)
    
    if nsamples > 0:
        MC.sample(nsamples, burn=burn)
        if return_MC:
            return MC

        MCfit = pymc.MAP(MC)
        MCfit.fit()
        T = MCfit.temperature.value
        column = MCfit.column.value

        if beta_guess is not None:
            beta = MCfit.beta.value
            return T,column,beta
        else:
            return T,column

    return MC
