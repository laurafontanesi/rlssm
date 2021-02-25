import pickle
import numpy as np

def load_model_results(filename):
    """Load model results from pickle.
    """
    return pickle.load(open(filename, "rb"))

def list_trial_variables(var_name, N):
    a = var_name + '[%s]'

    return np.core.defchararray.mod(a, np.arange(1, N+1))

def list_individual_variables(var_name, L):
    a = var_name + '_sbj[%s]'

    return np.core.defchararray.mod(a, np.arange(1, L+1))

def extract_trial_specific_vars(x, var_name, N):
    var_list = list_trial_variables(var_name, N)
    var_t = x[var_list]

    return var_t

def bci(x, alpha=0.05):
    """Calculate Bayesian credible interval (BCI).

    Parameters
    ----------

    x : array-like
        An array containing MCMC samples.

    alpha : float
        Desired probability of type I error (defaults to 0.05).

    Returns
    -------

    interval : numpy.ndarray
        Array containing the lower and upper bounds of the bci interval.

    """

    interval = np.nanpercentile(x, [(alpha/2)*100, (1-alpha/2)*100])

    return interval

# Code for caclulating hdi was taken from pymc3 and can be retrieved at
# https://github.com/pymc-devs/pymc3/blob/master/pymc3/stats.py
def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width.

    Parameters
    ----------

    x : array-like
        An sorted numpy array.

    alpha : float
        Desired probability of type I error (defaults to 0.05).

    Returns
    -------

    hdi_min : float
        The lower bound of the interval.

    hdi_max : float
        The upper bound of the interval.

    """

    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max


def hdi(x, alpha=0.05):
    """Calculate highest posterior density (HPD).

        Parameters
        ----------

        x : array-like
            An array containing MCMC samples.

        alpha : float
            Desired probability of type I error (defaults to 0.05).

    Returns
    -------

    interval : numpy.ndarray
        Array containing the lower and upper bounds of the hdi interval.

    """

    # Make a copy of trace
    x = x.copy()
     # Sort univariate node
    sx = np.sort(x)
    interval = np.array(calc_min_interval(sx, alpha))

    return interval
