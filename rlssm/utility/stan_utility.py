# Code mostly taken from M. Betancourt and can be retrieved at
# https://github.com/betanalpha/jupyter_case_studies/blob/master/principled_bayesian_workflow/stan_utility.py
import os
import pickle
from math import isnan
from math import isinf
from hashlib import md5
import cmdstanpy
import numpy

__dir__ = os.path.abspath(os.path.dirname(__file__))


def check_div(fit):
    """Check transitions that ended with a divergence.

    Parameters
    ----------

    fit : cmdstanpy.CmdStanModel
        The fitted stan model.

    """

    sampler_params = fit.draws_pd(inc_warmup=False)
    divergent = sampler_params['divergent__']
    n = sum(divergent)
    N = len(divergent)
    print('{} of {} iterations ended with a divergence ({}%)'.format(n, N, 100 * n / N))
    if n > 0:
        print('  Try running with larger adapt_delta to remove the divergences')


def check_treedepth(fit, max_depth=10):
    """Check transitions that ended prematurely due to maximum tree depth limit.

    Parameters
    ----------

    fit : cmdstanpy.CmdStanModel
        The fitted stan model.

    max_depth : int, default 10
        Maximum tree depth.

    """
    sampler_params = fit.draws_pd(inc_warmup=False)
    depths = sampler_params['treedepth__']
    n = sum(1 for x in depths if x == max_depth)
    N = len(depths)
    print(('{} of {} iterations saturated the maximum tree depth of {}' +
           ' ({}%)').format(n, N, max_depth, 100 * n / N))
    if n > 0:
        print('Run again with max_depth set to a larger value to avoid saturation')


def check_energy(fit):
    """Checks the energy Bayesian fraction of missing information (E-BFMI).

    Parameters
    ----------

    fit : cmdstanpy.CmdStanModel
        The fitted stan model.

    """
    # sampler_params = fit.draws_pd(inc_warmup=False)
    # no_warning = True
    # for chain_num, s in enumerate(sampler_params):
    #     energies = s['energy__']
    #     numer = sum((energies[i] - energies[i - 1])**2 for i in range(1, len(energies))) / len(energies)
    #     denom = numpy.var(energies)
    #     if numer / denom < 0.2:
    #         print('Chain {}: E-BFMI = {}'.format(chain_num, numer / denom))
    #         no_warning = False
    # if no_warning:
    #     print('E-BFMI indicated no pathological behavior')
    # else:
    #     print('  E-BFMI below 0.2 indicates you may need to reparameterize your model')

    sampler_params = fit.draws_pd(inc_warmup=False)
    no_warning = True

    energies = sampler_params['energy__']

    numer = sum((energies[i] - energies[i - 1]) ** 2 for i in range(1, len(energies))) / len(energies)
    denom = numpy.var(energies)
    if numer / denom < 0.2:
        print('Chain {}: E-BFMI = {}'.format(-1, numer / denom))  # TODO -1 should be chain number
        no_warning = False
    if no_warning:
        print('E-BFMI indicated no pathological behavior')
    else:
        print('  E-BFMI below 0.2 indicates you may need to reparameterize your model')


def check_n_eff(fit):
    """Checks the effective sample size per iteration.

    Parameters
    ----------

    fit : cmdstanpy.CmdStanModel
        The fitted stan model.

    """

    fit_summary = fit.summary(percentiles=[50])
    n_effs = fit_summary['N_Eff']
    names = list(fit_summary.index)
    n_iter = fit.draws().shape[0]

    no_warning = True
    for n_eff, name in zip(n_effs, names):
        ratio = n_eff / n_iter
        if ratio < 0.001:
            print('n_eff / iter for parameter {} is {}!'.format(name, ratio))
            print('E-BFMI below 0.2 indicates you may need to reparameterize your model')
            no_warning = False
    if no_warning:
        print('n_eff / iter looks reasonable for all parameters')
    else:
        print('n_eff / iter below 0.001 indicates that the effective sample size has likely been overestimated')


def check_rhat(fit):
    """Checks the potential scale reduction factors.

    Parameters
    ----------

    fit : cmdstanpy.CmdStanModel
        The fitted stan model.

    """

    fit_summary = fit.summary(percentiles=[50])
    rhats = fit_summary['R_hat']
    names = list(fit_summary.index)

    no_warning = True
    for rhat, name in zip(rhats, names):
        if (rhat > 1.01 or isnan(rhat) or isinf(rhat)):
            print('Rhat for parameter {} is {}!'.format(name, rhat))
            no_warning = False
    if no_warning:
        print('Rhat looks reasonable for all parameters')
    else:
        print('Rhat above 1.01 indicates that the chains very likely have not mixed')


def check_all_diagnostics(fit):
    """Checks all MCMC diagnostics, apart from rhat convergence.

    Parameters
    ----------

    fit : cmdstanpy.CmdStanModel
        The fitted stan model.

    """
    print("Checks MCMC diagnostics:")
    check_n_eff(fit)
    check_treedepth(fit)
    check_energy(fit)
    check_div(fit)


def _by_chain(unpermuted_extraction):
    num_chains = len(unpermuted_extraction[0])
    result = [[] for _ in range(num_chains)]
    for c in range(num_chains):
        for i in range(len(unpermuted_extraction)):
            result[c].append(unpermuted_extraction[i][c])
    return numpy.array(result)


def _shaped_ordered_params(fit):
    # flattened, unpermuted, by (iteration, chain)
    ef = fit.extract(permuted=False, inc_warmup=False)
    ef = _by_chain(ef)
    ef = ef.reshape(-1, len(ef[0][0]))
    ef = ef[:, 0:len(fit.flatnames)]  # drop lp__
    shaped = {}
    idx = 0
    for dim, param_name in zip(fit.par_dims, fit.extract().keys()):
        length = int(numpy.prod(dim))
        shaped[param_name] = ef[:, idx:idx + length]
        shaped[param_name].reshape(*([-1] + dim))
        idx += length
    return shaped


def partition_div(fit):
    """Returns parameter arrays separated into divergent and non-divergent transitions.

    Parameters
    ----------

    fit : cmdstanpy.CmdStanModel
        The fitted stan model.

    Returns
    -------

    nondiv_params : dict
        Dictionary containing the non-divergent transitions per parameter.

    div_params : dict
        Dictionary containing the divergent transitions per parameter.

    """
    sampler_params = fit.draws_pd(inc_warmup=False)
    div = numpy.concatenate([x['divergent__'] for x in sampler_params]).astype('int')
    params = _shaped_ordered_params(fit)
    nondiv_params = dict((key, params[key][div == 0]) for key in params)
    div_params = dict((key, params[key][div == 1]) for key in params)
    return nondiv_params, div_params


def compile_model(filename, model_name=None):
    """This will automatically cache models -
    great if you're just running a script on the command line.

    See https://mc-stan.org/cmdstanpy/

    Parameters
    ----------

    filename : str
        Location of the stan model.

    model_name : str, default None
        Optional model name to add to cashed model file.

    Returns
    -------

    sm : cmdstanpy.CmdStanModel
        The compiled stan model.

    """

    with open(filename) as f:
        model_code = f.read()
        code_hash = md5(model_code.encode('ascii')).hexdigest()
        path_pkl_fldr = os.path.join(os.path.dirname(os.path.dirname(__dir__)), "pkl_files")
        if not os.path.exists(path_pkl_fldr):
            os.makedirs(path_pkl_fldr)
        if model_name is None:
            cache_fn = os.path.join(path_pkl_fldr, f"cached-model-{code_hash}.pkl")
        else:
            cache_fn = os.path.join(path_pkl_fldr, f"cached-{model_name}-{code_hash}.pkl")
        try:
            with open(cache_fn, 'rb') as f1:
                sm = pickle.load(f1)
        except:
            sm = cmdstanpy.CmdStanModel(stan_file=filename)
            with open(cache_fn, 'wb') as f1:
                pickle.dump(sm, f1)
        else:
            print("Using cached StanModel")
        return sm
