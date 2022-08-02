from __future__ import absolute_import, division, print_function
import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rlssm import plotting
from .utils import list_individual_variables
from .stan_utility import check_all_diagnostics
from .random import random_ddm, random_rdm_2A

class FittedModel(object):
    def __init__(self,
                 stan_model,
                 data,
                 hierarchical_levels,
                 model_label,
                 family,
                 n_parameters_individual,
                 n_parameters_trial,
                 print_diagnostics,
                 priors):
        self.stan_model = stan_model
        self.model_label = model_label
        self.family = family
        self.priors = priors

        # Print mcmc diagnostics...
        if print_diagnostics:
            check_all_diagnostics(self.stan_model)

        self.data_info = {'N': data.shape[0], 'data':data}

        n_samples_after_warmup = self.stan_model.stan_args[0]['iter'] - self.stan_model.stan_args[0]['warmup']
        n_posterior_samples = n_samples_after_warmup / self.stan_model.stan_args[0]['thin']*len(self.stan_model.stan_args)

        self.parameters_info = {'hierarchical_levels': hierarchical_levels,
                                'n_parameters_individual':n_parameters_individual,
                                'n_parameters_trial': n_parameters_trial,
                                'n_posterior_samples': int(n_posterior_samples)}

        if self.parameters_info['hierarchical_levels'] == 2:
            self.data_info.update({'L': len(pd.unique(data.participant))})
            self.parameters_info.update({'n_parameters_group': n_parameters_individual*2,
                                         'n_parameters_hierarchical': n_parameters_individual*2 + n_parameters_individual*self.data_info['L']})

            r = re.compile("transf_.+")
            parameters_names_transf = list(filter(r.match, self.stan_model.flatnames))
            individual_parameters_names = [name[10:] for name in parameters_names_transf]

            r = re.compile("mu_.+")
            group_parameters_mu = list(filter(r.match, self.stan_model.flatnames))
            r = re.compile("sd_.+")
            group_parameters_sd = list(filter(r.match, self.stan_model.flatnames))

            group_parameters_names_transf = parameters_names_transf + group_parameters_sd # add transformed par names for plotting
            group_parameters_names = group_parameters_mu + group_parameters_sd

            r = re.compile("z_.+_trial.+")
            trials_deviations = list(filter(r.match, self.stan_model.flatnames))

            r = re.compile("z_.+")
            individual_deviations = list(filter(r.match, self.stan_model.flatnames))
            if len(trials_deviations) > 0:
                [individual_deviations.remove(el) for el in trials_deviations]

            parameters_names = group_parameters_names + individual_deviations
            parameters_names_all = parameters_names + trials_deviations

            self.parameters_info.update({'parameters_names': parameters_names, # group parameters and individual deviations
                                         'group_parameters_names': group_parameters_names, # group parameters
                                         'individual_parameters_names': individual_parameters_names, # names of individual parameters
                                         'group_parameters_names_transf': parameters_names_transf, # group parameters for plotting
                                         'parameters_names_all': parameters_names_all}) # all parameters for the rhat calculations

        else:
            self.data_info.update({'L': 1})

            r = re.compile("transf_.+")
            parameters_names_transf = list(filter(r.match, self.stan_model.flatnames))
            parameters_names = [name[7:] for name in parameters_names_transf]
            r = re.compile("z_.+_trial.+")
            parameters_names_all = parameters_names + list(filter(r.match, self.stan_model.flatnames))

            self.parameters_info.update({'parameters_names': parameters_names})
            self.parameters_info.update({'parameters_names_transf': parameters_names_transf}) # add transformed par names for plotting
            self.parameters_info.update({'parameters_names_all': parameters_names_all}) # for the rhat calculations

    def get_rhat(self):
        """Extracts rhat from stan model's summary as a pandas dataframe.
        Only considers parameters (Not all variables specified in stan's model).
        Note that, when DDM parameters are estimated at a trial level, these are included in the rhat stats.

        Returns
        -------

        convergence: DataFrame
            Data frame with rows the parameters and columns the rhat and variable names.

        """
        summary = self.stan_model.summary(pars=self.parameters_info['parameters_names_all'])
        convergence = pd.DataFrame({'rhat': np.array(summary['summary'])[:, 9],
                                    'variable': summary['summary_rownames']})
        return convergence

    def calculate_waic(self, pointwise=False):
        """Calculates the Watanabe-Akaike information criteria.

        Calculates pWAIC1 and pWAIC2
        according to http://www.stat.columbia.edu/~gelman/research/published/waic_understand3.pdf

        Parameters
        ----------

        pointwise : bool, default to False
            By default, gives the averaged waic.
            Set to True is you want additional waic per observation.

        Returns
        -------

        out: dict
            Dictionary containing lppd (log pointwise predictive density),
            p_waic, waic, waic_se (standard error of the waic), and
            pointwise_waic (when `pointwise` is True).

        """
        log_likelihood = self.stan_model['log_lik'] # n_samples X N observations
        likelihood = np.exp(log_likelihood)

        mean_l = np.mean(likelihood, axis=0) # N observations

        pointwise_lppd = np.log(mean_l)
        lppd = np.sum(pointwise_lppd)

        pointwise_var_l = np.var(log_likelihood, axis=0) # N observations
        var_l = np.sum(pointwise_var_l)

        pointwise_waic = - 2*pointwise_lppd +  2*pointwise_var_l
        waic = -2*lppd + 2*var_l
        waic_se = np.sqrt(self.data_info['N'] * np.var(pointwise_waic))

        if pointwise:
            out = {'lppd':lppd,
                   'p_waic':var_l,
                   'waic':waic,
                   'waic_se':waic_se,
                   'pointwise_waic':pointwise_waic}
        else:
            out = {'lppd':lppd,
                   'p_waic':var_l,
                   'waic':waic,
                   'waic_se':waic_se}
        return out

    def get_last_values(self):
        """Extracts the last posterior estimates values in each chain.

        Returns
        -------

        starting_points: DataFrame
             Data frame with as many rows as number of chains that were run.
             Parameter values are in separate columns.

        """
        samplesChains = self.stan_model.to_dataframe(pars=self.parameters_info['parameters_names_all'],
                                                     permuted=False,
                                                     diagnostics=False)
        starting_points = samplesChains[samplesChains['draw'] == max(samplesChains['draw'])]

        return starting_points

class ModelResults(object):
    def __init__(self,
                 model_label,
                 data_info,
                 parameters_info,
                 priors,
                 rhat,
                 waic,
                 last_values,
                 samples,
                 trial_samples):
        """Initiates a ModelResults object.

        Parameters
        ----------

        Attributes
        ----------

        """
        self.model_label = model_label
        self.data_info = data_info
        self.parameters_info = parameters_info
        self.priors = priors
        self.rhat = rhat
        self.waic = waic
        self.last_values = last_values
        self.samples = samples
        self.trial_samples = trial_samples

    def to_pickle(self, filename=None):
        """Pickle the fitted model's results object to file.

        This can be used to store the model's result
        and read them and inspect them at a later stage,
        without having to refit the model.

        Parameters
        ----------

        filename : str, optional
            File path where the pickled object will be stored.
            If not specified, is set to

        """

        dir_path = os.getcwd()#os.path.dirname(os.path.realpath(__file__))

        if filename is None:
            filename = os.path.join(dir_path, '{}.pkl'.format(self.model_label))
            print("Saving file as: {}".format(filename))

        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()

    def plot_posteriors(self,
                        gridsize=100,
                        clip=None,
                        show_intervals="HDI",
                        alpha_intervals=.05,
                        intervals_kws=None,
                        **kwargs):
        """Plots posterior predictives of the model's parameters.

        If the model is hierarchical, then only the group parameters are plotted.
        In particular, group means are plotted in the first row
        and group standard deviations are plotted in the second row.
        By default, 95 percent HDI are shown.
        The kernel density estimation is calculated using scipy.stats.gaussian_kde.

        Parameters
        ----------

        gridsize : int, default to 100
            Resolution of the kernel density estimation function, default to 100.

        clip : tuple of (float, float), optional
            Range for the kernel density estimation function.
            Default is min and max values of the distribution.

        show_intervals : str, default to "HDI"
            Either "HDI", "BCI", or None.
            HDI is better when the distribution is not simmetrical.
            If None, then no intervals are shown.

        alpha_intervals : float, default to .05
            Alpha level for the intervals.
            Default is 5 percent which gives 95 percent BCIs and HDIs.

        intervals_kws : dict
            Additional arguments for `matplotlib.axes.Axes.fill_between`
            that shows shaded intervals.
            By default, they are 50 percent transparent.

        Other Parameters
        ----------------

        **kwargs
            Additional parameters for seaborn.FacetGrid.

        Returns
        -------

        g : seaborn.FacetGrid

        """
        if self.parameters_info['hierarchical_levels'] == 2:
            cols = self.parameters_info['group_parameters_names_transf']
        else:
            cols = self.parameters_info['parameters_names_transf']

        dfm = pd.melt(self.samples[cols], value_vars=cols)
        g = sns.FacetGrid(dfm,
                          col="variable",
                          col_wrap=self.parameters_info['n_parameters_individual'],
                          sharex=False,
                          **kwargs)
        g.map(plotting.plot_posterior,
              "value",
              gridsize=gridsize,
              clip=clip,
              show_intervals=show_intervals,
              alpha_intervals=alpha_intervals,
              intervals_kws=intervals_kws)
        return g
