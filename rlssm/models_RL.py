from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from .models import Model
from .fits_RL import RLFittedModel_2A


class RLModel_2A(Model):
    """RLModel_2A allows to specify a reinforcement learning model.

    When initializing the model, you should specify whether the model is hierarchical or not.
    Additionally, you can specify the mechanisms that you wish to include or exclude.

    The underlying stan model will be compiled if no previously compiled model is found.
    After initializing the model, it can be fitted to a particular dataset using pystan.

    """
    def __init__(self,
                 hierarchical_levels,
                 increasing_sensitivity=False,
                 separate_learning_rates=False):
        """Initialize a RLModel_2A object.

        Note
        ----
        This model is restricted to two options per trial (coded as correct and incorrect).
        However, more than two options can be presented in the same learning session.

        Parameters
        ----------
        hierarchical_levels : int
            Set to 1 for individual data and to 2 for grouped data.

        increasing_sensitivity : bool, default False
            By default, sensitivity is fixed throughout learning.
            If set to True, sensitivity increases throughout learning.
            In particular, it increases as a power function of the n times an option has been seen
            (as in Yechiam & Busemeyer, 2005).

        separate_learning_rates : bool, default False
            By default, there is only one learning rate.
            If set to True, separate learning rates are estimated
            for positive and negative prediction errors.

        Attributes
        ----------
        model_label : str
            The label of the fully specified model.

        n_parameters_individual : int
            The number of individual parameters of the fully specified model.

        n_parameters_trial : int
            The number of parameters that are estimated at a trial level.

        stan_model_path : str
            The location of the stan model code.

        compiled_model : pystan.StanModel
            The compiled stan model.

        """
        super().__init__(hierarchical_levels, 'RL_2A')

        # Define the model parameters
        self.increasing_sensitivity = increasing_sensitivity
        self.separate_learning_rates = separate_learning_rates

        self.n_parameters_individual = 2
        self.n_parameters_trial = 0

        if increasing_sensitivity:
            self.model_label += '_pow'
            self.n_parameters_individual += 1

        if separate_learning_rates:
            self.model_label += '_2lr'
            self.n_parameters_individual += 1

        # Set the stan model path
        self._set_model_path()

        # Finally, compile the model
        self._compile_stan_model()

    def fit(self,
                data,
                K,
                initial_value_learning,
                alpha_priors=None,
                sensitivity_priors=None,
                consistency_priors=None,
                scaling_priors=None,
                alpha_pos_priors=None,
                alpha_neg_priors=None,
                include_rhat=True,
                include_waic=True,
                include_last_values=True,
                pointwise_waic=False,
                print_diagnostics=True,
                **kwargs):
        """Fits the specified reinforcement learning model to data.

        Parameters
        ----------
        data : DataFrame
            A pandas DataFrame containing data observations.

            Columns should include (it's OK if some of them are column indexes too):

            - *trial_block*, the number of trial in a learning session.
              Should be integers starting from 1.

            - *f_cor*, the output from the correct option in the presented pair
              (the option with higher outcome on average).

            - *f_inc*, the output from the incorrect option in the presented pair
              (the option with lower outcome on average).

            - *cor_option*, the number identifying the correct option in the presented pair
              (the option with higher outcome on average).

            - *inc_option*, the number identifying the incorrect option in the presented pair
              (the option with lower outcome on average).

            - *block_label*, the number identifying the learning session.
              Should be integers starting from 1.
              Set to 1 in case there is only one learning session.

            - *accuracy*, 0 if the incorrect option was chosen,
              1 if the correct option was chosen.

            If the model is hierarchical, also include:

            - *participant*, the participant number. Should be integers starting from 1.

            If increasing_sensitivity is True, also include:

            - *times_seen*, average number of times the presented options
              have been seen in a learning session.

        K : int
            Number of options per learning session.

        initial_value_learning : float
            The assumed value expectation in the first learning session.
            The learning value in the following learning sessions is set to
            the average learned value in the previous learning session.

        Returns
        -------
        res : rlssm.fits.RLModelResults

        Other Parameters
        ----------------

        alpha_priors : dict, optional
            Priors for the learning rate parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.
            Default is {'mu':0, 'sd':1} for non-hierarchical, and
            {'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':.1} for hierarchical models.

        sensitivity_priors : dict, optional
            Priors for the sensitivity parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        consistency_priors : dict, optional
            Priors for the consistency parameter
            (only meaningful if increasing_sensitivity is True).
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        scaling_priors : dict, optional
            Priors for the scaling parameter (only meaningful if increasing_sensitivity is True).
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        alpha_pos_priors : dict, optional
            Priors for the learning rate for the positive PE
            (only meaningful if separate_learning_rates is True).
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        alpha_neg_priors : dict, optional
            Priors for the learning rate for the negative PE
            (only meaningful if separate_learning_rates is True).
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        include_rhat : bool, default True
            Whether to calculate the Gelman-Rubin convergence diagnostic r hat
            (Gelman & Rubin, 1992).

        include_waic : bool, default True
            Whether to calculate the widely applicable information criterion
            (WAIC; Watanabe, 2013).

        pointwise_waic : bool, default False
            Whether to also inclue the pointwise WAIC.
            Only relevant if include_waic is True.

        include_last_values : bool, default True
            Whether to extract the last values for each chain.

        print_diagnostics : bool, default True
            Whether to print mcmc diagnostics after fitting.
            It is advised to leave it to True and always check, on top of the r hat.

        **kwargs
            Additional arguments to pystan.StanModel.sampling().

        """
        data.reset_index(inplace=True)

        N = data.shape[0] # n observations
        if self.hierarchical_levels == 2:
            # set default priors for the hierarchical model:
            if alpha_priors is None:
                alpha_priors = {'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':.1}
            if sensitivity_priors is None:
                sensitivity_priors = {'mu_mu':1, 'sd_mu':30, 'mu_sd':0, 'sd_sd':30}
            if consistency_priors is None:
                consistency_priors = {'mu_mu':1, 'sd_mu':30, 'mu_sd':0, 'sd_sd':30}
            if scaling_priors is None:
                scaling_priors = {'mu_mu':1, 'sd_mu':30, 'mu_sd':0, 'sd_sd':30}
            if alpha_pos_priors is None:
                alpha_pos_priors = {'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':1}
            if alpha_neg_priors is None:
                alpha_neg_priors = {'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':1}

            L = len(pd.unique(data.participant)) # n subjects (levels)
            data_dict = {'N': N,
                             'L': L,
                             'K': K,
                             'participant': data['participant'].values.astype(int),
                             'trial_block': data['trial_block'].values.astype(int),
                             'f_cor': data['f_cor'].values,
                             'f_inc': data['f_inc'].values,
                             'cor_option': data['cor_option'].values.astype(int),
                             'inc_option': data['inc_option'].values.astype(int),
                             'block_label': data['block_label'].values.astype(int),
                             'accuracy': data['accuracy'].values.astype(int),
                             'initial_value': initial_value_learning,
                             'alpha_priors': [alpha_priors['mu_mu'],
                                              alpha_priors['sd_mu'],
                                              alpha_priors['mu_sd'],
                                              alpha_priors['sd_sd']],
                             'sensitivity_priors': [sensitivity_priors['mu_mu'],
                                                    sensitivity_priors['sd_mu'],
                                                    sensitivity_priors['mu_sd'],
                                                    sensitivity_priors['sd_sd']]}

            # adjust priors for more complex models
            if self.increasing_sensitivity:
                data_dict.update({'consistency_priors': [consistency_priors['mu_mu'],
                                                         consistency_priors['sd_mu'],
                                                         consistency_priors['mu_sd'],
                                                         consistency_priors['sd_sd']],
                                  'scaling_priors': [scaling_priors['mu_mu'],
                                                     scaling_priors['sd_mu'],
                                                     scaling_priors['mu_sd'],
                                                     scaling_priors['sd_sd']]})
                data_dict.update({'times_seen': data['times_seen'].values})
                del data_dict['sensitivity_priors']
            if self.separate_learning_rates:
                data_dict.update({'alpha_pos_priors': [alpha_pos_priors['mu_mu'],
                                                       alpha_pos_priors['sd_mu'],
                                                       alpha_pos_priors['mu_sd'],
                                                       alpha_pos_priors['sd_sd']],
                                  'alpha_neg_priors': [alpha_neg_priors['mu_mu'],
                                                       alpha_neg_priors['sd_mu'],
                                                       alpha_neg_priors['mu_sd'],
                                                       alpha_neg_priors['sd_sd']]})
                del data_dict['alpha_priors']

        else:
            # set default priors for the non-hierarchical model:
            if alpha_priors is None:
                alpha_priors = {'mu':0, 'sd':1}
            if sensitivity_priors is None:
                sensitivity_priors = {'mu':1, 'sd':50}
            if consistency_priors is None:
                consistency_priors = {'mu':1, 'sd':50}
            if scaling_priors is None:
                scaling_priors = {'mu':1, 'sd':50}
            if alpha_pos_priors is None:
                alpha_pos_priors = {'mu':0, 'sd':1}
            if alpha_neg_priors is None:
                alpha_neg_priors = {'mu':0, 'sd':1}

            data_dict = {'N': N,
                         'K': K,
                         'trial_block': data['trial_block'].values.astype(int),
                         'f_cor': data['f_cor'].values,
                         'f_inc': data['f_inc'].values,
                         'cor_option': data['cor_option'].values.astype(int),
                         'inc_option': data['inc_option'].values.astype(int),
                         'block_label': data['block_label'].values.astype(int),
                         'accuracy': data['accuracy'].values.astype(int),
                         'initial_value': initial_value_learning,
                         'alpha_priors': [alpha_priors['mu'], alpha_priors['sd']],
                         'sensitivity_priors': [sensitivity_priors['mu'], sensitivity_priors['sd']]}

            # adjust priors for more complex models
            if self.increasing_sensitivity:
                data_dict.update({'consistency_priors': [consistency_priors['mu'],
                                                         consistency_priors['sd']],
                                  'scaling_priors': [scaling_priors['mu'],
                                                     scaling_priors['sd']]})
                data_dict.update({'times_seen': data['times_seen'].values})
                del data_dict['sensitivity_priors']
            if self.separate_learning_rates:
                data_dict.update({'alpha_pos_priors': [alpha_pos_priors['mu'],
                                                       alpha_pos_priors['sd']],
                                  'alpha_neg_priors': [alpha_neg_priors['mu'],
                                                       alpha_neg_priors['sd']]})
                del data_dict['alpha_priors']

        # start sampling...
        fitted_model = self.compiled_model.sampling(data_dict, **kwargs)

        fitted_model = RLFittedModel_2A(fitted_model,
                                     data,
                                     self.hierarchical_levels,
                                     self.model_label,
                                     self.family,
                                     self.n_parameters_individual,
                                     self.n_parameters_trial,
                                     print_diagnostics)
        res = fitted_model.extract_results(include_rhat,
                                           include_waic,
                                           pointwise_waic,
                                           include_last_values)

        return res
