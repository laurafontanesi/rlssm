from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from .models import Model
from .fits_DDM import DDMFittedModel


class DDModel(Model):
    """DDModel allows to specify a diffusion decision model.

    When initializing the model, you should specify whether the model is hierarchical or not.
    Additionally, you can specify the mechanisms that you wish to include or exclude.

    The underlying stan model will be compiled if no previously compiled model is found.
    After initializing the model, it can be fitted to a particular dataset using pystan.

    """
    def __init__(self,
                     hierarchical_levels,
                     starting_point_bias=False,
                     drift_variability=False,
                     starting_point_variability=False,
                     drift_starting_point_correlation=False,
                     drift_starting_point_beta_correlation=False,
                     drift_starting_point_regression=False):
        """Initialize a DDModel object.

        Note
        ----
        This model is restricted to two options per trial (coded as correct and incorrect).

        Parameters
        ----------

        hierarchical_levels : int
            Set to 1 for individual data and to 2 for grouped data.

        starting_point_bias : bool, default False
            By default, there is no starting point bias.
            If set to True, the starting point bias is estimated.

        drift_variability : bool, default False
            By default, there is no drift-rate variability across trials.
            If set to True, the standard deviation of the drift-rate across trials is estimated.

        starting_point_variability : bool, default False
            By default, there is no starting point bias variability across trials.
            If set to True, the standard deviation of the starting point bias across trials
            is estimated.

        drift_starting_point_correlation : bool, default False
            By default, the correlation between these 2 parameters is not estimated.
            If set to True, the 2 parameters are assumed to come
            from a multinormal distribution.
            Only relevant when drift_variability and starting_point_variability are True.

        drift_starting_point_beta_correlation : bool, default False
            If True, trial-by-trial drift-rate, rel_sp and an external
            variable beta are assumed to come from a multinormal distribution.
                 Only relevant when drift_variability and starting_point_variability are True.

        drift_starting_point_regression : bool, default False
            If True, two regression coefficients are estimated, for trial drift
            and relative starting point, and an external variable beta.
            Only relevant when drift_variability and starting_point_variability are True.

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
        super().__init__(hierarchical_levels, "DDM")

        # Define the model parameters
        self.starting_point_bias = starting_point_bias
        self.drift_variability = drift_variability
        self.starting_point_variability = starting_point_variability
        self.drift_starting_point_correlation = drift_starting_point_correlation
        self.drift_starting_point_beta_correlation = drift_starting_point_beta_correlation
        self.drift_starting_point_regression = drift_starting_point_regression

        self.n_parameters_individual = 3
        self.n_parameters_trial = 0

        if self.starting_point_bias:
            self.model_label += '_bias'
            self.n_parameters_individual += 1

        if self.drift_variability:
            self.model_label += '_driftvar'
            self.n_parameters_individual += 1
            self.n_parameters_trial += 1

        if self.starting_point_variability:
            self.model_label += '_spvar'
            self.n_parameters_individual += 1
            self.n_parameters_trial += 1

        if self.drift_starting_point_correlation and not self.drift_starting_point_beta_correlation:
            self.model_label += '_corr'
            # add the corr coefficient
            self.n_parameters_individual += 1

        if self.drift_starting_point_beta_correlation:
            self.model_label += '_corr_beta'
            # add 3 correlation coefficients, plus mean and sd of the beta variable
            self.n_parameters_individual += 5

        if self.drift_starting_point_regression:
            self.model_label += '_beta'
            # add 2 correlation coefficients
            self.n_parameters_individual += 2

        # Set the stan model path
        self._set_model_path()

        # Finally, compile the model
        self._compile_stan_model()

    def fit(self,
            data,
            drift_priors=None,
            threshold_priors=None,
            ndt_priors=None,
            rel_sp_priors=None,
            starting_point=.5,
            drift_trialmu_priors=None,
            drift_trialsd_priors=None,
            rel_sp_trialmu_priors=None,
            rel_sp_trialsd_priors=None,
            corr_matrix_prior=1,
            beta_trialmu_priors=None,
            beta_trialsd_priors=None,
            regression_coefficients_priors=None,
            include_rhat=True,
            include_waic=True,
            include_last_values=True,
            pointwise_waic=False,
            print_diagnostics=True,
            **kwargs):
        """Fits the specified diffusion decision model to data.

        Parameters
        ----------

        data : DataFrame
            A pandas DataFrame containing data observations.

            Columns should include:

            - *rt*, response times in seconds.

            - *accuracy*, 0 if the incorrect option was chosen,
              1 if the correct option was chosen.

            If the model is hierarchical, also include:

            - *participant*, the participant number.
              Should be integers starting from 1.

            When either drift_starting_point_correlation, drift_starting_point_beta_correlation,
            or drift_starting_point_regression are True, also include:

            - *beta*, the external variable to correlate/regress to drift and rel_sp.

        Returns
        -------
        res : rlssm.fits.DDModelResults

        Other Parameters
        ----------------

        starting_point : float, default .5
            The relative starting point of the diffusion process.
            By default there is no bias, so the starting point is .5.
            Should be between 0 and 1.

        drift_priors : dict, optional
            Priors for the drift-rate parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        threshold_priors : dict, optional
            Priors for the threshold parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        ndt_priors : dict, optional
            Priors for the non decision time parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        rel_sp_priors : dict, optional
            Priors for the relative starting point parameter
            (only meaningful if starting_point_bias is True).
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        drift_trialmu_priors : dict, optional
            Priors for the mean drift-rate across trials
            (only meaningful if drift_variability is True).
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        drift_trialsd_priors : dict, optional
            Priors for the standard deviation of the drift-rate across trials
            (only meaningful if drift_variability is True).
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        rel_sp_trialmu_priors : dict, optional
            Priors for the standard deviation of the relative starting point across trials
            (only meaningful if starting_point_variability is True).
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        rel_sp_trialsd_priors : dict, optional
            Priors for the standard deviation of the relative starting point across trials
            (only meaningful if starting_point_variability is True).
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        corr_matrix_prior : float, default to 1
            Prior for the eta parameter of the LKJ prior of the correlation matrix
            (only meaningful if drift_starting_point_correlation is True).

        beta_trialmu_priors : dict, optional
            Priors for the mean beta across trials
            (only meaningful if drift_starting_point_beta_correlation is True).
            Mean and standard deviation of the prior distr.

        beta_trialsd_priors : dict, optional
            Priors for the standard deviation of the beta across trials
            (only meaningful if drift_starting_point_beta_correlation is True).
            Mean and standard deviation of the prior distr.

        regression_coefficients_priors : dict, optional
            Priors for the regression coefficients
            (only relevant if drift_starting_point_regression is True).
            Mean and standard deviation of the prior distr.

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

        # transform data variables
        data['accuracy_neg'] = -1
        data.loc[data.accuracy == 1, 'accuracy_neg'] = 1
        data['accuracy_flipped'] = -(data['accuracy']-1)

        if self.hierarchical_levels == 2:
            # set default priors for the hierarchical model:
            if drift_priors is None:
                drift_priors = {'mu_mu':1, 'sd_mu':5, 'mu_sd':0, 'sd_sd':5}
            if threshold_priors is None:
                threshold_priors = {'mu_mu':1, 'sd_mu':3, 'mu_sd':0, 'sd_sd':3}
            if ndt_priors is None:
                ndt_priors = {'mu_mu':1, 'sd_mu':1, 'mu_sd':0, 'sd_sd':1}
            if rel_sp_priors is None:
                rel_sp_priors = {'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':1}
            if drift_trialmu_priors is None:
                drift_trialmu_priors = {'mu_mu':1, 'sd_mu':5, 'mu_sd':0, 'sd_sd':5}
            if drift_trialsd_priors is None:
                drift_trialsd_priors = {'mu':0, 'sd':3}
            if rel_sp_trialmu_priors is None:
                rel_sp_trialmu_priors = {'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':1}
            if rel_sp_trialsd_priors is None:
                rel_sp_trialsd_priors = {'mu':0, 'sd':3}

            L = len(pd.unique(data.participant)) # n subjects (levels)
            data_dict = {'N': N,
                         'L': L,
                         'participant': data['participant'].values.astype(int),
                         'rt': data['rt'].values,
                         'accuracy': data['accuracy_neg'].values.astype(int),
                         'drift_priors': [drift_priors['mu_mu'],
                                          drift_priors['sd_mu'],
                                          drift_priors['mu_sd'],
                                          drift_priors['sd_sd']],
                         'threshold_priors': [threshold_priors['mu_mu'],
                                              threshold_priors['sd_mu'],
                                              threshold_priors['mu_sd'],
                                              threshold_priors['sd_sd']],
                         'ndt_priors': [ndt_priors['mu_mu'],
                                        ndt_priors['sd_mu'],
                                        ndt_priors['mu_sd'],
                                        ndt_priors['sd_sd']],
                         'starting_point': starting_point
                        }

            # adjust priors for more complex models
            if self.starting_point_bias:
                data_dict.update({'rel_sp_priors': [rel_sp_priors['mu_mu'],
                                                    rel_sp_priors['sd_mu'],
                                                    rel_sp_priors['mu_sd'],
                                                    rel_sp_priors['sd_sd']],
                                  'accuracy_flipped': data['accuracy_flipped'].values.astype(int)})
                del data_dict['starting_point']

                if self.starting_point_variability:
                    data_dict.update({'rel_sp_trialmu_priors': [rel_sp_trialmu_priors['mu_mu'],
                                                                rel_sp_trialmu_priors['sd_mu'],
                                                                rel_sp_trialmu_priors['mu_sd'],
                                                                rel_sp_trialmu_priors['sd_sd']],
                                      'rel_sp_trialsd_priors': [rel_sp_trialsd_priors['mu'],
                                                                rel_sp_trialsd_priors['sd']]})
                    del data_dict['rel_sp_priors']

            else:
                if self.starting_point_variability:
                    data_dict.update({'rel_sp_trialsd_priors': [rel_sp_trialsd_priors['mu'],
                                                                rel_sp_trialsd_priors['sd']],
                                      'accuracy_flipped': data['accuracy_flipped'].values.astype(int)})

            if self.drift_variability:
                data_dict.update({'drift_trialmu_priors': [drift_trialmu_priors['mu_mu'],
                                                           drift_trialmu_priors['sd_mu'],
                                                           drift_trialmu_priors['mu_sd'],
                                                           drift_trialmu_priors['sd_sd']],
                                  'drift_trialsd_priors': [drift_trialsd_priors['mu'],
                                                           drift_trialsd_priors['sd']]})
                del data_dict['drift_priors']

        else:
            # set default priors for the non-hierarchical model:
            if drift_priors is None:
                drift_priors = {'mu':1, 'sd':5}
            if threshold_priors is None:
                threshold_priors = {'mu':0, 'sd':5}
            if ndt_priors is None:
                ndt_priors = {'mu':0, 'sd':5}
            if rel_sp_priors is None:
                rel_sp_priors = {'mu':0, 'sd':.8}
            if drift_trialmu_priors is None:
                drift_trialmu_priors = {'mu':1, 'sd':5}
            if drift_trialsd_priors is None:
                drift_trialsd_priors = {'mu':0, 'sd':5}
            if rel_sp_trialmu_priors is None:
                rel_sp_trialmu_priors = {'mu':0, 'sd':.8}
            if rel_sp_trialsd_priors is None:
                rel_sp_trialsd_priors = {'mu':0, 'sd':.5}
            if beta_trialmu_priors is None:
                beta_trialmu_priors = {'mu':0, 'sd':10}
            if beta_trialsd_priors is None:
                beta_trialsd_priors = {'mu':0, 'sd':10}
            if regression_coefficients_priors is None:
                regression_coefficients_priors = {'mu':0, 'sd':5}

            data_dict = {'N': N,
                         'rt': data['rt'].values,
                         'accuracy': data['accuracy_neg'].values.astype(int),
                         'drift_priors': [drift_priors['mu'], drift_priors['sd']],
                         'threshold_priors': [threshold_priors['mu'], threshold_priors['sd']],
                         'ndt_priors': [ndt_priors['mu'], ndt_priors['sd']],
                         'starting_point': starting_point
                        }
            #adjust priors for more complex models
            if self.starting_point_bias:
                data_dict.update({'rel_sp_priors': [rel_sp_priors['mu'], rel_sp_priors['sd']],
                                  'accuracy_flipped': data['accuracy_flipped'].values.astype(int)})
                del data_dict['starting_point']

                if self.starting_point_variability:
                    data_dict.update({'rel_sp_trialmu_priors': [rel_sp_trialmu_priors['mu'],
                                                                rel_sp_trialmu_priors['sd']],
                                      'rel_sp_trialsd_priors': [rel_sp_trialsd_priors['mu'],
                                                                rel_sp_trialsd_priors['sd']]})
                    del data_dict['rel_sp_priors']

            else:
                if self.starting_point_variability:
                    data_dict.update({'rel_sp_trialsd_priors': [rel_sp_trialsd_priors['mu'],
                                                                rel_sp_trialsd_priors['sd']],
                                      'accuracy_flipped': data['accuracy_flipped'].values.astype(int)})

            if self.drift_variability:
                data_dict.update({'drift_trialmu_priors': [drift_trialmu_priors['mu'],
                                                           drift_trialmu_priors['sd']],
                                  'drift_trialsd_priors': [drift_trialsd_priors['mu'],
                                                           drift_trialsd_priors['sd']]})
                del data_dict['drift_priors']

            if self.drift_starting_point_correlation and not self.drift_starting_point_beta_correlation:
                data_dict.update({'n_cor_par': 2,
                                  'corr_matrix_prior': corr_matrix_prior})
            if self.drift_starting_point_beta_correlation:
                data_dict.update({'n_cor_par': 3,
                                  'corr_matrix_prior': corr_matrix_prior,
                                  'beta': data['beta'].values,
                                  'beta_trialmu_priors': [beta_trialmu_priors['mu'],
                                                          beta_trialmu_priors['sd']],
                                  'beta_trialsd_priors': [beta_trialsd_priors['mu'],
                                                          beta_trialsd_priors['sd']]})
            if self.drift_starting_point_regression:
                data_dict.update({'beta': data['beta'].values,
                                  'regression_coefficients_priors': [regression_coefficients_priors['mu'],
                                                                     regression_coefficients_priors['sd']]})

        # start sampling...
        fitted_model = self.compiled_model.sampling(data_dict, **kwargs)

        fitted_model = DDMFittedModel(fitted_model,
                                      data,
                                      self.hierarchical_levels,
                                      self.model_label,
                                      self.family,
                                      self.n_parameters_individual,
                                      self.n_parameters_trial,
                                      print_diagnostics,
                                      self.starting_point_bias,
                                      self.drift_variability,
                                      self.starting_point_variability,
                                      self.drift_starting_point_correlation,
                                      self.drift_starting_point_beta_correlation,
                                      self.drift_starting_point_regression)
        res = fitted_model.extract_results(include_rhat,
                                           include_waic,
                                           pointwise_waic,
                                           include_last_values)

        return res

class RLDDModel(Model):
    """RLDDModel allows to specify a combination of reinforcement learning
    and diffusion decision models.

    When initializing the model, you should specify whether the model is hierarchical or not.
    Additionally, you can specify the mechanisms that you wish to include or exclude.

    The underlying stan model will be compiled if no previously compiled model is found.
    After initializing the model, it can be fitted to a particular dataset using pystan.

    """
    def __init__(self,
                 hierarchical_levels,
                 nonlinear_mapping=False,
                 separate_learning_rates=False,
                 threshold_modulation=False):
        """Initialize a RLDDModel object.

        Note
        ----
        This model is restricted to two options per trial (coded as correct and incorrect).
        However, more than two options can be presented in the same learning session.

        Parameters
        ----------

        hierarchical_levels : int
             Set to 1 for individual data and to 2 for grouped data.

        nonlinear_mapping : bool, default False
             By default, the mapping between value differences and drift-rate is linear.
             If set to True, a non-linear mapping function is estimated.

        separate_learning_rates : bool, default False
             By default, there is only one learning rate.
             If set to True, separate learning rates are estimated
             for positive and negative prediction errors.

        threshold_modulation : bool, default False
             By default, the threshold is independent on the presented options.
             If set to True, the threshold can decrease or increase
             depending on the average value of the presented options.

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
        super().__init__(hierarchical_levels, "RLDDM")

        # Define the model parameters
        self.nonlinear_mapping = nonlinear_mapping
        self.separate_learning_rates = separate_learning_rates
        self.threshold_modulation = threshold_modulation

        self.n_parameters_individual = 4
        self.n_parameters_trial = 0

        if nonlinear_mapping:
            self.model_label += '_nonlin'
            self.n_parameters_individual += 1

        if separate_learning_rates:
            self.model_label += '_2lr'
            self.n_parameters_individual += 1

        if threshold_modulation:
            self.model_label += '_thrmod'
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
            drift_scaling_priors=None,
            threshold_priors=None,
            ndt_priors=None,
            drift_asymptote_priors=None,
            threshold_modulation_priors=None,
            alpha_pos_priors=None,
            alpha_neg_priors=None,
            include_rhat=True,
            include_waic=True,
            pointwise_waic=False,
            include_last_values=True,
            print_diagnostics=True,
            **kwargs):
        """Fits the specified reinforcement learning diffusion decision model to data.

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

            - *rt*, response times in seconds.

            - *accuracy*, 0 if the incorrect option was chosen,
              1 if the correct option was chosen.

            If the model is hierarchical, also include:

            - *participant*, the participant number.
              Should be integers starting from 1.

        K : int
            Number of options per learning session.

        initial_value_learning : float
            The assumed value expectation in the first learning session.
            The learning value in the following learning sessions is set to
            the average learned value in the previous learning session.

        Returns
        -------
        res : rlssm.fits.DDModelResults

        Other Parameters
        ----------------

        alpha_priors : dict, optional
            Priors for the learning rate parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        drift_scaling_priors : dict, optional
            Priors for the drift scaling parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        threshold_priors : dict, optional
            Priors for the threshold parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        ndt_priors : dict, optional
            Priors for the non decision time parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        drift_asymptote_priors: dict, optional
            Priors for the drift-rate asymptote (only meaningful if nonlinear_mapping is True).
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        threshold_modulation_priors : dict, optional
            Priors for the threshold coefficient (only meaningful if threshold_modulation is True).
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

        # transform data variables
        data['accuracy_neg'] = -1
        data.loc[data.accuracy == 1, 'accuracy_neg'] = 1

        if self.hierarchical_levels == 2:
            # set default priors for the non-hierarchical model:
            if alpha_priors is None:
                alpha_priors = {'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':.1}
            if drift_scaling_priors is None:
                drift_scaling_priors = {'mu_mu':1, 'sd_mu':30, 'mu_sd':0, 'sd_sd':30}
            if threshold_priors is None:
                threshold_priors = {'mu_mu':1, 'sd_mu':3, 'mu_sd':0, 'sd_sd':3}
            if ndt_priors is None:
                ndt_priors = {'mu_mu':1, 'sd_mu':1, 'mu_sd':0, 'sd_sd':1}
            if drift_asymptote_priors is None:
                drift_asymptote_priors = {'mu_mu':1, 'sd_mu':30, 'mu_sd':0, 'sd_sd':30}
            if alpha_pos_priors is None:
                alpha_pos_priors = {'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':.1}
            if alpha_neg_priors is None:
                alpha_neg_priors = {'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':.1}
            if threshold_modulation_priors is None:
                threshold_modulation_priors = {'mu_mu':0, 'sd_mu':10, 'mu_sd':0, 'sd_sd':10}

            L = len(pd.unique(data.participant)) # n subjects (levels)
            data_dict = {'N': N,
                         'K': K,
                         'L': L,
                         'participant': data['participant'].values.astype(int),
                         'trial_block': data['trial_block'].values.astype(int),
                         'f_cor': data['f_cor'].values,
                         'f_inc': data['f_inc'].values,
                         'cor_option': data['cor_option'].values.astype(int),
                         'inc_option': data['inc_option'].values.astype(int),
                         'block_label': data['block_label'].values.astype(int),
                         'rt': data['rt'].values,
                         'accuracy': data['accuracy_neg'].values.astype(int),
                         'initial_value': initial_value_learning,
                         'alpha_priors': [alpha_priors['mu_mu'],
                                          alpha_priors['sd_mu'],
                                          alpha_priors['mu_sd'],
                                          alpha_priors['sd_sd']],
                         'drift_scaling_priors': [drift_scaling_priors['mu_mu'],
                                                  drift_scaling_priors['sd_mu'],
                                                  drift_scaling_priors['mu_sd'],
                                                  drift_scaling_priors['sd_sd']],
                         'threshold_priors': [threshold_priors['mu_mu'],
                                              threshold_priors['sd_mu'],
                                              threshold_priors['mu_sd'],
                                              threshold_priors['sd_sd']],
                         'ndt_priors': [ndt_priors['mu_mu'],
                                        ndt_priors['sd_mu'],
                                        ndt_priors['mu_sd'],
                                        ndt_priors['sd_sd']],
                         'starting_point': .5
                        }

            # adjust priors for more complex models
            if self.nonlinear_mapping:
                data_dict.update({'drift_asymptote_priors': [drift_asymptote_priors['mu_mu'],
                                                             drift_asymptote_priors['sd_mu'],
                                                             drift_asymptote_priors['mu_sd'],
                                                             drift_asymptote_priors['sd_sd']]})
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
            if self.threshold_modulation:
                data_dict.update({'threshold_modulation_priors':
                                  [threshold_modulation_priors['mu_mu'],
                                   threshold_modulation_priors['sd_mu'],
                                   threshold_modulation_priors['mu_sd'],
                                   threshold_modulation_priors['sd_sd']]})

        else:
            # set default priors for the non-hierarchical model:
            if alpha_priors is None:
                alpha_priors = {'mu':0, 'sd':1}
            if drift_scaling_priors is None:
                drift_scaling_priors = {'mu':1, 'sd':50}
            if threshold_priors is None:
                threshold_priors = {'mu':1, 'sd':5}
            if ndt_priors is None:
                ndt_priors = {'mu':1, 'sd':1}
            if drift_asymptote_priors is None:
                drift_asymptote_priors = {'mu':1, 'sd':50}
            if alpha_pos_priors is None:
                alpha_pos_priors = {'mu':0, 'sd':1}
            if alpha_neg_priors is None:
                alpha_neg_priors = {'mu':0, 'sd':1}
            if threshold_modulation_priors is None:
                threshold_modulation_priors = {'mu':0, 'sd':10}

            data_dict = {'N': N,
                         'K': K,
                         'trial_block': data['trial_block'].values.astype(int),
                         'f_cor': data['f_cor'].values,
                         'f_inc': data['f_inc'].values,
                         'cor_option': data['cor_option'].values.astype(int),
                         'inc_option': data['inc_option'].values.astype(int),
                         'block_label': data['block_label'].values.astype(int),
                         'rt': data['rt'].values,
                         'accuracy': data['accuracy_neg'].values.astype(int),
                         'initial_value': initial_value_learning,
                         'alpha_priors': [alpha_priors['mu'], alpha_priors['sd']],
                         'drift_scaling_priors': [drift_scaling_priors['mu'],
                                                  drift_scaling_priors['sd']],
                         'threshold_priors': [threshold_priors['mu'], threshold_priors['sd']],
                         'ndt_priors': [ndt_priors['mu'], ndt_priors['sd']],
                         'starting_point': .5
                        }
            # adjust priors for more complex models
            if self.nonlinear_mapping:
                data_dict.update({'drift_asymptote_priors': [drift_asymptote_priors['mu'],
                                                             drift_asymptote_priors['sd']]})
            if self.separate_learning_rates:
                data_dict.update({'alpha_pos_priors': [alpha_pos_priors['mu'],
                                                       alpha_pos_priors['sd']],
                                  'alpha_neg_priors': [alpha_neg_priors['mu'],
                                                       alpha_neg_priors['sd']]})
                del data_dict['alpha_priors']
            if self.threshold_modulation:
                data_dict.update({'threshold_modulation_priors':
                                  [threshold_modulation_priors['mu'],
                                   threshold_modulation_priors['sd']]})

        # start sampling...
        fitted_model = self.compiled_model.sampling(data_dict, **kwargs)

        fitted_model = DDMFittedModel(fitted_model,
                                      data,
                                      self.hierarchical_levels,
                                      self.model_label,
                                      self.family,
                                      self.n_parameters_individual,
                                      self.n_parameters_trial,
                                      print_diagnostics,
                                      False,
                                      False,
                                      False,
                                      False,
                                      False,
                                      False)
        res = fitted_model.extract_results(include_rhat,
                                           include_waic,
                                           pointwise_waic,
                                           include_last_values)

        return res
