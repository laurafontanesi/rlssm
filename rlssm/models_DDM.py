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

        # Define default priors
        if self.hierarchical_levels == 1:
            self.priors = dict(
                drift_priors={'mu':1, 'sd':5},
                threshold_priors={'mu':0, 'sd':5},
                ndt_priors={'mu':0, 'sd':5},
                rel_sp_priors={'mu':0, 'sd':.8},
                drift_trialmu_priors={'mu':1, 'sd':5},
                drift_trialsd_priors={'mu':0, 'sd':5},
                rel_sp_trialmu_priors={'mu':0, 'sd':.8},
                rel_sp_trialsd_priors={'mu':0, 'sd':.5},
                beta_trialmu_priors={'mu':0, 'sd':10},
                beta_trialsd_priors={'mu':0, 'sd':10},
                regression_coefficients_priors={'mu':0, 'sd':5},
                corr_matrix_prior=1
                )
        else:
            self.priors = dict(
                drift_priors={'mu_mu':1, 'sd_mu':5, 'mu_sd':0, 'sd_sd':5},
                threshold_priors={'mu_mu':1, 'sd_mu':3, 'mu_sd':0, 'sd_sd':3},
                ndt_priors={'mu_mu':1, 'sd_mu':1, 'mu_sd':0, 'sd_sd':1},
                rel_sp_priors={'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':1},
                drift_trialmu_priors={'mu_mu':1, 'sd_mu':5, 'mu_sd':0, 'sd_sd':5},
                drift_trialsd_priors={'mu':0, 'sd':3},
                rel_sp_trialmu_priors={'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':1},
                rel_sp_trialsd_priors={'mu':0, 'sd':3},
                beta_trialmu_priors={'mu':0, 'sd':10},
                beta_trialsd_priors={'mu':0, 'sd':10},
                regression_coefficients_priors={'mu':0, 'sd':5},
                corr_matrix_prior=1
                )

        # Set up model label and priors for mechanisms
        if self.starting_point_bias:
            self.model_label += '_bias'
            self.n_parameters_individual += 1
        else:
            del self.priors['rel_sp_priors']

        if self.drift_variability:
            self.model_label += '_driftvar'
            self.n_parameters_individual += 1
            self.n_parameters_trial += 1
            del self.priors['drift_priors']
        else:
            del self.priors['drift_trialmu_priors']
            del self.priors['drift_trialsd_priors']

        if self.starting_point_variability:
            self.model_label += '_spvar'
            self.n_parameters_individual += 1
            self.n_parameters_trial += 1
            # when you are estimating both mean and sd
            if self.starting_point_bias:
                del self.priors['rel_sp_priors']
        else:
            del self.priors['rel_sp_trialmu_priors']
            del self.priors['rel_sp_trialsd_priors']

        # for nDDM or hDDM
        if self.drift_starting_point_correlation and not self.drift_starting_point_beta_correlation:
            self.model_label += '_corr'
            # add the corr coefficient
            self.n_parameters_individual += 1
            del self.priors['beta_trialmu_priors']
            del self.priors['beta_trialsd_priors']
            del self.priors['regression_coefficients_priors']

        elif self.drift_starting_point_beta_correlation:
            self.model_label += '_corr_beta'
            # add 3 correlation coefficients, plus mean and sd of the beta variable
            self.n_parameters_individual += 5
            del self.priors['regression_coefficients_priors']

        elif self.drift_starting_point_regression:
            self.model_label += '_beta'
            # add 2 correlation coefficients
            self.n_parameters_individual += 2
            del self.priors['beta_trialmu_priors']
            del self.priors['beta_trialsd_priors']
            del self.priors['corr_matrix_prior']
        else:
            del self.priors['beta_trialmu_priors']
            del self.priors['beta_trialsd_priors']
            del self.priors['regression_coefficients_priors']
            del self.priors['corr_matrix_prior']

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
            corr_matrix_prior=None,
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

        # change default priors:
        if drift_priors is not None:
            self.priors['drift_priors'] = drift_priors
        if threshold_priors is not None:
            self.priors['threshold_priors'] = threshold_priors
        if ndt_priors is not None:
            self.priors['ndt_priors'] = ndt_priors
        if rel_sp_priors is not None:
            self.priors['rel_sp_priors'] = rel_sp_priors
        if drift_trialmu_priors is not None:
            self.priors['drift_trialmu_priors'] = drift_trialmu_priors
        if drift_trialsd_priors is not None:
            self.priors['drift_trialsd_priors'] = drift_trialsd_priors
        if rel_sp_trialmu_priors is not None:
            self.priors['rel_sp_trialmu_priors'] = rel_sp_trialmu_priors
        if rel_sp_trialsd_priors is not None:
            self.priors['rel_sp_trialsd_priors'] = rel_sp_trialsd_priors
        if beta_trialmu_priors is not None:
            self.priors['beta_trialmu_priors'] = beta_trialmu_priors
        if beta_trialsd_priors is not None:
            self.priors['beta_trialsd_priors'] = beta_trialsd_priors
        if regression_coefficients_priors is not None:
            self.priors['regression_coefficients_priors'] = regression_coefficients_priors
        if corr_matrix_prior is not None:
            self.priors['corr_matrix_prior'] = corr_matrix_prior

        data_dict = {'N': N,
                     'rt': data['rt'].values,
                     'accuracy': data['accuracy_neg'].values.astype(int),
                     'starting_point': starting_point}

        if self.hierarchical_levels == 2:
            keys_priors = ["mu_mu", "sd_mu", "mu_sd", "sd_sd"]
            L = len(pd.unique(data.participant)) # n subjects (levels)
            data_dict.update({'L': L, 
                              'participant': data['participant'].values.astype(int)})
        else:
            keys_priors = ["mu", "sd"]

        # Add data for mechanisms:
        # starting point bias priors
        if self.starting_point_bias:
            data_dict.update({'accuracy_flipped': data['accuracy_flipped'].values.astype(int)})
            del data_dict['starting_point']
        elif self.starting_point_variability:
            data_dict.update({'accuracy_flipped': data['accuracy_flipped'].values.astype(int)})

        # for nDDM or hDDM
        if self.drift_starting_point_correlation and not self.drift_starting_point_beta_correlation:
            data_dict.update({'n_cor_par': 2})
        elif self.drift_starting_point_beta_correlation:
            data_dict.update({'n_cor_par': 3,
                              'beta': data['beta'].values})
        elif self.drift_starting_point_regression:
            data_dict.update({'beta': data['beta'].values})

        # Add priors:
        print("Fitting the model using the priors:")
        for par in self.priors.keys():
            data_dict.update({par: [self.priors[par][key] for key in keys_priors]})
            print(par, self.priors[par])

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
                                      self.priors,
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

        # Define default priors
        if self.hierarchical_levels == 1:
            self.priors = dict(
                alpha_priors={'mu':0, 'sd':1},
                alpha_pos_priors={'mu':0, 'sd':1},
                alpha_neg_priors={'mu':0, 'sd':1},
                drift_scaling_priors={'mu':1, 'sd':50},
                drift_asymptote_priors={'mu':1, 'sd':50},
                threshold_priors={'mu':1, 'sd':5},
                threshold_modulation_priors={'mu':0, 'sd':10},
                ndt_priors={'mu':1, 'sd':1}
                )
        else:
            self.priors = dict(
                alpha_priors={'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':.1},
                alpha_pos_priors={'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':.1},
                alpha_neg_priors={'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':.1},
                drift_scaling_priors={'mu_mu':1, 'sd_mu':30, 'mu_sd':0, 'sd_sd':30},
                drift_asymptote_priors={'mu_mu':1, 'sd_mu':30, 'mu_sd':0, 'sd_sd':30},
                threshold_priors={'mu_mu':1, 'sd_mu':3, 'mu_sd':0, 'sd_sd':3},
                threshold_modulation_priors={'mu_mu':0, 'sd_mu':10, 'mu_sd':0, 'sd_sd':10},
                ndt_priors={'mu_mu':1, 'sd_mu':1, 'mu_sd':0, 'sd_sd':1}
                )

        if self.nonlinear_mapping:
            self.model_label += '_nonlin'
            self.n_parameters_individual += 1
        else:
            del self.priors['drift_asymptote_priors']

        if self.separate_learning_rates:
            self.model_label += '_2lr'
            self.n_parameters_individual += 1
            del self.priors['alpha_priors']
        else:
            del self.priors['alpha_pos_priors']
            del self.priors['alpha_neg_priors']

        if self.threshold_modulation:
            self.model_label += '_thrmod'
            self.n_parameters_individual += 1
        else:
            del self.priors['threshold_modulation_priors']

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

        # change default priors:
        if alpha_priors is not None:
            self.priors['alpha_priors'] = alpha_priors
        if alpha_pos_priors is not None:
            self.priors['alpha_pos_priors'] = alpha_pos_priors
        if alpha_neg_priors is not None:
            self.priors['alpha_neg_priors'] = alpha_neg_priors
        if drift_scaling_priors is not None:
            self.priors['drift_scaling_priors'] = drift_scaling_priors
        if drift_asymptote_priors is not None:
            self.priors['drift_asymptote_priors'] = drift_asymptote_priors
        if threshold_priors is not None:
            self.priors['threshold_priors'] = threshold_priors
        if threshold_modulation_priors is not None:
            self.priors['threshold_modulation_priors'] = threshold_modulation_priors
        if ndt_priors is not None:
            self.priors['ndt_priors'] = ndt_priors

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
                     'starting_point': .5}

        if self.hierarchical_levels == 2:
            keys_priors = ["mu_mu", "sd_mu", "mu_sd", "sd_sd"]
            L = len(pd.unique(data.participant)) # n subjects (levels)
            data_dict.update({'L': L, 
                              'participant': data['participant'].values.astype(int)})
        else:
            keys_priors = ["mu", "sd"]

        # Add priors:
        print("Fitting the model using the priors:")
        for par in self.priors.keys():
            data_dict.update({par: [self.priors[par][key] for key in keys_priors]})
            print(par, self.priors[par])

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
                                      self.priors,
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