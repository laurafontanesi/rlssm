from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from .models import Model
from .fits_race import raceFittedModel_2A

class ALBAModel_2A(Model):
    """ALBAModel_2A allows to specify a advantage linear ballistic accumulator model for 2 alternatives.

    When initializing the model, you should specify whether the model is hierarchical or not.

    The underlying stan model will be compiled if no previously compiled model is found.
    After initializing the model, it can be fitted to a particular dataset using pystan.

    """
    def __init__(self, hierarchical_levels):
        """Initialize a ALBAModel_2A object.

        Note
        ----
        This model is restricted to two options per trial (coded as correct and incorrect).

        Parameters
        ----------

        hierarchical_levels : int
            Set to 1 for individual data and to 2 for grouped data.

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

        super().__init__(hierarchical_levels, "ALBA_2A")

        # Define the model parameters

        self.n_parameters_individual = 6 # k, A, tau, v0, ws, wd
        self.n_parameters_trial = 0


        # Set the stan model path
        self._set_model_path()

        # Finally, compile the model
        self._compile_stan_model()

    def fit(self,
            data,
            k_priors=None,
            A_priors=None,
            tau_priors=None,
            v0_priors=None,
            ws_priors=None,
            wd_priors=None,
            include_rhat=True,
            include_waic=True,
            pointwise_waic=False,
            include_last_values=True,
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

        Returns
        -------
        res : rlssm.fits.DDModelResults

        Other Parameters
        ----------------

        starting_point : float, default .5
            The relative starting point of the diffusion process.
            By default there is no bias, so the starting point is .5.
            Should be between 0 and 1.

        k_priors : dict, optional
            Priors for the k parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        A_priors : dict, optional
            Priors for the A parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        tau_priors : dict, optional
            Priors for the non decision time parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        v0_priors : dict, optional
            Priors for the v0 parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        ws_priors : dict, optional
            Priors for the ws parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        wd_priors : dict, optional
            Priors for the wd parameter.
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

        data['accuracy_rescale'] = 2
        data.loc[data.accuracy == 1, 'accuracy_rescale'] = 1

        if self.hierarchical_levels == 2:
            # set default priors for the hierarchical model:
            if k_priors is None:
                k_priors = {'mu_mu':1, 'sd_mu':1, 'mu_sd':0, 'sd_sd':1}
            if A_priors is None:
                A_priors = {'mu_mu':1, 'sd_mu':1, 'mu_sd':0, 'sd_sd':1}
            if tau_priors is None:
                tau_priors = {'mu_mu':1, 'sd_mu':1, 'mu_sd':0, 'sd_sd':1}
            if v0_priors is None:
                v0_priors = {'mu_mu':9, 'sd_mu':3, 'mu_sd':2, 'sd_sd':1}
            if ws_priors is None:
                ws_priors = {'mu_mu':0, 'sd_mu':1, 'mu_sd':2, 'sd_sd':1}
            if wd_priors is None:
                wd_priors = {'mu_mu':3, 'sd_mu':1, 'mu_sd':3, 'sd_sd':1}



            L = len(pd.unique(data.participant)) # n subjects (levels)

            data_dict = {'N': N,
                             'L': L,
                             'participant': data['participant'].values.astype(int),
                             'rt': data['rt'].values,
                             'accuracy': data['accuracy_rescale'].values.astype(int),
                             'S_cor': data['S_cor'].values,
                             'S_inc': data['S_inc'].values,
                             'k_priors': [k_priors['mu_mu'],
                                               k_priors['sd_mu'],
                                               k_priors['mu_sd'],
                                               k_priors['sd_sd']],
                             'A_priors': [A_priors['mu_mu'],
                                            A_priors['sd_mu'],
                                            A_priors['mu_sd'],
                                            A_priors['sd_sd']],
                            'tau_priors': [tau_priors['mu_mu'],
                                             tau_priors['sd_mu'],
                                             tau_priors['mu_sd'],
                                             tau_priors['sd_sd']],
                             'v0_priors': [v0_priors['mu_mu'],
                                                 v0_priors['sd_mu'],
                                                 v0_priors['mu_sd'],
                                                 v0_priors['sd_sd']],
                            'ws_priors': [ws_priors['mu_mu'],
                                                ws_priors['sd_mu'],
                                                ws_priors['mu_sd'],
                                                ws_priors['sd_sd']],
                            'wd_priors': [wd_priors['mu_mu'],
                                                wd_priors['sd_mu'],
                                                wd_priors['mu_sd'],
                                                wd_priors['sd_sd']]}
            # adjust priors for more complex models

        else:
            # set default priors for the non-hierarchical model:
            if k_priors is None:
                k_priors = {'mu':1, 'sd':1}
            if A_priors is None:
                A_priors = {'mu':0.3, 'sd':1}
            if tau_priors is None:
                tau_priors = {'mu':0, 'sd':1}
            if v0_priors is None:
                v0_priors = {'mu':9, 'sd':2}
            if ws_priors is None:
                ws_priors = {'mu':0, 'sd':2}
            if wd_priors is None:
                wd_priors = {'mu':3, 'sd':3}

            data_dict = {'N': N,
                              'rt': data['rt'].values,
                              'accuracy': data['accuracy_rescale'].values.astype(int),
                              'S_cor': data['S_cor'].values,
                              'S_inc': data['S_inc'].values,
                              'k_priors': [k_priors['mu'], k_priors['sd']],
                              'A_priors': [A_priors['mu'], A_priors['sd']],
                              'tau_priors': [tau_priors['mu'], tau_priors['sd']],
                              'v0_priors': [v0_priors['mu'], v0_priors['sd']],
                              'ws_priors': [ws_priors['mu'], ws_priors['sd']],
                              'wd_priors': [wd_priors['mu'], wd_priors['sd']]
                              }

        # start sampling...
        fitted_model = self.compiled_model.sampling(data_dict, **kwargs)

        fitted_model = raceFittedModel_2A(fitted_model,
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

class RLALBAModel_2A(Model):
    """RLALBAModel_2A allows to specify a combination of reinforcement learning
    and advantage linear ballistic accumulator models.

    When initializing the model, you should specify whether the model is hierarchical or not.
    Additionally, you can specify the mechanisms that you wish to include or exclude.

    The underlying stan model will be compiled if no previously compiled model is found.
    After initializing the model, it can be fitted to a particular dataset using pystan.

    """
    def __init__(self, hierarchical_levels,
                 separate_learning_rates=False):
        """Initialize a RLALBAModel_2A object.

        Note
        ----
        This model is restricted to two options per trial (coded as correct and incorrect).
        However, more than two options can be presented in the same learning session.

        Parameters
        ----------

        hierarchical_levels : int
             Set to 1 for individual data and to 2 for grouped data.

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
        super().__init__(hierarchical_levels, "RLALBA_2A")

        self.separate_learning_rates = separate_learning_rates

        self.n_parameters_individual = 7 # k, A, tau, v0, ws, wd, learning rate
        self.n_parameters_trial = 0

        if self.separate_learning_rates:
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
            k_priors=None,
            A_priors=None,
            tau_priors=None,
            alpha_priors=None,
            v0_priors=None,
            ws_priors=None,
            wd_priors=None,
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

        k_priors : dict, optional
            Priors for the k parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        A_priors : dict, optional
            Priors for the A parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        tau_priors : dict, optional
            Priors for the tau parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        alpha_priors : dict, optional
            Priors for the learning rate parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        v0_priors : dict, optional
            Priors for the v0 parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        ws_priors : dict, optional
            Priors for the ws parameter.
            In case it is not a hierarchical model: Mean and standard deviation of the prior distr.
            In case it is a hierarchical model: Means and standard deviations of the hyper priors.

        wd_priors : dict, optional
            Priors for the wd parameter.
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

        data['accuracy_rescale'] = 2
        data.loc[data.accuracy == 1, 'accuracy_rescale'] = 1
        if self.hierarchical_levels == 2:
            # set default priors for the hierarchical model:
            if k_priors is None:
                k_priors = {'mu_mu':1, 'sd_mu':1, 'mu_sd':0, 'sd_sd':1}
            if A_priors is None:
                A_priors = {'mu_mu':0, 'sd_mu':0.1, 'mu_sd':0, 'sd_sd':1}
            if tau_priors is None:
                tau_priors = {'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':1}
            if v0_priors is None:
                v0_priors = {'mu_mu':9, 'sd_mu':1, 'mu_sd':2, 'sd_sd':1}
            if ws_priors is None:
                ws_priors = {'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':1}
            if wd_priors is None:
                wd_priors = {'mu_mu':3, 'sd_mu':1, 'mu_sd':3, 'sd_sd':1}
            if alpha_priors is None:
                alpha_priors = {'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':.1}
            if alpha_pos_priors is None:
                alpha_pos_priors = {'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':.1}
            if alpha_neg_priors is None:
                alpha_neg_priors = {'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':.1}

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
                             'accuracy': data['accuracy_rescale'].values.astype(int),
                             'initial_value': initial_value_learning,
                             'k_priors': [k_priors['mu_mu'],
                                               k_priors['sd_mu'],
                                               k_priors['mu_sd'],
                                               k_priors['sd_sd']],
                             'A_priors': [A_priors['mu_mu'],
                                               A_priors['sd_mu'],
                                               A_priors['mu_sd'],
                                               A_priors['sd_sd']],
                             'tau_priors': [tau_priors['mu_mu'],
                                                  tau_priors['sd_mu'],
                                                  tau_priors['mu_sd'],
                                                  tau_priors['sd_sd']],
                             'v0_priors': [v0_priors['mu_mu'],
                                               v0_priors['sd_mu'],
                                               v0_priors['mu_sd'],
                                               v0_priors['sd_sd']],
                             'ws_priors': [ws_priors['mu_mu'],
                                               ws_priors['sd_mu'],
                                               ws_priors['mu_sd'],
                                               ws_priors['sd_sd']],
                             'wd_priors': [wd_priors['mu_mu'],
                                               wd_priors['sd_mu'],
                                               wd_priors['mu_sd'],
                                               wd_priors['sd_sd']],
                             'alpha_priors': [alpha_priors['mu_mu'],
                                              alpha_priors['sd_mu'],
                                              alpha_priors['mu_sd'],
                                              alpha_priors['sd_sd']]
                            }

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
            # set default priors for the hierarchical model:
            if k_priors is None:
                k_priors = {'mu':1, 'sd':1}
            if A_priors is None:
                A_priors = {'mu':0.3, 'sd':1}
            if tau_priors is None:
                tau_priors = {'mu':0, 'sd':1}
            if alpha_priors is None:
                alpha_priors = {'mu':0, 'sd':1}
            if v0_priors is None:
                v0_priors = {'mu':9, 'sd':2}
            if ws_priors is None:
                ws_priors = {'mu':0, 'sd':2}
            if wd_priors is None:
                wd_priors = {'mu':3, 'sd':3}
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
                         'rt': data['rt'].values,
                         'accuracy': data['accuracy_rescale'].values.astype(int),
                         'initial_value': initial_value_learning,
                         'k_priors': [k_priors['mu'], k_priors['sd']],
                         'A_priors': [A_priors['mu'], A_priors['sd']],
                         'tau_priors': [tau_priors['mu'], tau_priors['sd']],
                         'v0_priors': [v0_priors['mu'], v0_priors['sd']],
                         'ws_priors': [ws_priors['mu'], ws_priors['sd']],
                         'wd_priors': [wd_priors['mu'], wd_priors['sd']],
                         'alpha_priors': [alpha_priors['mu'], alpha_priors['sd']]
                        }

            if self.separate_learning_rates:
                data_dict.update({'alpha_pos_priors': [alpha_pos_priors['mu'],
                                                       alpha_pos_priors['sd']],
                                  'alpha_neg_priors': [alpha_neg_priors['mu'],
                                                       alpha_neg_priors['sd']]})
                del data_dict['alpha_priors']

        fitted_model = self.compiled_model.sampling(data_dict, **kwargs)

        fitted_model = raceFittedModel_2A(fitted_model,
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
