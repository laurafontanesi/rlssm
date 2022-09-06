from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
import pandas as pd

from rlssm.fit.fits import FittedModel, ModelResults
from rlssm.plot import plotting
from rlssm.utility.utils import list_individual_variables


class RLFittedModel_2A(FittedModel):

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
        self.family = family
        super().__init__(stan_model,
                         data,
                         hierarchical_levels,
                         model_label,
                         family,
                         n_parameters_individual,
                         n_parameters_trial,
                         print_diagnostics,
                         priors)

    def extract_results(self, include_rhat, include_waic, pointwise_waic, include_last_values):
        if include_rhat:
            rhat = self.get_rhat()
        else:
            rhat = None

        if include_waic:
            waic = self.calculate_waic(pointwise_waic)
        else:
            waic = None

        if include_last_values:
            last_values = self.get_last_values()
        else:
            last_values = None

        # main parameters
        if self.parameters_info['hierarchical_levels'] == 2:
            main_parameters = self.parameters_info['group_parameters_names_transf']

            main_parameters = np.append(main_parameters,
                                        [p + '_sbj' for p in self.parameters_info['individual_parameters_names']])

            # for p in self.parameters_info['individual_parameters_names']:
            #     main_parameters = np.append(main_parameters, list_individual_variables(p, self.data_info['L']))

        else:
            main_parameters = self.parameters_info['parameters_names_transf']

        samples = self.stan_model.draws_pd(vars=main_parameters)

        # trial parameters
        trial_samples = {'log_p_t': np.asarray(self.stan_model.draws_pd(vars=['log_p_t']))}

        res = RLModelResults_2A(self.model_label,
                                self.data_info,
                                self.parameters_info,
                                self.priors,
                                rhat,
                                waic,
                                last_values,
                                samples,
                                trial_samples,
                                self.family)
        return res


class RLModelResults_2A(ModelResults):
    """RLModelResults allows to perform various model checks
    on fitted RL_2A models.

    In particular, this can be used to to visualize the estimated
    posterior distributions and to calculate and visualize the
    estimated posterior predictives distributions.

    """

    def __init__(self,
                 model_label,
                 data_info,
                 parameters_info,
                 priors,
                 rhat,
                 waic,
                 last_values,
                 samples,
                 trial_samples,
                 family):
        self.family = family
        super().__init__(model_label,
                         data_info,
                         parameters_info,
                         priors,
                         rhat,
                         waic,
                         last_values,
                         samples,
                         trial_samples)

    def get_posterior_predictives(self, n_posterior_predictives=500):
        """Calculates posterior predictives of choices.

        Parameters
        ----------

        n_posterior_predictives : int
             Number of posterior samples to use for posterior predictives calculation.
             If n_posterior_predictives is bigger than the posterior samples,
             then calculation will continue with the total number of posterior samples.

        Returns
        -------

        pp_acc : numpy.ndarray
            Array of shape (n_samples, n_trials).

        """
        if n_posterior_predictives > self.parameters_info['n_posterior_samples']:
            warnings.warn("Cannot have more posterior predictive samples than posterior samples. " \
                          "Will continue with n_posterior_predictives=%s" % self.parameters_info['n_posterior_samples'],
                          UserWarning,
                          stacklevel=2)
            n_posterior_predictives = self.parameters_info['n_posterior_samples']

        log_p = self.trial_samples['log_p_t'][:n_posterior_predictives, :]

        p = np.exp(log_p)

        pp_acc = np.random.binomial(n=1, p=p)
        return pp_acc

    def get_posterior_predictives_df(self, n_posterior_predictives=500):
        """Calculates posterior predictives of choices.

        Parameters
        ----------

        n_posterior_predictives : int
             Number of posterior samples to use for posterior predictives calculation.
             If n_posterior_predictives is bigger than the posterior samples,
             then calculation will continue with the total number of posterior samples.

        Returns
        -------

        out : DataFrame
            Data frame of shape (n_samples, n_trials).

        """
        pp_acc = self.get_posterior_predictives(n_posterior_predictives)

        out = pd.DataFrame(pp_acc,
                           index=pd.Index(np.arange(1, len(pp_acc) + 1), name='sample'),
                           columns=pd.MultiIndex.from_product((['accuracy'],
                                                               np.arange(pp_acc.shape[1]) + 1),
                                                              names=['variable', 'trial']))
        return out

    def get_posterior_predictives_summary(self, n_posterior_predictives=500):
        """Calculates summary of posterior predictives of choices.

        The mean proportion of choices (in this case coded as accuracy) is calculated
        for each posterior sample across all trials.

        Parameters
        ----------

        n_posterior_predictives : int
             Number of posterior samples to use for posterior predictives calculation.
             If n_posterior_predictives is bigger than the posterior samples,
             then calculation will continue with the total number of posterior samples.

        Returns
        -------

        out : DataFrame
            Data frame, where every row corresponds to a posterior sample.
            The column contains the mean accuracy for each posterior sample
            over the whole dataset.

        """
        pp = self.get_posterior_predictives_df(n_posterior_predictives)
        out = pd.DataFrame({'mean_accuracy': pp['accuracy'].mean(axis=1)})

        return out

    def plot_mean_posterior_predictives(self,
                                        n_posterior_predictives,
                                        **kwargs):
        """Plots the mean posterior predictives of choices.

        The mean proportion of choices (in this case coded as accuracy) is calculated
        for each posterior sample across all trials,
        and then it's plotted as a distribution.
        The mean accuracy in the data is plotted as vertical line.
        This allows to compare the real mean with the BCI or HDI of the predictions.

        Parameters
        ----------

        n_posterior_predictives : int
             Number of posterior samples to use for posterior predictives calculation.
             If n_posterior_predictives is bigger than the posterior samples,
             then calculation will continue with the total number of posterior samples.

        Other Parameters
        ----------------

        **kwargs : dict
            Keyword arguments to be passed to the `plotting.plot_mean_prediction`.

        Returns
        -------

        ax : matplotlib.axes.Axes
            Returns the `matplotlib.axes.Axes` object with the plot
            for further tweaking.

        """

        pp_df = self.get_posterior_predictives_summary(n_posterior_predictives)

        ax = plotting.plot_mean_prediction(pp_df,
                                           self.data_info['data'],
                                           y_data='accuracy',
                                           y_predictions='mean_accuracy',
                                           **kwargs)
        return ax

    def get_grouped_posterior_predictives_summary(self,
                                                  grouping_vars,
                                                  n_posterior_predictives=500):
        """Calculates summary of posterior predictives of choices,
        separately for a list of grouping variables.

        The mean proportion of choices (in this case coded as accuracy) is calculated
        for each posterior sample across all trials
        in all conditions combination.

        For example, if grouping_vars=['reward', 'difficulty'],
        posterior predictives will be collapsed
        for all combinations of levels of the reward and difficulty variables.

        Parameters
        ----------

        grouping_vars :  list of strings
             They should be existing grouping variables in the data.

        n_posterior_predictives : int
             Number of posterior samples to use for posterior predictives calculation.
             If n_posterior_predictives is bigger than the posterior samples,
             then calculation will continue with the total number of posterior samples.

        Returns
        -------

        out : DataFrame
             Pandas DataFrame.
             The column contains the mean accuracy.
             The row indes is a pandas.MultIndex, with the grouping variables
             as higher level and number of samples as lower level.

        """

        data_copy = self.data_info['data'].copy()
        data_copy['trial'] = np.arange(1, self.data_info['N'] + 1)
        data_copy.set_index('trial', inplace=True)

        pp = self.get_posterior_predictives_df(n_posterior_predictives=n_posterior_predictives)

        tmp = pp.copy().T.reset_index().set_index('trial')
        tmp = pd.merge(tmp,
                       data_copy[grouping_vars],
                       left_index=True, right_index=True).reset_index()
        tmp_accuracy = tmp[tmp.variable == 'accuracy'].drop('variable',
                                                            axis=1)

        out = tmp_accuracy.groupby(grouping_vars).mean().drop('trial',
                                                              axis=1).stack().to_frame('mean_accuracy')
        out.index.rename(np.append(grouping_vars, 'sample'), inplace=True)

        return out

    def plot_mean_grouped_posterior_predictives(self,
                                                grouping_vars,
                                                n_posterior_predictives=500,
                                                **kwargs):
        """Plots the mean posterior predictives of choices,
        separately for either 1 or 2 grouping variables.

        The first grouping variable will be plotted on the x-axis.
        The second grouping variable, if provided, will be showed
        with a different color per variable level.

        Parameters
        ----------

        grouping_vars :  list of strings
             They should be existing grouping variables in the data.
             The list should be of lenght 1 or 2.

        n_posterior_predictives : int
             Number of posterior samples to use for posterior predictives calculation.
             If n_posterior_predictives is bigger than the posterior samples,
             then calculation will continue with the total number of posterior samples.

        Other Parameters
        ----------------

        **kwargs : dict
            Keyword arguments to be passed to the `plotting.plot_grouped_mean_prediction`.

        Returns
        -------

        ax : matplotlib.axes.Axes
            Returns the `matplotlib.axes.Axes` object with the plot
            for further tweaking.

        """

        if np.sum(len(grouping_vars) == np.array([1, 2])) < 1:
            raise ValueError("must be a list of either 1 or values")

        pp = self.get_grouped_posterior_predictives_summary(grouping_vars,
                                                            n_posterior_predictives)

        if len(grouping_vars) == 1:
            ax = plotting.plot_grouped_mean_prediction(x=grouping_vars[0],
                                                       y_data='accuracy',
                                                       y_predictions='mean_accuracy',
                                                       predictions=pp,
                                                       data=self.data_info['data'],
                                                       **kwargs)

        else:
            ax = plotting.plot_grouped_mean_prediction(x=grouping_vars[0],
                                                       y_data='accuracy',
                                                       y_predictions='mean_accuracy',
                                                       predictions=pp,
                                                       data=self.data_info['data'],
                                                       hue=grouping_vars[1],
                                                       **kwargs)
        return ax
