import numpy as np
import pandas as pd
from scipy import stats

# RL_2A + DDM
from rlssm.random.random_DDM import random_ddm
from rlssm.random.random_RL import _simulate_delta_rule_2A


def simulate_rlddm_2A(task_design,
                      gen_alpha,
                      gen_drift_scaling,
                      gen_threshold,
                      gen_ndt,
                      initial_value_learning=0,
                      **kwargs):
    """Simulates behavior (rt and accuracy) according to a RLDDM model,

    where the learning component is the Q learning
    (delta learning rule) and the choice rule is the DDM.

    Simulates data for one participant.

    In this parametrization, it is assumed that 0 is the lower threshold,
    and the diffusion process starts halfway through the threshold value.

    Note
    ----
    The number of options can be actaully higher than 2,
    but only 2 options (one correct, one incorrect) are presented
    in each trial.
    It is important that "trial_block" is set to 1 at the beginning
    of each learning session (when the Q values at resetted)
    and that the "block_label" is set to 1 at the beginning of the
    experiment for each participants.
    There is no special requirement for the participant number.

    Parameters
    ----------

    task_design : DataFrame
        `pandas.DataFrame`, with n_trials_block*n_blocks rows.
        Columns contain:
        "f_cor", "f_inc", "trial_type", "cor_option", "inc_option",
        "trial_block", "block_label", "participant".

    gen_alpha : float or list of floats
        The generating learning rate.
        It should be a value between 0 (no updating) and 1 (full updating).
        If a list of 2 values is provided then 2 separate learning rates
        for positive and negative prediction error are used.

    gen_drift_scaling : float
        Drift-rate scaling of the RLDDM.

    gen_threshold : float
        Threshold of the diffusion decision model.
        Should be positive.

    gen_ndt : float
        Non decision time of the diffusion decision model, in seconds.
        Should be positive.

    initial_value_learning : float, default 0
        The initial value for Q learning.

    Other Parameters
    ----------------

    **kwargs : dict
        Additional arguments to be passed further.

    Returns
    -------

    data : DataFrame

    Examples
    --------

        >>> data_non_hier = simulate_rlddm_2A(task_design=self.dm_2_non_hier_alpha, gen_alpha=[.1, .01], gen_drift_scaling=.1, gen_threshold=1, gen_ndt=.23, initial_value_learning=0)
        >>> print(data_non_hier.head())

                                             trial trial_type  ...        rt  accuracy
        participant block_label trial_block                    ...
        1           1           1                1        2-4  ...  1.489023       1.0
                                2                2        1-3  ...  0.990023       1.0
                                3                3        1-2  ...  1.771023       0.0
                                4                4        2-4  ...  1.366023       1.0
                                5                5        3-4  ...  0.865023       1.0
    """
    data = task_design.copy()

    if (type(gen_alpha) == float) | (type(gen_alpha) == int):
        data['alpha'] = gen_alpha
        data = pd.concat([data, _simulate_delta_rule_2A(task_design=task_design,
                                                        alpha=gen_alpha,
                                                        initial_value_learning=initial_value_learning)],
                         axis=1)

    elif type(gen_alpha) is list:
        if len(gen_alpha) == 2:
            data['alpha_pos'] = gen_alpha[0]
            data['alpha_neg'] = gen_alpha[1]
            data = pd.concat([data, _simulate_delta_rule_2A(task_design=task_design,
                                                            alpha=None,
                                                            initial_value_learning=initial_value_learning,
                                                            alpha_pos=gen_alpha[0],
                                                            alpha_neg=gen_alpha[1])],
                             axis=1)

        elif len(gen_alpha) == 3:
            pass  # implement here Stefano's learning rule
        else:
            raise ValueError("The gen_alpha list should be of either length 2 or 3.")
    else:
        raise TypeError("The gen_alpha should be either a list or a float/int.")

    data['drift_scaling'] = gen_drift_scaling
    data['threshold'] = gen_threshold
    data['ndt'] = gen_ndt
    data['drift'] = gen_drift_scaling * (data['Q_cor'] - data['Q_inc'])

    # simulate responses
    rt, acc = random_ddm(data['drift'], data['threshold'], data['ndt'], .5, **kwargs)
    data['rt'] = rt
    data['accuracy'] = acc

    data = data.set_index(['participant', 'block_label', 'trial_block'])
    return data


def simulate_hier_rlddm_2A(task_design,
                           gen_mu_alpha, gen_sd_alpha,
                           gen_mu_drift_scaling, gen_sd_drift_scaling,
                           gen_mu_threshold, gen_sd_threshold,
                           gen_mu_ndt, gen_sd_ndt,
                           initial_value_learning=0,
                           **kwargs):
    """Simulates behavior (rt and accuracy) according to a RLDDM model,

    where the learning component is the Q learning
    (delta learning rule) and the choice rule is the DDM.

    Simulates hierarchical data for a group of participants.

    In this parametrization, it is assumed that 0 is the lower threshold,
    and the diffusion process starts halfway through the threshold value.

    The individual parameters have the following distributions:

    - alpha ~ Phi(normal(gen_mu_alpha, gen_sd_alpha))

    - drift_scaling ~ log(1 + exp(normal(gen_mu_drift, gen_sd_drift)))

    - threshold ~ log(1 + exp(normal(gen_mu_threshold, gen_sd_threshold)))

    - ndt ~ log(1 + exp(normal(gen_mu_ndt, gen_sd_ndt)))

    When 2 learning rates are estimated:

    - alpha_pos ~ Phi(normal(gen_mu_alpha[0], gen_sd_alpha[1]))

    - alpha_neg ~ Phi(normal(gen_mu_alpha[1], gen_sd_alpha[1]))

    Note
    ----
    The number of options can be actaully higher than 2,
    but only 2 options (one correct, one incorrect) are presented
    in each trial.
    It is important that "trial_block" is set to 1 at the beginning
    of each learning session (when the Q values at resetted)
    and that the "block_label" is set to 1 at the beginning of the
    experiment for each participants.
    There is no special requirement for the participant number.

    Parameters
    ----------

    task_design : DataFrame
        `pandas.DataFrame`, with n_trials_block*n_blocks rows.

    gen_mu_alpha : float or list of floats
        The generating group mean of the learning rate.
        If a list of 2 values is provided then 2 separate learning rates
        for positive and negative prediction error are used.

    gen_sd_alpha : float or list of floats
        The generating group SD of the learning rate.
        If a list of 2 values is provided then 2 separate learning rates
        for positive and negative prediction error are used.

    gen_mu_drift_scaling : float
        Group-mean of the drift-rate
        scaling of the RLDDM.

    gen_sd_drift_scaling: float
        Group-standard deviation of the drift-rate
        scaling of the RLDDM.

    gen_mu_threshold : float
        Group-mean of the threshold of the RLDDM.

    gen_sd_threshold: float
        Group-standard deviation of the threshold
        of the RLDDM.

    gen_mu_ndt : float
        Group-mean of the non decision time of the RLDDM.

    gen_sd_ndt : float
        Group-standard deviation of the non decision time
        of the RLDDM.

    Optional Parameters
    -------------------

    initial_value_learning : float, default 0
        The initial value for Q learning.

    Other Parameters
    ----------------

    **kwargs : dict
        Additional arguments to be passed further.

    Returns
    -------

    data : DataFrame

    Examples
    --------

        >>> data_hier_2alpha = simulate_hier_rlddm_2A(task_design=dm_hier, gen_mu_alpha=[-.5, -1], gen_sd_alpha=[.1, .1], gen_mu_drift_scaling=.1, gen_sd_drift_scaling=.5, gen_mu_threshold=1, gen_sd_threshold=.1, gen_mu_ndt=.23, gen_sd_ndt=.05, initial_value_learning=20)
        >>> print(data_hier_2alpha.head())

                                             trial trial_type  ...        rt  accuracy
        participant block_label trial_block                    ...
        1           1           1                1        2-4  ...  1.081138       0.0
                                2                2        1-3  ...  1.150138       1.0
                                3                3        1-2  ...  0.979138       0.0
                                4                4        3-4  ...  0.996138       0.0
                                5                5        3-4  ...  1.117138       1.0
    """
    data = task_design.copy()
    participants = pd.unique(data["participant"])
    n_participants = len(participants)
    if n_participants < 2:
        raise ValueError("You only have one participant. Use simulate_rl_2A instead.")

    if type(gen_mu_alpha) != type(gen_sd_alpha):
        raise TypeError("gen_mu_alpha and gen_sd_alpha should be of the same type.")

    if (type(gen_mu_alpha) == float) | (type(gen_mu_alpha) == int):
        parameters = pd.DataFrame(
            {'alpha': stats.norm.cdf(np.random.normal(gen_mu_alpha, gen_sd_alpha, n_participants)),
             'drift_scaling': np.log(
                 1 + np.exp(np.random.normal(gen_mu_drift_scaling, gen_sd_drift_scaling, n_participants))),
             'threshold': np.log(1 + np.exp(np.random.normal(gen_mu_threshold, gen_sd_threshold, n_participants))),
             'ndt': np.log(1 + np.exp(np.random.normal(gen_mu_ndt, gen_sd_ndt, n_participants)))},
            index=participants)
        data = pd.concat([data.set_index('participant'), parameters], axis=1, ignore_index=False).reset_index().rename(
            columns={'index': 'participant'})
        data = pd.concat([data, _simulate_delta_rule_2A(task_design,
                                                        parameters.alpha.values,
                                                        initial_value_learning)],
                         axis=1)

    elif type(gen_mu_alpha) is list:
        if len(gen_mu_alpha) != len(gen_sd_alpha):
            raise ValueError("gen_mu_alpha and gen_sd_alpha should be of the same length.")
        if len(gen_mu_alpha) == 2:
            parameters = pd.DataFrame(
                {'alpha_pos': stats.norm.cdf(np.random.normal(gen_mu_alpha[0], gen_sd_alpha[0], n_participants)),
                 'alpha_neg': stats.norm.cdf(np.random.normal(gen_mu_alpha[1], gen_sd_alpha[1], n_participants)),
                 'drift_scaling': np.log(
                     1 + np.exp(np.random.normal(gen_mu_drift_scaling, gen_sd_drift_scaling, n_participants))),
                 'threshold': np.log(1 + np.exp(np.random.normal(gen_mu_threshold, gen_sd_threshold, n_participants))),
                 'ndt': np.log(1 + np.exp(np.random.normal(gen_mu_ndt, gen_sd_ndt, n_participants)))},
                index=participants)
            data = pd.concat([data.set_index('participant'), parameters], axis=1,
                             ignore_index=False).reset_index().rename(columns={'index': 'participant'})
            data = pd.concat([data, _simulate_delta_rule_2A(task_design=task_design,
                                                            alpha=None,
                                                            initial_value_learning=initial_value_learning,
                                                            alpha_pos=parameters.alpha_pos.values,
                                                            alpha_neg=parameters.alpha_neg.values)],
                             axis=1)

        elif len(gen_mu_alpha) == 3:
            pass  # implement here Stefano's learning rule
        else:
            raise ValueError("The gen_mu_alpha list should be of either length 2 or 3.")
    else:
        raise TypeError("The gen_alpha should be either a list or a float/int.")

    data['drift'] = data['drift_scaling'] * (data['Q_cor'] - data['Q_inc'])

    # simulate responses
    rt, acc = random_ddm(data['drift'], data['threshold'], data['ndt'], .5, **kwargs)
    data['rt'] = rt
    data['accuracy'] = acc

    data = data.set_index(['participant', 'block_label', 'trial_block'])
    return data
