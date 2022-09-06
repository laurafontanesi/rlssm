import numpy as np
import pandas as pd
from scipy import stats

from rlssm.random.random_common import _simulate_delta_rule_2A, _soft_max_2A


def simulate_rl_2A(task_design,
                   gen_alpha,
                   gen_sensitivity,
                   initial_value_learning=0):
    """Simulates behavior (accuracy) according to a RL_2A model,

    where the learning component is the Q learning
    (delta learning rule) and the choice rule is the softmax.

    This function is to simulate data for, for example, parameter recovery.
    Simulates data for one participant.

    Note
    ----
    The number of options can be actually higher than 2,
    but only 2 options (one correct, one incorrect) are presented
    in each trial.
    It is important that "trial_block" is set to 1 at the beginning
    of each learning session (when the Q values at resetted)
    and that the "block_label" is set to 1 at the beginning of the
    experiment for each participant.
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

    gen_sensitivity : float
        The generating sensitivity parameter for the soft_max choice rule.
        It should be a value higher than 0
        (the higher, the more sensitivity to value differences).

    initial_value_learning : float, default 0
        The initial value for Q learning.

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, that is the task_design, plus:
        'Q_cor', 'Q_inc', 'alpha', 'sensitivity',
        'p_cor', and 'accuracy'.

    Examples
    --------
        >>> data = simulate_rl_2A(task_design=dm_non_hier, gen_alpha=.1, gen_sensitivity=.5,
                                    initial_value_learning=20)

        >>> print(data)
                                             trial trial_type  ...     p_cor  accuracy
        participant block_label trial_block                    ...
        1           1           1                1        1-2  ...  0.500000         1
                                2                2        3-4  ...  0.610639         1
                                3                3        1-3  ...  0.821274         1
                                4                4        1-3  ...  0.943880         1
                                5                5        1-2  ...  0.196945         0
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

    data['sensitivity'] = gen_sensitivity
    data['p_cor'] = data.apply(_soft_max_2A, axis=1)
    data['accuracy'] = stats.bernoulli.rvs(data['p_cor'].values)  # simulate choices

    data = data.set_index(['participant', 'block_label', 'trial_block'])
    return data


def simulate_hier_rl_2A(task_design,
                        gen_mu_alpha, gen_sd_alpha,
                        gen_mu_sensitivity, gen_sd_sensitivity,
                        initial_value_learning=0):
    """Simulates behavior (accuracy) according to a RL_2A model,
    where the learning component is the Q learning
    (delta learning rule) and the choice rule is the softmax.

    Simulates hierarchical data for a group of participants.
    The individual parameters have the following distributions:

    - alpha ~ Phi(normal(gen_mu_alpha, gen_sd_alpha))

    - sensitivity ~ log(1 + exp(normal(gen_mu_sensitivity, gen_sd_sensitivity)))

    When 2 learning rates are estimated:

    - alpha_pos ~ Phi(normal(gen_mu_alpha[0], gen_sd_alpha[1]))

    - alpha_neg ~ Phi(normal(gen_mu_alpha[1], gen_sd_alpha[1]))

    Note
    ----
    The number of options can be actually higher than 2,
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

    gen_mu_alpha : float or list of floats
        The generating group mean of the learning rate.
        If a list of 2 values is provided then 2 separate learning rates
        for positive and negative prediction error are used.

    gen_sd_alpha : float or list of floats
        The generating group SD of the learning rate.
        If a list of 2 values is provided then 2 separate learning rates
        for positive and negative prediction error are used.

    gen_mu_sensitivity : float
        The generating group mean of the sensitivity parameter
        for the soft_max choice rule.

    gen_sd_sensitivity : float
        The generating group SD of the sensitivity parameter
        for the soft_max choice rule.

    initial_value_learning : float, default 0
        The initial value for Q learning.

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, that is the task_design, plus:
        'Q_cor', 'Q_inc', 'alpha', 'sensitivity',
        'p_cor', and 'accuracy'.

    Examples
    --------
        >>> data_hier = simulate_hier_rl_2A(task_design=dm_hier, gen_mu_alpha=-.5, gen_sd_alpha=.1,
                                             gen_mu_sensitivity=.5, gen_sd_sensitivity=.1,
                                             initial_value_learning=20)

        >>> print(data_hier)
                                             trial trial_type  ...     p_cor  accuracy
        participant block_label trial_block                    ...
        1           1           1                1        1-2  ...  0.500000         0
                                2                2        3-4  ...  0.965112         1
                                3                3        1-2  ...  0.660162         0
                                4                4        2-4  ...  0.999997         1
                                5                5        1-2  ...  0.999863         1

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
             'sensitivity': np.log(
                 1 + np.exp(np.random.normal(gen_mu_sensitivity, gen_sd_sensitivity, n_participants)))},
            index=participants)
        data = pd.concat([data.set_index('participant'), parameters], axis=1, ignore_index=False).reset_index().rename(
            columns={'index': 'participant'})

        data = pd.concat([data, _simulate_delta_rule_2A(task_design=task_design,
                                                        alpha=parameters.alpha.values,
                                                        initial_value_learning=initial_value_learning)],
                         axis=1)

    elif type(gen_mu_alpha) is list:
        if len(gen_mu_alpha) != len(gen_sd_alpha):
            raise ValueError("gen_mu_alpha and gen_sd_alpha should be of the same lenght.")
        if len(gen_mu_alpha) == 2:
            parameters = pd.DataFrame(
                {'alpha_pos': stats.norm.cdf(np.random.normal(gen_mu_alpha[0], gen_sd_alpha[0], n_participants)),
                 'alpha_neg': stats.norm.cdf(np.random.normal(gen_mu_alpha[1], gen_sd_alpha[1], n_participants)),
                 'sensitivity': np.log(
                     1 + np.exp(np.random.normal(gen_mu_sensitivity, gen_sd_sensitivity, n_participants)))},
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

    data['p_cor'] = data.apply(_soft_max_2A, axis=1)
    data['accuracy'] = stats.bernoulli.rvs(data['p_cor'].values)  # simulate choices

    data = data.set_index(['participant', 'block_label', 'trial_block'])
    return data
