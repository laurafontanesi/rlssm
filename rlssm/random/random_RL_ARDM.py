import numpy as np
import pandas as pd
from scipy import stats

from rlssm.random.random_RDM import random_rdm_2A
from rlssm.random.random_common import _simulate_delta_rule_2A


def simulate_rlardm_2A(task_design,
                       gen_alpha,
                       gen_threshold,
                       gen_ndt,
                       gen_v0,
                       gen_ws,
                       gen_wd,
                       initial_value_learning=0,
                       gen_drift_trial_sd=None,
                       **kwargs):
    """Simulates behavior (rt and accuracy) according to the Advantage Racing Diffusion Model.

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

    gen_threshold : float
        Threshold of the ardm model. Should be positive.

    gen_ndt : float
        Non decision time of the ardm model, in seconds. Should be positive.

    gen_v0 : float
        The Bias parameter; ensures each accumulator has a positive drift rate, and eventually reaches threshold.
        Must be positive.

    gen_ws : float
        Sum weight: must be positive.

    gen_wd : float
        Difference weight: must be positive.

    initial_value_learning : float, default 0
        The initial value for Q learning.

    gen_drift_trial_sd : float, default None
        Across trial variability in the drift-rate. Should be positive.

    Other Parameters
    ----------------

    **kwargs : dict
        Additional parameters to be passed to `random_rdm_2A`.

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, with n_trials rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters
        (both for each trial and across-trials when there is across-trial variability).

    Examples
    --------

        >>> data_hier = simulate_hier_rlardm(task_design=dm_hier,
                                              gen_mu_alpha=[-.5, -1], gen_sd_alpha=[.1, .1],
                                              gen_mu_threshold=2, gen_sd_threshold=.1,
                                              gen_mu_ndt=.23, gen_sd_ndt=.1,
                                              gen_mu_v0=3, gen_sd_v0=.1,
                                              gen_mu_ws=-4, gen_sd_ws=.01,
                                              gen_mu_wd=-2, gen_sd_wd=.01,
                                              initial_value_learning=0,
                                              gen_drift_trial_sd=None)
        >>> print(data_hier.head())

                                       block_label trial_type  ...        rt  accuracy
        participant trial trial_block                          ...
        1           1     1                      1        1-3  ...  1.183774       1.0
                    2     2                      1        1-3  ...  1.236774       1.0
                    3     3                      1        1-3  ...  1.104774       0.0
                    4     4                      1        1-2  ...  1.089774       0.0
                    5     5                      1        1-3  ...  1.262774       1.0
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
            raise ValueError("Not implemented yet.")  # implement here Stefano's learning rule
        else:
            raise ValueError("The gen_alpha list should be of either length 2 or 3.")
    else:
        raise TypeError("The gen_alpha should be either a list or a float/int.")

    n_trials = np.shape(data['f_cor'])[0]

    data['threshold'] = gen_threshold
    data['ndt'] = gen_ndt

    if gen_drift_trial_sd is None:
        data['cor_drift'] = gen_v0 + gen_wd * (data['f_cor'] - data['f_inc']) + gen_ws * (data['f_cor'] + data['f_inc'])
        data['inc_drift'] = gen_v0 + gen_wd * (data['f_inc'] - data['f_cor']) + gen_ws * (data['f_cor'] + data['f_inc'])
    else:
        raise ValueError("Not implemented yet.")

    rt, acc = random_rdm_2A(cor_drift=data['cor_drift'], inc_drift=data['inc_drift'],
                            threshold=data['threshold'], ndt=data['ndt'])

    data['rt'] = rt
    data['accuracy'] = acc
    data['trial'] = np.arange(1, n_trials + 1)

    data = data.set_index(['participant', 'trial'])

    return data


def simulate_hier_rlardm(task_design,
                         gen_mu_alpha, gen_sd_alpha,
                         gen_mu_threshold, gen_sd_threshold,
                         gen_mu_ndt, gen_sd_ndt,
                         gen_mu_v0, gen_sd_v0,
                         gen_mu_ws, gen_sd_ws,
                         gen_mu_wd, gen_sd_wd,
                         initial_value_learning=0,
                         gen_drift_trial_sd=None,
                         **kwargs):
    """Simulate behavior (rt and accuracy) according to a hierarchical Advantage Racing Diffusion Model.

    Parameters
    ---------
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

    gen_mu_threshold : float
        Group-mean threshold of the Advantage Racing Diffusion Model.

    gen_sd_threshold : float
        Group-standard deviation of the threshold of the Advantage Racing Diffusion Model.

    gen_mu_ndt : float
        Group-mean non-decision time of the Advantage Racing Diffusion Model.

    gen_sd_ndt : float
        Group-standard deviation of the non-decision time of the Advantage Racing Diffusion Model.

    gen_mu_v0 : float or list of floats
        The mean of the Bias parameter; ensures each accumulator has a positive drift rate.
        It eventually reaches sp_trial_var.

    gen_sd_v0 : float or list of floats
        The SD of the Bias parameter; ensures each accumulator has a positive drift rate.
        It eventually reaches sp_trial_var.

    gen_mu_ws : float or list of floats
        The mean of the Sum Weight parameter.

    gen_sd_ws : float or list of floats
        The SD of the Sum Weight parameter.

    gen_mu_wd : float or list of floats
        The mean of the Difference Weight parameter.

    gen_sd_wd : float or list of floats
        The SD of the Difference Weight parameter.

    initial_value_learning : float, default 0
        The initial value for Q learning.

    gen_drift_trial_sd : float, default None
        Across trial variability in the drift-rate.
        Should be positive.

    Other parameters
    ----------------

    **kwargs : dict
        Additional parameters to be passed to `random_rdm_2A`.

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, with n_trials*n_participants rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters (at the participant level).

        >>> data1 = simulate_ardm_2A(gen_S_cor=np.random.normal(.4, 0.01, 100),
                                      gen_S_inc=np.random.normal(.3, 0.01, 100),
                                      gen_threshold=2, gen_ndt=.2, gen_v0=1,
                                      gen_ws=.7, gen_wd=1, gen_drift_trial_sd=.1)
        >>> print(data1.head())

                           threshold       ndt  ...        rt  accuracy
        participant trial                       ...
        1           1       1.304838  0.778967  ...  1.166967       0.0
                    2       1.304838  0.778967  ...  1.398967       1.0
                    3       1.304838  0.778967  ...  0.934967       1.0
                    4       1.304838  0.778967  ...  1.801967       0.0
                    5       1.304838  0.778967  ...  1.167967       0.0
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
             'threshold': np.log(1 + np.exp(np.random.normal(gen_mu_threshold, gen_sd_threshold, n_participants))),
             'ndt': np.log(1 + np.exp(np.random.normal(gen_mu_ndt, gen_sd_ndt, n_participants))),
             'v0': np.log(1 + np.exp(np.random.normal(gen_mu_v0, gen_sd_v0, n_participants))),
             'wd': np.log(1 + np.exp(np.random.normal(gen_mu_wd, gen_sd_wd, n_participants))),
             'ws': np.log(1 + np.exp(np.random.normal(gen_mu_ws, gen_sd_ws, n_participants)))},
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
                 'threshold': np.log(1 + np.exp(np.random.normal(gen_mu_threshold, gen_sd_threshold, n_participants))),
                 'ndt': np.log(1 + np.exp(np.random.normal(gen_mu_ndt, gen_sd_ndt, n_participants))),
                 'v0': np.log(1 + np.exp(np.random.normal(gen_mu_v0, gen_sd_v0, n_participants))),
                 'wd': np.log(1 + np.exp(np.random.normal(gen_mu_wd, gen_sd_wd, n_participants))),
                 'ws': np.log(1 + np.exp(np.random.normal(gen_mu_ws, gen_sd_ws, n_participants)))},
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

    if gen_drift_trial_sd is None:
        data['cor_drift'] = data['v0'] + data['wd'] * (data['Q_cor'] - data['Q_inc']) + data['ws'] * (
                data['Q_cor'] + data['Q_inc'])
        data['inc_drift'] = data['v0'] + data['wd'] * (data['Q_inc'] - data['Q_cor']) + data['ws'] * (
                data['Q_cor'] + data['Q_inc'])
    else:
        raise ValueError("Not implemented yet.")

    rt, acc = random_rdm_2A(cor_drift=data['cor_drift'], inc_drift=data['inc_drift'], threshold=data['threshold'],
                            ndt=data['ndt'])

    data['rt'] = rt
    data['accuracy'] = acc
    # data['trial'] = np.tile(np.arange(1, n_trials + 1), n_participants)

    data = data.set_index(['participant', 'trial', 'trial_block'])

    return data
