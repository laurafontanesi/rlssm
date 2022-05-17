import numpy as np
import pandas as pd
from scipy import stats

from rlssm.random.random_LBA import random_lba_2A
from rlssm.random.random_common import _simulate_delta_rule_2A


def simulate_rlalba_2A(task_design,
                       gen_alpha,
                       gen_threshold,  # A
                       gen_ndt,  # tau
                       gen_rel_sp,  # k
                       gen_v0,
                       gen_ws,
                       gen_wd,
                       initial_value_learning=0,
                       gen_drift_trial_sd=None,
                       participant_label=1,
                       **kwargs):
    """Simulates behavior (rt and accuracy) according to the RL-ALBA model.

    Note
    ----
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
        Threshold of the rlalba model. Should be positive.

    gen_ndt : float
        Non decision time of the rlalba model, in seconds. Should be positive.

    gen_rel_sp : float
        Relative starting point of the rlalba model. Should be higher than 0 and smaller than 1.

    gen_v0 : float
        The Bias parameter; ensures each accumulator has a positive drift rate, and eventually reaches threshold.
        Must be positive.

    gen_ws : float
        Sum weight: must be positive.

    gen_wd : float
        Difference weight: must be positive.

    Optional Parameters
    -------------------
    initial_value_learning : float
        The initial value for Q learning.

    gen_drift_trial_sd : float, default None
        Across trial variability in the drift-rate. Should be positive.

    participant_label : string or float, default 1
        What will appear in the participant column of the output data.

    kwargs : dict
        Additional parameters to be passed to `random_lba_2A`.
    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, with n_trials rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters
        (both for each trial and across-trials when there is across-trial variability).

    Example
    -------
        >>> self.dm_non_hier = generate_task_design_fontanesi(n_trials_block=80, n_blocks=3,
                                                              n_participants=1, trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                              mean_options=[34, 38, 50, 54], sd_options=[5, 5, 5, 5])

        >>> self.data1 = simulate_rlalba_2A(task_design=self.dm_non_hier,
                                        gen_alpha=0.1, gen_threshold=2,  # A
                                        gen_ndt=.2,  # tau
                                        gen_rel_sp=.2,  # k
                                        gen_v0=1, gen_ws=7, gen_wd=1,
                                        gen_drift_trial_sd=None, participant_label=1)
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

    f_cor = data['f_cor']
    f_inc = data['f_inc']

    n_trials = np.shape(f_cor)[0]

    data['threshold'] = gen_threshold
    data['ndt'] = gen_ndt
    data['rel_sp'] = gen_rel_sp

    gen_cor_drift = gen_v0 + gen_wd * (f_cor - f_inc) + gen_ws * (f_cor + f_inc)
    gen_inc_drift = gen_v0 + gen_wd * (f_inc - f_cor) + gen_ws * (f_cor + f_inc)

    if gen_drift_trial_sd is None:
        data['cor_drift'] = gen_cor_drift
        data['inc_drift'] = gen_inc_drift
    else:
        data['cor_drift'] = np.random.normal(gen_cor_drift, gen_drift_trial_sd)
        data['inc_drift'] = np.random.normal(gen_inc_drift, gen_drift_trial_sd)

    rt, acc = random_lba_2A(cor_drift=data['cor_drift'], inc_drift=data['inc_drift'],
                            threshold=data['threshold'], ndt=data['ndt'], rel_sp=data['rel_sp'])

    data['rt'] = rt
    data['accuracy'] = acc
    data['trial'] = np.arange(1, n_trials + 1)

    data = data.set_index(['participant', 'trial', 'trial_block'])

    return data


def simulate_hier_rlalba(task_design, n_trials, gen_mu_alpha, gen_sd_alpha,
                         gen_v0, gen_ws, gen_wd,
                         gen_mu_drift_cor, gen_sd_drift_cor,
                         gen_mu_drift_inc, gen_sd_drift_inc,
                         gen_mu_threshold, gen_sd_threshold,
                         gen_mu_ndt, gen_sd_ndt,
                         gen_mu_rel_sp=.5, gen_sd_rel_sp=None,
                         initial_value_learning=0,
                         gen_drift_trial_sd=None,
                         **kwargs):
    """Simulate behavior (rt and accuracy) according to a hierarchical RL-ALBA model.

    Parameters
    ----------
    task_design : DataFrame
        `pandas.DataFrame`, with n_trials_block*n_blocks rows.
        Columns contain:
        "f_cor", "f_inc", "trial_type", "cor_option", "inc_option",
        "trial_block", "block_label", "participant".

    n_trials : int
        Number of trials to simulate.

    gen_mu_alpha : float or list of floats
        The generating group mean of the learning rate.
        If a list of 2 values is provided then 2 separate learning rates
        for positive and negative prediction error are used.

    gen_sd_alpha : float or list of floats
        The generating group SD of the learning rate.
        If a list of 2 values is provided then 2 separate learning rates
        for positive and negative prediction error are used.

    gen_v0 : float
        The Bias parameter; ensures each accumulator has a positive drift rate, and eventually reaches threshold.
        Must be positive.

    gen_ws : float
        Sum weight: must be positive.

    gen_wd : float
        Difference weight: must be positive.

    gen_mu_drift_cor : float
        Mean of the drift rate for correct trials.

    gen_sd_drift_cor : float
        Standard deviation of the drift rate for correct trials.

    gen_mu_drift_inc : float
        Mean of the drift rate for incorrect trials.

    gen_sd_drift_inc : float
        Standard deviation of the drift rate for incorrect trials.

    gen_mu_threshold : float
        Group-mean threshold of the advantage linear ballistic accumulator.

    gen_sd_threshold : float
        Group-standard deviation of the threshold of the advantage linear ballistic accumulator.

    gen_mu_ndt : float
        Group-mean non-decision time of the advantage linear ballistic accumulator.

    gen_sd_ndt : float
        Group-standard deviation of the non-decision time of the advantage linear ballistic accumulator.

    Optional parameters
    -------------------
    gen_mu_rel_sp : float, default .5
        Relative starting point of the linear ballistic accumulator.
        When `gen_sd_rel_sp` is not specified, `gen_mu_rel_sp` is
        fixed across participants to .5.
        When `gen_sd_rel_sp` is specified, `gen_mu_rel_sp` is the
        group-mean of the starting point.

    gen_sd_rel_sp : float, default None
        Group-standard deviation of the relative starting point of the linear ballistic accumulator.

    initial_value_learning : float
        The initial value for Q learning.

    gen_drift_trial_sd : float, optional
        Across trial variability in the drift-rate.
        Should be positive.

    kwargs : dict
        Additional parameters to be passed to `random_lba_2A`.

    Returns
    -------
    data : DataFrame
        `pandas.DataFrame`, with n_trials*n_participants rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters (at the participant level).

    Example
    -------
    >>>         self.dm_hier = generate_task_design_fontanesi(n_trials_block=80,
                                                              n_blocks=3,
                                                              n_participants=30,
                                                              trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                              mean_options=[34, 38, 50, 54],
                                                              sd_options=[5, 5, 5, 5])

    >>>         self.data_hier = simulate_hier_rlalba(task_design=self.dm_hier,
                                              n_trials=100,
                                              gen_mu_alpha=[-.5, -1],
                                              gen_sd_alpha=[.1, .1],
                                              gen_v0=1, gen_ws=.7, gen_wd=1,
                                              gen_mu_drift_cor=.4, gen_sd_drift_cor=0.01,
                                              gen_mu_drift_inc=.3, gen_sd_drift_inc=0.01,
                                              gen_mu_threshold=1, gen_sd_threshold=.1,
                                              gen_mu_ndt=.23, gen_sd_ndt=.1,
                                              gen_mu_rel_sp=.5, gen_sd_rel_sp=None,
                                              initial_value_learning=0,
                                              gen_drift_trial_sd=None)
    """
    data = task_design.copy()
    participants = pd.unique(data["participant"])
    n_participants = len(participants)
    n_block_labels = len(pd.unique(data["block_label"]))
    n_blocks = len(pd.unique(data["trial_block"]))

    if n_participants < 2:
        raise ValueError("You only have one participant. Use simulate_rl_2A instead.")

    if type(gen_mu_alpha) != type(gen_sd_alpha):
        raise TypeError("gen_mu_alpha and gen_sd_alpha should be of the same type.")

    if (type(gen_mu_alpha) == float) | (type(gen_mu_alpha) == int):
        parameters = pd.DataFrame(
            {'alpha': stats.norm.cdf(np.random.normal(gen_mu_alpha, gen_sd_alpha, n_participants)),
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
            raise ValueError("gen_mu_alpha and gen_sd_alpha should be of the same lenght.")
        if len(gen_mu_alpha) == 2:
            parameters = pd.DataFrame(
                {'alpha_pos': stats.norm.cdf(np.random.normal(gen_mu_alpha[0], gen_sd_alpha[0], n_participants)),
                 'alpha_neg': stats.norm.cdf(np.random.normal(gen_mu_alpha[1], gen_sd_alpha[1], n_participants)),
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

    gen_S_cor = np.random.normal(gen_mu_drift_cor, gen_sd_drift_cor, n_participants)
    gen_S_inc = np.random.normal(gen_mu_drift_inc, gen_sd_drift_inc, n_participants)

    cor_drift_sbj = gen_v0 + gen_wd * (gen_S_cor - gen_S_inc) + gen_ws * (gen_S_cor + gen_S_inc)
    inc_drift_sbj = gen_v0 + gen_wd * (gen_S_inc - gen_S_cor) + gen_ws * (gen_S_cor + gen_S_inc)

    threshold_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_threshold, gen_sd_threshold, n_participants)))
    ndt_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_ndt, gen_sd_ndt, n_participants)))

    # data['participant'] = np.repeat(np.arange(n_participants) + 1, n_trials)
    data['threshold'] = np.repeat(threshold_sbj, n_blocks * n_block_labels)
    data['ndt'] = np.repeat(ndt_sbj, n_blocks * n_block_labels)

    if gen_drift_trial_sd is None:
        data['cor_drift'] = np.repeat(cor_drift_sbj, n_blocks * n_block_labels)
        data['inc_drift'] = np.repeat(inc_drift_sbj, n_blocks * n_block_labels)
    else:
        data['cor_drift'] = np.random.normal(np.repeat(cor_drift_sbj, n_trials), gen_drift_trial_sd)
        data['inc_drift'] = np.random.normal(np.repeat(inc_drift_sbj, n_trials), gen_drift_trial_sd)

    if gen_sd_rel_sp is None:
        data['rel_sp'] = np.repeat(.5, n_participants * n_blocks * n_block_labels)
    else:
        rel_sp_sbj = stats.norm.cdf(np.random.normal(gen_mu_rel_sp, gen_sd_rel_sp, n_participants))
        data['rel_sp'] = np.repeat(rel_sp_sbj, n_trials)

    rt, acc = random_lba_2A(cor_drift=data['cor_drift'], inc_drift=data['inc_drift'], threshold=data['threshold'],
                            ndt=data['ndt'], rel_sp=data['rel_sp'])

    data['rt'] = rt
    data['accuracy'] = acc
    # data['trial'] = np.tile(np.arange(1, n_trials + 1), n_blocks * n_block_labels)

    data = data.set_index(['participant', 'trial', 'trial_block'])

    return data
