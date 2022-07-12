import random

import numpy as np
import pandas as pd


def generate_task_design_fontanesi(n_trials_block,
                                   n_blocks,
                                   n_participants,
                                   trial_types,
                                   mean_options,
                                   sd_options):
    """Generates the RL_2A stimuli as in the 2019 Fontanesi et al.'s paper.

    Note
    ----
    In the original paper we corrected for repetition
    and order presentation of values too.
    This is not implemented here.

    Parameters
    ----------

    n_trials_block : int
        Number of trials per learning session.

    n_blocks : int
        Number of learning session per participant.

    n_participants : int
        Number of participants.

    trial_types : list of strings
        List containing possible pairs of options.
        E.g., in the original experiment: ['1-2', '1-3', '2-4', '3-4'].
        It is important that they are separated by a '-',
        and that they are numbered from 1 to n_options (4 in the example).
        Also, the "incorrect" option of the couple should go first in each pair.

    mean_options : list or array of floats
        Mean reward for each option.
        The length should correspond to n_options.

    sd_options : list or array of floats
        SD reward for each option.
        The length should correspond to n_options.

    Returns
    -------

    task_design : DataFrame
        `pandas.DataFrame`, with n_trials_block*n_blocks rows.
        Columns contain:
        "f_cor", "f_inc", "trial_type", "cor_option", "inc_option",
        "trial_block", "block_label", "participant".

    """
    if n_trials_block % len(trial_types) > 0:
        raise ValueError("The number of trials in a block should be a multiple of the number of trial types.")

    def generate_trial_type_sequence(n_trials_block, trial_types):
        """Check that the same trial type does not repeat more than 3 times in a row.
        """
        n_trial_types = len(trial_types)
        sequence = list(trial_types) * int(n_trials_block / n_trial_types)
        random.shuffle(sequence)

        count = 3
        while count < len(sequence):
            if sequence[count] == sequence[count - 1] == sequence[count - 2] == sequence[count - 3]:
                random.shuffle(sequence)
                count = 2
            count += 1

        return np.array(sequence)

    n_trials = n_trials_block * n_blocks

    task_design = pd.DataFrame({'participant': np.repeat(np.arange(1, n_participants + 1), n_trials),
                                'block_label': np.tile(np.repeat(np.arange(1, n_blocks + 1), n_trials_block),
                                                       n_participants),
                                'trial_block': np.tile(np.arange(1, n_trials_block + 1), n_blocks * n_participants),
                                'trial': np.tile(np.arange(1, n_trials + 1), n_participants)})

    task_design['trial_type'] = np.concatenate(
        [generate_trial_type_sequence(n_trials_block, trial_types) for i in range(n_blocks * n_participants)])
    task_design[['inc_option', 'cor_option']] = task_design.trial_type.str.split("-", expand=True).astype(int)

    options = pd.unique(task_design[['inc_option', 'cor_option']].values.ravel('K'))
    options = np.sort(options.astype(int))  # sorted option numbers
    n_options = len(options)
    print(f"The task will be created with the following {n_options} options: {options}.")
    print(f"With mean (respectively): {mean_options} and SD: {sd_options}.")

    def reward_options(row):
        """Sample a reward from normal distribution for the cor/inc options in each row.
        """
        index_inc = int(row.inc_option - 1)
        f_inc = np.round(np.random.normal(mean_options[index_inc], sd_options[index_inc]))
        index_cor = int(row.cor_option - 1)
        f_cor = np.round(np.random.normal(mean_options[index_cor], sd_options[index_cor]))

        return pd.Series({'f_inc': int(f_inc), 'f_cor': int(f_cor)})

    task_design[['f_inc', 'f_cor']] = task_design.apply(reward_options, axis=1)

    return task_design


def _simulate_delta_rule_2A(task_design,
                            alpha,
                            initial_value_learning,
                            alpha_pos=None,
                            alpha_neg=None):
    """Q learning (delta learning rule) for two alternatives
    (one correct, one incorrect).

    Parameters
    ----------

    task_design : DataFrame
        `pandas.DataFrame`, with n_trials_block*n_blocks rows.
        Columns contain:
        "f_cor", "f_inc", "trial_type", "cor_option", "inc_option",
        "trial_block", "block_label", "participant".

    alpha : float
        The generating learning rate.
        It should be a value between 0 (no updating) and 1 (full updating).

    initial_value_learning : float
        The initial value for Q learning.

    Optional parameters
    -------------------

    alpha_pos : float, default None
        If a value for both alpha_pos and alpha_neg is provided,
        separate learning rates are estimated
        for positive and negative prediction errors.

    alpha_neg : float, default None
        If a value for both alpha_pos and alpha_neg is provided,
        separate learning rates are estimated
        for positive and negative prediction errors.

    Returns
    -------

    Q_series : Series
        The series of learned Q values (separately for correct and incorrect options).

    """
    if (alpha_pos is not None) & (alpha_neg is not None):
        separate_learning_rates = True
    else:
        separate_learning_rates = False

    for label in ["f_cor", "f_inc", "cor_option", "inc_option", "trial_block", "block_label", "participant"]:
        if not label in task_design.columns:
            raise ValueError(f"Column {label} should be included in the task_design.")
    if separate_learning_rates:
        if type(alpha_pos) is not np.ndarray:
            alpha_pos = np.array([alpha_pos])
        if type(alpha_neg) is not np.ndarray:
            alpha_neg = np.array([alpha_neg])
    else:
        if type(alpha) is not np.ndarray:
            alpha = np.array([alpha])

    participants = pd.unique(task_design["participant"])
    n_participants = len(participants)

    if separate_learning_rates:
        if (n_participants != len(alpha_pos)) | (n_participants != len(alpha_neg)):
            raise ValueError("The learning rates should be as many as the number of participants in the task_design.")
    else:
        if n_participants != len(alpha):
            raise ValueError("The learning rates should be as many as the number of participants in the task_design.")

    n_trials = task_design.shape[0]
    options = pd.unique(task_design[['inc_option', 'cor_option']].values.ravel('K'))
    n_options = len(options)  # n Q values to be learned

    Q_cor = np.array([])
    Q_inc = np.array([])
    Q_min = np.array([])
    Q_max_t = np.array([])
    Q_mean_t = np.array([])
    for n in range(n_trials):
        index_cor = int(task_design.cor_option.values[n] - 1)
        index_inc = int(task_design.inc_option.values[n] - 1)
        index_participant = np.where(participants == task_design.participant.values[n])[0][0]

        if task_design.trial_block.values[n] == 1:
            Q = np.ones(n_options) * initial_value_learning
        else:
            if separate_learning_rates:
                pe_cor = task_design.f_cor.values[n] - Q[index_cor]
                pe_inc = task_design.f_inc.values[n] - Q[index_inc]
                if pe_cor > 0:
                    Q[index_cor] += alpha_pos[index_participant] * (task_design.f_cor.values[n] - Q[index_cor])
                else:
                    Q[index_cor] += alpha_neg[index_participant] * (task_design.f_cor.values[n] - Q[index_cor])
                if pe_inc > 0:
                    Q[index_inc] += alpha_pos[index_participant] * (task_design.f_inc.values[n] - Q[index_inc])
                else:
                    Q[index_inc] += alpha_neg[index_participant] * (task_design.f_inc.values[n] - Q[index_inc])
            else:
                Q[index_cor] += alpha[index_participant] * (task_design.f_cor.values[n] - Q[index_cor])
                Q[index_inc] += alpha[index_participant] * (task_design.f_inc.values[n] - Q[index_inc])

        Q_cor = np.append(Q_cor, Q[index_cor])
        Q_inc = np.append(Q_inc, Q[index_inc])
        Q_min = np.append(Q_min, np.min(Q))
        Q_max_t = np.append(Q_max_t, np.max([Q[index_cor], Q[index_inc]]))
        Q_mean_t = np.append(Q_mean_t, (Q[index_cor] + Q[index_inc])/2)

    return pd.DataFrame({'Q_cor': Q_cor,
                         'Q_inc': Q_inc, 
                         'Q_min': Q_min,
                         'Q_max_t': Q_max_t,
                         'Q_mean_t': Q_mean_t})


def _soft_max_2A(row):
    nom = np.exp(row.Q_cor * row.sensitivity)
    denom = np.sum([nom + np.exp(row.Q_inc * row.sensitivity)])
    return nom / denom
