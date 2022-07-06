import pandas as pd

from rlssm.random.random_RDM import random_rdm_2A
from rlssm.random.random_common import _simulate_delta_rule_2A


def simulate_rlrdm_2A(task_design,
                      gen_alpha,
                      gen_drift_scaling,
                      gen_threshold,
                      gen_ndt,
                      initial_value_learning=0,
                      **kwargs):
    """Simulates behavior (rt and accuracy) according to the RL-RDM model.

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

    gen_drift_scaling:
        Drift-rate scaling of the RL-RDM model.

    gen_threshold : float
        Threshold of the diffusion decision model.
        Should be positive.

    gen_ndt : float
        Non decision time of the diffusion decision model, in seconds.
        Should be positive.

    Optional Parameters
    -------------------

    initial_value_learning : float, default 0
        The initial value for Q learning.

    Other Parameters
    ----------------

    **kwargs : dict
        Additional parameters to be passed to `random_lba_2A`.

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, with n_trials rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters
        (both for each trial and across-trials when there is across-trial variability).

    Examples
    --------

        >>> data = simulate_rlrdm_2A(task_design=dm, gen_drift_scaling=.1,
                                      gen_threshold=1, gen_ndt=.23, initial_value_learning=0)
        >>> print(data.head())

                                             trial trial_type  ...     rt  accuracy
        participant block_label trial_block                    ...
        1           1           1                1        3-4  ...  0.886       1.0
                                2                2        1-3  ...  2.376       1.0
                                3                3        1-3  ...  0.473       0.0
                                4                4        1-2  ...  0.630       0.0
                                5                5        3-4  ...  0.420       1.0

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
    data['cor_drift'] = gen_drift_scaling * (data['Q_cor'])
    data['inc_drift'] = gen_drift_scaling * (data['Q_inc'])

    # simulate responses
    rt, acc = random_rdm_2A(data['cor_drift'],
                            data['inc_drift'],
                            data['threshold'],
                            data['ndt'], **kwargs)
    data['rt'] = rt
    data['accuracy'] = acc

    data = data.set_index(['participant', 'block_label', 'trial_block'])
    return data
