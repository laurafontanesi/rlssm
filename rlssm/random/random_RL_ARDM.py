import numpy as np
import pandas as pd
from scipy import stats

from rlssm.random.random_RDM import random_rdm_2A
from rlssm.random.random_common import _simulate_delta_rule_2A


def simulate_ardm_2A(task_design,
                     gen_alpha,
                     gen_threshold,  # A
                     gen_ndt,  # tau
                     gen_v0,
                     gen_ws,
                     gen_wd,
                     initial_value_learning=0,
                     gen_drift_trial_sd=None,
                     participant_label=1,
                     **kwargs):
    """Simulates behavior (rt and accuracy) according to the Advantage Racing Diffusion Model.

    Note
    ----
    Parameters
    ----------
    gen_S_cor : float
        Brightness of correct trials.

    gen_S_inc : float
        Brightness of incorrect trials.

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

    Optional Parameters
    -------------------

    gen_drift_trial_sd : float, default None
        Across trial variability in the drift-rate. Should be positive.

    participant_label : string or float, default 1
        What will appear in the participant column of the output data.

    kwargs : dict
        Additional parameters to be passed to `random_rdm_2A`.
    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, with n_trials rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters
        (both for each trial and across-trials when there is across-trial variability).

    Example
    -------

        >>> data1 = simulate_ardm_2A(gen_S_cor=np.random.normal(.4, 0.01, 100),
                                      gen_S_inc=np.random.normal(.3, 0.01, 100),
                                      gen_threshold=2, gen_ndt=.2, gen_v0=1,
                                      gen_ws=.7, gen_wd=1,
                                      gen_drift_trial_sd=.1)

        >>> print(data1.head())
                                          S_cor     S_inc  threshold  ...  inc_drift     rt  accuracy
            participant trial                                 ...
            1           1      0.418064  0.315546          2  ...   1.474992  1.497       0.0
                        2      0.402899  0.319556          2  ...   1.413985  0.940       0.0
                        3      0.410951  0.302430          2  ...   1.420837  1.829       0.0
                        4      0.407772  0.286868          2  ...   1.404168  1.289       0.0
                        5      0.398231  0.308216          2  ...   1.362828  1.847       1.0
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

    gen_cor_drift = gen_v0 + gen_wd * (f_cor - f_inc) + gen_ws * (f_cor + f_inc)
    gen_inc_drift = gen_v0 + gen_wd * (f_inc - f_cor) + gen_ws * (f_cor + f_inc)

    if gen_drift_trial_sd is None:
        data['cor_drift'] = gen_cor_drift
        data['inc_drift'] = gen_inc_drift
    else:
        data['cor_drift'] = np.random.normal(gen_cor_drift, gen_drift_trial_sd)
        data['inc_drift'] = np.random.normal(gen_inc_drift, gen_drift_trial_sd)

    rt, acc = random_rdm_2A(cor_drift=data['cor_drift'], inc_drift=data['inc_drift'],
                            threshold=data['threshold'], ndt=data['ndt'])

    data['rt'] = rt
    data['accuracy'] = acc
    data['trial'] = np.arange(1, n_trials + 1)

    data = data.set_index(['participant', 'trial'])

    return data


def simulate_hier_rlardm(task_design, n_trials, gen_mu_alpha, gen_sd_alpha,
                         gen_v0, gen_ws, gen_wd,
                         gen_mu_drift_cor, gen_sd_drift_cor,
                         gen_mu_drift_inc, gen_sd_drift_inc,
                         gen_mu_threshold, gen_sd_threshold,
                         gen_mu_ndt, gen_sd_ndt,
                         initial_value_learning=0,
                         gen_drift_trial_sd=None,
                         **kwargs):
    """Simulate behavior (rt and accuracy) according to a hierarchical Advantage Racing Diffusion Model.

    Parameters
    ---------
    n_trials : int
        Number of trials to simulate.

    n_participants : int
        Number of participants to simulate.

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
        Group-mean threshold of the advantage Racing Diffusion Model.

    gen_sd_threshold : float
        Group-standard deviation of the threshold of the advantage Racing Diffusion Model.

    gen_mu_ndt : float
        Group-mean non-decision time of the advantage Racing Diffusion Model.

    gen_sd_ndt : float
        Group-standard deviation of the non-decision time of the advantage Racing Diffusion Model.

    Optional parameters
    -------------------

    gen_drift_trial_sd : float, optional
        Across trial variability in the drift-rate.
        Should be positive.

    kwargs : dict
        Additional parameters to be passed to `random_rdm_2A`.

    Returns
    -------
    data : DataFrame
        `pandas.DataFrame`, with n_trials*n_participants rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters (at the participant level).

    Example
    -------

        >>> data_hier = simulate_hier_ardm(n_trials=100, n_participants=30, gen_v0=1,
                                            gen_ws=.7, gen_wd=1,
                                            gen_mu_drift_cor=.4, gen_sd_drift_cor=0.01,
                                            gen_mu_drift_inc=.3, gen_sd_drift_inc=0.01,
                                            gen_mu_threshold=1, gen_sd_threshold=.1,
                                            gen_mu_ndt=.23, gen_sd_ndt=.1,
                                            gen_drift_trial_sd=None)

        >>> print(data_hier.head())

                                   threshold    ndt  cor_drift  inc_drift     rt  accuracy
        participant trial
        1           1       1.256843  0.749   1.583609   1.367835  1.342       1.0
                    2       1.256843  0.749   1.583609   1.367835  1.008       0.0
                    3       1.256843  0.749   1.583609   1.367835  1.672       0.0
                    4       1.256843  0.749   1.583609   1.367835  1.201       0.0
                    5       1.256843  0.749   1.583609   1.367835  0.961       0.0
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

    threshold_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_threshold, gen_sd_threshold, n_participants)))
    ndt_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_ndt, gen_sd_ndt, n_participants)))

    cor_drift_sbj = gen_v0 + gen_wd * (gen_S_cor - gen_S_inc) + gen_ws * (gen_S_cor + gen_S_inc)
    inc_drift_sbj = gen_v0 + gen_wd * (gen_S_inc - gen_S_cor) + gen_ws * (gen_S_cor + gen_S_inc)

    # data['participant'] = np.repeat(np.arange(n_participants) + 1, n_trials)
    data['threshold'] = np.repeat(threshold_sbj, n_blocks * n_block_labels)
    data['ndt'] = np.repeat(ndt_sbj, n_blocks * n_block_labels)

    if gen_drift_trial_sd is None:
        data['cor_drift'] = np.repeat(cor_drift_sbj, n_blocks * n_block_labels)
        data['inc_drift'] = np.repeat(inc_drift_sbj, n_blocks * n_block_labels)
    else:
        data['cor_drift'] = np.random.normal(np.repeat(cor_drift_sbj, n_trials), gen_drift_trial_sd)
        data['inc_drift'] = np.random.normal(np.repeat(inc_drift_sbj, n_trials), gen_drift_trial_sd)

    rt, acc = random_rdm_2A(cor_drift=data['cor_drift'], inc_drift=data['inc_drift'], threshold=data['threshold'],
                            ndt=data['ndt'])

    data['rt'] = rt
    data['accuracy'] = acc
    # data['trial'] = np.tile(np.arange(1, n_trials + 1), n_participants)

    data = data.set_index(['participant', 'trial', 'trial_block'])

    return data
