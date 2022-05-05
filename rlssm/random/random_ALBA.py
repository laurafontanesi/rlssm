import numpy as np
import pandas as pd
from scipy import stats

from rlssm.random.random_LBA import random_lba_2A


def simulate_alba_2A(gen_S_cor,
                     gen_S_inc,
                     gen_threshold,  # A
                     gen_ndt,  # tau
                     gen_rel_sp,  # k
                     gen_v0,
                     gen_ws,
                     gen_wd,
                     gen_drift_trial_sd=None,
                     participant_label=1,
                     **kwargs):
    """Simulates behavior (rt and accuracy) according to the Advantage Linear Ballistic Accumulator.

    Note
    ----
    Parameters
    ----------
    gen_S_cor : float
        Brightness of correct trials.

    gen_S_inc : float
        Brightness of incorrect trials.

    gen_threshold : float
        Threshold of the ALBA model. Should be positive.

    gen_ndt : float
        Non decision time of the ALBA model, in seconds. Should be positive.

    gen_rel_sp : float
        Relative starting point of the ALBA model. Should be higher than 0 and smaller than 1.

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

        >>> data1 = simulate_alba_2A(gen_S_cor=np.random.normal(.4, 0.01, 100),
                                      gen_S_inc=np.random.normal(.3, 0.01, 100),
                                      gen_threshold=2, gen_ndt=.2,
                                      gen_rel_sp=.2, gen_v0=1,
                                      gen_ws=.7, gen_wd=1,
                                      gen_drift_trial_sd=.1)

                                    S_cor     S_inc  threshold  ...  inc_drift        rt  accuracy
        participant trial                                 ...
        1           1      0.398110  0.289163          2  ...   1.365724  0.947678       0.0
                    2      0.403420  0.286855          2  ...   1.306719  0.448267       1.0
                    3      0.370557  0.293840          2  ...   1.316852  0.583431       1.0
                    4      0.399567  0.310241          2  ...   1.547729  0.712771       0.0
                    5      0.410774  0.288761          2  ...   1.263873  0.735102       0.0
    """
    # return a pandas dataframe with the following columns:
    # index: participant + trial, cor_drift, inc_drift, threshold, ndt, rt, accuracy
    n_trials = np.shape(gen_S_cor)[0]

    data = pd.DataFrame({'participant': np.repeat(participant_label, n_trials)})

    data['S_cor'] = gen_S_cor
    data['S_inc'] = gen_S_inc
    data['threshold'] = gen_threshold
    data['ndt'] = gen_ndt
    data['rel_sp'] = gen_rel_sp

    gen_cor_drift = gen_v0 + gen_wd * (gen_S_cor - gen_S_inc) + gen_ws * (gen_S_cor + gen_S_inc)
    gen_inc_drift = gen_v0 + gen_wd * (gen_S_inc - gen_S_cor) + gen_ws * (gen_S_cor + gen_S_inc)

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

    data = data.set_index(['participant', 'trial'])

    return data


def simulate_hier_alba(n_trials, n_participants,
                       gen_v0, gen_ws, gen_wd,
                       gen_mu_drift_cor, gen_sd_drift_cor,
                       gen_mu_drift_inc, gen_sd_drift_inc,
                       gen_mu_threshold, gen_sd_threshold,
                       gen_mu_ndt, gen_sd_ndt,
                       gen_mu_rel_sp=.5, gen_sd_rel_sp=None,
                       gen_drift_trial_sd=None,
                       **kwargs):
    """Simulate behavior (rt and accuracy) according to a hierarchical Advantage Linear Ballistic Accumulator.

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
        >>> data_hier = simulate_hier_alba(n_trials=100, n_participants=30, gen_v0=1,
                                            gen_ws=.7, gen_wd=1,
                                            gen_mu_drift_cor=.4, gen_sd_drift_cor=0.01,
                                            gen_mu_drift_inc=.3, gen_sd_drift_inc=0.01,
                                            gen_mu_threshold=1, gen_sd_threshold=.1,
                                            gen_mu_ndt=.23, gen_sd_ndt=.1,
                                            gen_mu_rel_sp=.5, gen_sd_rel_sp=None,
                                            gen_drift_trial_sd=None)
        >>> print(data_hier.head())

                                   threshold       ndt  cor_drift  ...  rel_sp        rt  accuracy
        participant trial                                  ...
        1           1       1.223785  0.903213   1.589093  ...     0.5  1.675216       1.0
                    2       1.223785  0.903213   1.589093  ...     0.5  1.431298       0.0
                    3       1.223785  0.903213   1.589093  ...     0.5  1.344879       0.0
                    4       1.223785  0.903213   1.589093  ...     0.5  1.717715       0.0
                    5       1.223785  0.903213   1.589093  ...     0.5  1.490318       1.0

    """
    data = pd.DataFrame([])

    gen_S_cor = np.random.normal(gen_mu_drift_cor, gen_sd_drift_cor, n_participants)
    gen_S_inc = np.random.normal(gen_mu_drift_inc, gen_sd_drift_inc, n_participants)

    threshold_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_threshold, gen_sd_threshold, n_participants)))
    ndt_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_ndt, gen_sd_ndt, n_participants)))

    cor_drift_sbj = gen_v0 + gen_wd * (gen_S_cor - gen_S_inc) + gen_ws * (gen_S_cor + gen_S_inc)
    inc_drift_sbj = gen_v0 + gen_wd * (gen_S_inc - gen_S_cor) + gen_ws * (gen_S_cor + gen_S_inc)

    data['participant'] = np.repeat(np.arange(n_participants) + 1, n_trials)
    data['threshold'] = np.repeat(threshold_sbj, n_trials)
    data['ndt'] = np.repeat(ndt_sbj, n_trials)

    if gen_drift_trial_sd is None:
        data['cor_drift'] = np.repeat(cor_drift_sbj, n_trials)
        data['inc_drift'] = np.repeat(inc_drift_sbj, n_trials)
    else:
        data['cor_drift'] = np.random.normal(np.repeat(cor_drift_sbj, n_trials), gen_drift_trial_sd)
        data['inc_drift'] = np.random.normal(np.repeat(inc_drift_sbj, n_trials), gen_drift_trial_sd)

    if gen_sd_rel_sp is None:
        data['rel_sp'] = np.repeat(.5, n_participants * n_trials)
    else:
        rel_sp_sbj = stats.norm.cdf(np.random.normal(gen_mu_rel_sp, gen_sd_rel_sp, n_participants))
        data['rel_sp'] = np.repeat(rel_sp_sbj, n_trials)

    rt, acc = random_lba_2A(cor_drift=data['cor_drift'], inc_drift=data['inc_drift'], threshold=data['threshold'],
                            ndt=data['ndt'], rel_sp=data['rel_sp'])

    data['rt'] = rt
    data['accuracy'] = acc
    data['trial'] = np.tile(np.arange(1, n_trials + 1), n_participants)

    data = data.set_index(['participant', 'trial'])

    return data
