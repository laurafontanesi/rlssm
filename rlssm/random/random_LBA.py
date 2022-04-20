import numpy as np
import pandas as pd
from scipy import stats


def random_lba_2A(cor_drift, inc_drift, threshold, ndt, rel_sp):
    """Simulates behavior (rt and accuracy) according to the Linear Ballistic Accumulator.

    Parameters
    ----------

    rel_sp : float
        Relative starting point of the Linear Ballistic Accumulator. Also called k.

    threshold : float
        Threshold of the Linear Ballistic Accumulator. Also called A.

    ndt : float
        Non-decision time of the Linear Ballistic Accumulator. Also called tau.

    cor_drift : float
        Drift-rate of the Linear Ballistic Accumulator - correct responses.

    inc_drift : float
        Drift-rate of the Linear Ballistic Accumulator - incorrect responses.

    Returns
    -------

    rt : numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated response times according to the Linear Ballistic Accumulator.
        Every element corresponds to the set of parameters given as input with the same shape.

    acc: numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated accuracy according to the Linear Ballistic Accumulator.
        Every element corresponds to the set of parameters given as input with the same shape.
    """
    shape = cor_drift.shape
    acc = np.empty(shape)
    rt = np.empty(shape)
    acc[:] = np.nan
    rt[:] = np.nan

    b = rel_sp + threshold
    one_pose = True
    v_cor = np.array(cor_drift)
    v_inc = np.array(inc_drift)

    # this while loop might be wrong
    while one_pose:
        ind = np.logical_and(v_cor < 0, v_inc < 0)
        v_cor[ind] = np.random.normal(cor_drift[ind], np.ones(cor_drift[ind].shape))
        v_inc[ind] = np.random.normal(inc_drift[ind], np.ones(inc_drift[ind].shape))
        one_pose = np.sum(ind) > 0

    start_cor = np.random.uniform(np.zeros(threshold.shape), threshold)
    start_inc = np.random.uniform(np.zeros(threshold.shape), threshold)

    ttf_cor = (b - start_cor) / v_cor
    ttf_inc = (b - start_inc) / v_inc

    ind = np.logical_and(ttf_cor <= ttf_inc, 0 < ttf_cor)
    acc[ind] = 1
    rt[ind] = ttf_cor[ind] + ndt[ind]

    ind = np.logical_and(ttf_inc < 0, 0 < ttf_cor)
    acc[ind] = 1
    rt[ind] = ttf_cor[ind] + ndt[ind]

    ind = np.logical_and(ttf_inc < ttf_cor, 0 < ttf_inc)
    acc[ind] = 0
    rt[ind] = ttf_inc[ind] + ndt[ind]

    ind = np.logical_and(ttf_cor < 0, 0 < ttf_inc)
    acc[ind] = 0
    rt[ind] = ttf_inc[ind] + ndt[ind]

    return rt, acc


def simulate_lba_2A(n_trials,
                    gen_cor_drift,
                    gen_inc_drift,
                    gen_threshold,  # A
                    gen_ndt,  # tau
                    gen_rel_sp,  # k
                    gen_drift_trial_sd=None,
                    participant_label=1,
                    **kwargs):
    """Simulates behavior (rt and accuracy) according to the Linear Ballistic Accumulator.

    Note
    ----
    When `gen_drift_trial_sd` is not specified, there is no across-trial variability
    in the drift-rate.

    Instead, when `gen_drift_trial_sd` is specified, the trial-by-trial drift-rate
    has the following distribution:

    - drift ~ normal(gen_drift, gen_drift_trial_sd).

    Parameters
    ----------
    n_trials : int
        Number of trials to simulate.

    gen_cor_drift : float
        Drift-rate of the Linear Ballistic Accumulator - correct trials.

    gen_inc_drift : float
        Drift-rate of the Linear Ballistic Accumulator - incorrect trials.

    gen_threshold : float
        Threshold of the Linear Ballistic Accumulator. Also called A.

    gen_ndt : float
        Non-decision time of the Linear Ballistic Accumulator. Also called tau.

    gen_rel_sp : float
        Relative starting point of the Linear Ballistic Accumulator. Also called k.

    Optional Parameters
    -------------------

    gen_drift_trial_sd : float, default None
        Across trial variability in the drift-rate.
        Should be positive.

    participant_label : string or float, default 1
        What will appear in the participant column of the output data.

    **kwargs
        Additional arguments to rlssm.random.random_lba_2A().

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, with n_trials rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters
        (both for each trial and across-trials when there is across-trial variability).

    Example
    -------

        >>> data1 = simulate_lba_2A(n_trials=1000,
                                     gen_cor_drift=.6,
                                     gen_inc_drift=.4,
                                     gen_threshold=1.5,
                                     gen_ndt=.23,
                                     gen_rel_sp=.8,
                                     gen_drift_trial_sd=.5)

        >>> print(data1.head())

                             k   tau    A  cor_drift  inc_drift        rt  accuracy
        participant trial
        1           1      0.8  0.23  1.5   0.453216   0.806479  1.534426       0.0
                    2      0.8  0.23  1.5   0.725115  -0.061165  1.861768       1.0
                    3      0.8  0.23  1.5   0.478014   0.412137  3.400750       1.0
                    4      0.8  0.23  1.5   0.966026   1.184433  1.290519       1.0
                    5      0.8  0.23  1.5   1.125275   0.268871  1.906879       1.0
    """
    # return a pandas dataframe with the following columns:
    # index: participant + trial, cor_drift, inc_drift, threshold, ndt, rt, accuracy
    data = pd.DataFrame({'participant': np.repeat(participant_label, n_trials)})

    data['rel_sp'] = gen_rel_sp
    data['ndt'] = gen_ndt
    data['threshold'] = gen_threshold

    if gen_drift_trial_sd is None:
        data['cor_drift'] = gen_cor_drift
        data['inc_drift'] = gen_inc_drift
    else:
        data['cor_drift'] = np.random.normal(gen_cor_drift, gen_drift_trial_sd, n_trials)
        data['inc_drift'] = np.random.normal(gen_inc_drift, gen_drift_trial_sd, n_trials)

    rt, acc = random_lba_2A(cor_drift=data['cor_drift'], inc_drift=data['inc_drift'],
                            threshold=data['threshold'], ndt=data['ndt'], rel_sp=data['rel_sp'])

    data['rt'] = rt
    data['accuracy'] = acc
    data['trial'] = np.arange(1, n_trials + 1)

    data = data.set_index(['participant', 'trial'])

    return data


def simulate_hier_lba(n_trials, n_participants,
                      gen_mu_drift_cor, gen_sd_drift_cor,
                      gen_mu_drift_inc, gen_sd_drift_inc,
                      gen_mu_threshold, gen_sd_threshold,
                      gen_mu_ndt, gen_sd_ndt,
                      gen_mu_rel_sp=.5, gen_sd_rel_sp=None,
                      gen_drift_trial_sd=None,
                      **kwargs):
    """Simulate behavior (rt and accuracy) according to a hierarchical linear ballistic accumulator.

    Parameters
    ----------

    n_trials : int
        Number of trials to be simulated.

    n_participants : int
        Number of participants to be simulated.

    gen_mu_drift_cor : float
        Mean of the drift rate for correct trials.

    gen_sd_drift_cor : float
        Standard deviation of the drift rate for correct trials.

    gen_mu_drift_inc : float
        Mean of the drift rate for incorrect trials.

    gen_sd_drift_inc : float
        Standard deviation of the drift rate for incorrect trials.

    gen_mu_threshold : float
        Group-mean threshold of the linear ballistic accumulator.

    gen_sd_threshold : float
        Group-standard deviation of the threshold of the linear ballistic accumulator.

    gen_mu_ndt : float
        Group-mean non-decision time of the linear ballistic accumulator.

    gen_sd_ndt : float
        Group-standard deviation of the non-decision time of the linear ballistic accumulator.

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

    **kwargs : dict
        Additional arguments to `rlssm.random.random_lba_2A()`.

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, with n_trials*n_participants rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters (at the participant level).

    Example
    -------

        >>> data_hier = simulate_hier_lba(n_trials=100, n_participants=30,
                                           gen_mu_drift_cor=1, gen_sd_drift_cor=.5,
                                           gen_mu_drift_inc=1, gen_sd_drift_inc=.5,
                                           gen_mu_threshold=1, gen_sd_threshold=.1,
                                           gen_mu_ndt=.23, gen_sd_ndt=.1,
                                           gen_mu_rel_sp=.1, gen_sd_rel_sp=.05)

        >>> print(data_hier.head())

                           threshold       ndt  cor_drift  ...    rel_sp        rt  accuracy
        participant trial                                  ...
        1           1       1.329862  0.825293   1.593415  ...  0.559247  1.867468       1.0
                    2       1.329862  0.825293   1.593415  ...  0.559247  1.633809       1.0
                    3       1.329862  0.825293   1.593415  ...  0.559247  1.664186       1.0
                    4       1.329862  0.825293   1.593415  ...  0.559247  1.722936       1.0
                    5       1.329862  0.825293   1.593415  ...  0.559247  1.996309       1.0
    """
    data = pd.DataFrame([])

    cor_drift_sbj = np.random.normal(gen_mu_drift_cor, gen_sd_drift_cor, n_participants)
    inc_drift_sbj = np.random.normal(gen_mu_drift_inc, gen_sd_drift_inc, n_participants)

    threshold_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_threshold, gen_sd_threshold, n_participants)))
    ndt_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_ndt, gen_sd_ndt, n_participants)))

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
        data['rel_sp'] = np.repeat(.5, n_trials)
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
