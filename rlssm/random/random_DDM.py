import numpy as np
import pandas as pd
from scipy import stats


def random_ddm(drift, threshold, ndt, rel_sp=.5, noise_constant=1, dt=0.001, max_rt=10):
    """Simulates behavior (rt and accuracy) according to the diffusion decision model.

    In this parametrization, it is assumed that 0 is the lower threshold,
    and, when rel_sp=1/2, the diffusion process starts halfway through the threshold value.

    Note
    ----

    This function is mainly for the posterior predictive calculations.
    It assumes that drift, threshold and ndt are provided as numpy.ndarray
    of shape (n_samples, n_trials).

    However, it also works when the rel_sp is given as a float.
    Drift, threshold and ndt should have the same shape.

    Parameters
    ----------

    drift : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Drift-rate of the diffusion decision model.

    threshold : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Threshold of the diffusion decision model.

    ndt : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Non decision time of the diffusion decision model, in seconds.

    rel_sp : numpy.ndarray or float, default .5
        When is an array , shape is usually (n_samples, n_trials).
        Relative starting point of the diffusion decision model.

    noise_constant : float, default 1
        Scaling factor of the diffusion decision model.
        If changed, drift and threshold would be scaled accordingly.
        Not to be changed in most applications.

    dt : float, default 0.001
        Controls the time resolution of the diffusion decision model. Default is 1 msec.
        Lower values of dt make the function more precise but much slower.

    max_rt : float, default 10
        Controls the maximum rts that can be predicted.
        Making this higher might make the function a bit slower.

    Returns
    -------

    rt : numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated response times according to the diffusion decision model.
        Every element corresponds to the set of parameters given as input with the same shape.

    acc: numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated accuracy according to the diffusion decision model.
        Every element corresponds to the set of parameters given as input with the same shape.

    """

    shape = drift.shape

    acc = np.empty(shape)
    acc[:] = np.nan
    rt = np.empty(shape)
    rt[:] = np.nan
    rel_sp = np.ones(shape) * rel_sp
    max_tsteps = max_rt / dt

    # initialize the diffusion process
    x = np.ones(shape) * rel_sp * threshold
    tstep = 0
    ongoing = np.array(np.ones(shape), dtype=bool)

    # start accumulation process
    while np.sum(ongoing) > 0 and tstep < max_tsteps:
        x[ongoing] += np.random.normal(drift[ongoing] * dt,
                                       noise_constant * np.sqrt(dt),
                                       np.sum(ongoing))
        tstep += 1

        # ended trials
        ended_correct = (x >= threshold)
        ended_incorrect = (x <= 0)

        # store results and filter out ended trials
        if np.sum(ended_correct) > 0:
            acc[np.logical_and(ended_correct, ongoing)] = 1
            rt[np.logical_and(ended_correct, ongoing)] = dt * tstep + ndt[np.logical_and(ended_correct,
                                                                                         ongoing)]
            ongoing[ended_correct] = False

        if np.sum(ended_incorrect) > 0:
            acc[np.logical_and(ended_incorrect, ongoing)] = 0
            rt[np.logical_and(ended_incorrect, ongoing)] = dt * tstep + ndt[np.logical_and(ended_incorrect,
                                                                                           ongoing)]
            ongoing[ended_incorrect] = False

    return rt, acc


def random_ddm_vector(drift, threshold, ndt, rel_sp=.5, noise_constant=1, dt=0.001, rt_max=10):
    """Simulates behavior (rt and accuracy) according to the diffusion decision model.

    In this parametrization, it is assumed that 0 is the lower threshold,
    and, when rel_sp=1/2, the diffusion process starts halfway through the threshold value.

    Note
    ----

    This is a vectorized version of rlssm.random.random_ddm().
    It seems to be generally slower, but might work for higher dt values
    and shorter rt_max (with less precision).
    There is more trade-off between dt and rt_max here
    compared to the random_ddm function.

    This function is mainly for the posterior predictive calculations.
    It assumes that drift, threshold and ndt are provided as numpy.ndarray
    of shape (n_samples, n_trials).

    However, it also works when the rel_sp and/or the ndt are given as floats.
    Drift and threshold should have the same shape.

    Parameters
    ----------

    drift : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Drift-rate of the diffusion decision model.

    threshold : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Threshold of the diffusion decision model.

    ndt : numpy.ndarray or float
        Shape is usually (n_samples, n_trials).
        Non decision time of the diffusion decision model, in seconds.

    rel_sp : numpy.ndarray or float, default .5
        When is an array , shape is usually (n_samples, n_trials).
        Relative starting point of the diffusion decision model.

    noise_constant : float, default 1
        Scaling factor of the diffusion decision model.
        If changed, drift and threshold would be scaled accordingly.
        Not to be changed in most applications.

    dt : float, default 0.001
        Controls the time resolution of the diffusion decision model. Default is 1 msec.
        Lower values of dt make the function more precise but much slower.

    rt_max : float, default 10
        Controls the maximum rts that can be predicted.
        Making this higher might make the function a bit slower.

    Returns
    -------

    rt : numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated response times according to the diffusion decision model.
        Every element corresponds to the set of parameters given as input with the same shape.

    acc: numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated accuracy according to the diffusion decision model.
        Every element corresponds to the set of parameters given as input with the same shape.

    """

    n_tpoints = int(rt_max / dt)
    traces_size = (n_tpoints,) + drift.shape

    # initialize the diffusion process
    threshold = threshold[np.newaxis, :]
    starting_values = threshold * rel_sp

    # start accumulation process
    traces = np.random.normal(drift * dt, noise_constant * np.sqrt(dt), size=traces_size)
    traces = starting_values + np.cumsum(traces, axis=0)

    # look for threshold crossings
    up_passes = (traces >= threshold)
    down_passes = (traces <= 0)
    threshold_passes = np.argmax(np.logical_or(up_passes, down_passes), axis=0)

    # get rts
    rt = threshold_passes * dt
    rt[rt == 0] = np.nan
    rt += ndt

    # get accuracy
    grid = np.ogrid[0:traces_size[0], 0:traces_size[1], 0:traces_size[2]]
    grid[0] = threshold_passes
    value_passes = traces[tuple(grid)].reshape(rt.shape)
    acc = (value_passes > 0).astype(int)
    acc[rt == 0] = np.nan

    return rt, acc


def simulate_ddm(n_trials,
                 gen_drift,
                 gen_threshold,
                 gen_ndt,
                 gen_rel_sp=.5,
                 participant_label=1,
                 gen_drift_trial_sd=None,
                 gen_rel_sp_trial_sd=None,
                 **kwargs):
    """Simulates behavior (rt and accuracy) according to the diffusion decision model.

    This function is to simulate data for, for example, parameter recovery.

    Simulates data for one participant.

    In this parametrization, it is assumed that 0 is the lower threshold,
    and, when `rel_sp` = .5, the diffusion process starts halfway through the threshold value.

    Note
    ----
    When `gen_drift_trial_sd` is not specified, there is no across-trial variability
    in the drift-rate.

    Instead, when `gen_drift_trial_sd` is specified, the trial-by-trial drift-rate
    has the following distribution:

    - drift ~ normal(gen_drift, gen_drift_trial_sd).

    Similarly, when `gen_rel_sp_trial_sd` is not specified, there is no across-trial
    variability starting point.

    Instead, when `gen_rel_sp_trial_sd` is specified, the trial-by-trial relative
    starting point has the following distribution:

    - rel_sp ~ Phi(normal(rel_sp, gen_rel_sp_trial_sd)).

    In this case, `gen_rel_sp` is first trasformed to the -Inf/+Inf scale,
    so the input value is the same (no bias still corresponds to .5).

    Parameters
    ----------

    n_trials : int
        Number of trials to be simulated.

    gen_drift : float
        Drift-rate of the diffusion decision model.

    gen_threshold : float
        Threshold of the diffusion decision model.
        Should be positive.

    gen_ndt : float
        Non decision time of the diffusion decision model, in seconds.
        Should be positive.

    gen_rel_sp : float, default .5
        Relative starting point of the diffusion decision model.
        Should be higher than 0 and smaller than 1.
        When is 0.5 (default), there is no bias.

    participant_label : string or float, default 1
        What will appear in the participant column of the output data.

    gen_drift_trial_sd : float, default None
        Across trial variability in the drift-rate.
        Should be positive.

    gen_rel_sp_trial_sd : float, default None
        Across trial variability in the realtive starting point.
        Should be positive.

    Other Parameters
    ----------------

    **kwargs
        Additional arguments to `rlssm.random.random_ddm`.

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, with n_trials rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters
        (both for each trial and across-trials when there is across-trial variability).

    Examples
    --------
    Simulate 1000 trials from 1 participant.

    Relative starting point is set towards the upper bound (higher than .5),
    so in this case there will be more accurate and fast decisions::

        from rlssm.random import simulate_ddm
        >>> data1 = simulate_ddm(n_trials=1000,
                                gen_drift=.8,
                                gen_threshold=1.3,
                                gen_ndt=.23,
                                gen_rel_sp=.6)

        >>> print(data1.head())
                           drift  rel_sp  threshold   ndt     rt  accuracy
        participant trial
        1           1        0.8     0.6        1.3  0.23  0.854       1.0
                    2        0.8     0.6        1.3  0.23  0.276       1.0
                    3        0.8     0.6        1.3  0.23  0.261       1.0
                    4        0.8     0.6        1.3  0.23  0.286       1.0
                    5        0.8     0.6        1.3  0.23  2.173       1.0

    To have trial number as a column::

        >>> print(data1.reset_index())
             participant  trial  drift  rel_sp  threshold   ndt     rt  accuracy
        0              1      1    0.8     0.6        1.3  0.23  0.349       1.0
        1              1      2    0.8     0.6        1.3  0.23  0.299       1.0
        2              1      3    0.8     0.6        1.3  0.23  0.660       0.0
        3              1      4    0.8     0.6        1.3  0.23  0.507       1.0
        4              1      5    0.8     0.6        1.3  0.23  0.844       0.0
        ..           ...    ...    ...     ...        ...   ...    ...       ...
        995            1    996    0.8     0.6        1.3  0.23  0.451       1.0
        996            1    997    0.8     0.6        1.3  0.23  1.054       1.0
        997            1    998    0.8     0.6        1.3  0.23  0.324       1.0
        998            1    999    0.8     0.6        1.3  0.23  0.292       1.0
        999            1   1000    0.8     0.6        1.3  0.23  0.522       0.0

        [1000 rows x 8 columns]

    """
    data = pd.DataFrame({'participant': np.repeat(participant_label, n_trials)})

    if gen_drift_trial_sd is None:
        data['drift'] = gen_drift
    else:
        data['drift'] = np.random.normal(gen_drift, gen_drift_trial_sd, n_trials)
        data['drift_trial_mu'] = gen_drift
        data['drift_trial_sd'] = gen_drift_trial_sd

    if gen_rel_sp_trial_sd is None:
        data['rel_sp'] = gen_rel_sp
    else:
        gen_rel_sp = stats.norm.ppf(gen_rel_sp)
        data['rel_sp'] = stats.norm.cdf(np.random.normal(gen_rel_sp, gen_rel_sp_trial_sd, n_trials))
        data['drift_trial_mu'] = gen_rel_sp
        data['transf_rel_sp_trial_mu'] = stats.norm.cdf(gen_rel_sp)
        data['rel_sp_trial_sd'] = gen_rel_sp_trial_sd

    data['threshold'] = gen_threshold
    data['ndt'] = gen_ndt

    rt, acc = random_ddm(data['drift'], data['threshold'], data['ndt'], data['rel_sp'], **kwargs)

    data['rt'] = rt
    data['accuracy'] = acc
    data['trial'] = np.arange(1, n_trials + 1)

    return data.set_index(['participant', 'trial'])


def simulate_hier_ddm(n_trials, n_participants,
                      gen_mu_drift, gen_sd_drift,
                      gen_mu_threshold, gen_sd_threshold,
                      gen_mu_ndt, gen_sd_ndt,
                      gen_mu_rel_sp=.5, gen_sd_rel_sp=None,
                      gen_drift_trial_sd=None,
                      **kwargs):
    """Simulates behavior (rt and accuracy) according to the diffusion decision model.

    This function is to simulate data for, for example, parameter recovery.

    Simulates hierarchical data for a group of participants.

    In this parametrization, it is assumed that 0 is the lower threshold,
    and, when `rel_sp` = .5, the diffusion process starts halfway through the threshold value.

    The individual parameters have the following distributions:

    - drift ~ normal(gen_mu_drift, gen_sd_drift)

    - threshold ~ log(1 + exp(normal(gen_mu_threshold, gen_sd_threshold)))

    - ndt ~ log(1 + exp(normal(gen_mu_ndt, gen_sd_ndt)))

    Note
    ----

    When `gen_drift_trial_sd` is not specified, there is no across-trial variability
    in the drift-rate.

    Instead, when `gen_drift_trial_sd` is specified, the trial-by-trial drift-rate
    has the following distribution:

    - drift ~ normal(gen_drift, gen_drift_trial_sd).

    When `gen_rel_sp_trial_sd` is not specified, there is no across-trial
    variability starting point.

    Instead, when `gen_rel_sp_trial_sd` is specified, the trial-by-trial relative
    starting point has the following distribution:

    - rel_sp ~ Phi(normal(rel_sp, gen_rel_sp_trial_sd)).

    Parameters
    ----------

    n_trials : int
        Number of trials to be simulated.

    n_participants : int
        Number of participants to be simulated.

    gen_mu_drift : float
        Group-mean of the drift-rate
        of the diffusion decision model.

    gen_sd_drift: float
        Group-standard deviation of the drift-rate
        of the diffusion decision model.

    gen_mu_threshold : float
        Group-mean of the threshold
        of the diffusion decision model.

    gen_sd_threshold: float
        Group-standard deviation of the threshold
        of the diffusion decision model.

    gen_mu_ndt : float
        Group-mean of the non decision time
        of the diffusion decision model.

    gen_sd_ndt : float
        Group-standard deviation of the non decision time
        of the diffusion decision model.

    gen_mu_rel_sp : float, default .5
        Relative starting point of the diffusion decision model.
        When `gen_sd_rel_sp` is not specified, `gen_mu_rel_sp` is
        fixed across participants.
        When `gen_sd_rel_sp` is specified, `gen_mu_rel_sp` is the
        group-mean of the starting point.


    gen_sd_rel_sp : float, default None
        Across trial variability in the drift-rate.
        Should be positive.


    gen_drift_trial_sd : float, default None
        Across trial variability in the realtive starting point.
        Should be positive.

    Other Parameters
    ----------------

    **kwargs
        Additional arguments to `rlssm.random.random_ddm`.

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, with n_trials*n_participants rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters (at the participant level).

    Examples
    --------

    Simulate data from 15 participants, with 200 trials each.

    Relative starting point is on average across participants
    towards the upper bound. So, in this case, there will be
    more accurate and fast decisions::

        from rlssm.random import simulate_hier_ddm
        >>> data = simulate_hier_ddm(n_trials=200,
                                     n_participants=15,
                                     gen_mu_drift=.6, gen_sd_drift=.3,
                                     gen_mu_threshold=.5, gen_sd_threshold=.1,
                                     gen_mu_ndt=-1.2, gen_sd_ndt=.05,
                                     gen_mu_rel_sp=.1, gen_sd_rel_sp=.05)

        >>> print(data.head())
                              drift  threshold       ndt    rel_sp        rt  accuracy
        participant trial
        1           1      0.773536   1.753562  0.300878  0.553373  0.368878       1.0
                    1      0.773536   1.753562  0.300878  0.553373  0.688878       1.0
                    1      0.773536   1.753562  0.300878  0.553373  0.401878       1.0
                    1      0.773536   1.753562  0.300878  0.553373  1.717878       1.0
                    1      0.773536   1.753562  0.300878  0.553373  0.417878       1.0

    Get mean response time and accuracy per participant::

        >>> print(data.groupby('participant').mean()[['rt', 'accuracy']])
                           rt  accuracy
        participant
        1            0.990313     0.840
        2            0.903228     0.740
        3            1.024509     0.815
        4            0.680104     0.760
        5            0.994501     0.770
        6            0.910615     0.865
        7            0.782978     0.700
        8            1.189268     0.740
        9            0.997170     0.760
        10           0.966897     0.750
        11           0.730522     0.855
        12           1.011454     0.590
        13           0.972070     0.675
        14           0.849755     0.625
        15           0.940542     0.785

    To have participant and trial numbers as a columns::

        >>> print(data.reset_index())
              participant  trial     drift  threshold       ndt    rel_sp        rt  accuracy
        0               1      1  0.773536   1.753562  0.300878  0.553373  0.368878       1.0
        1               1      1  0.773536   1.753562  0.300878  0.553373  0.688878       1.0
        2               1      1  0.773536   1.753562  0.300878  0.553373  0.401878       1.0
        3               1      1  0.773536   1.753562  0.300878  0.553373  1.717878       1.0
        4               1      1  0.773536   1.753562  0.300878  0.553373  0.417878       1.0
        ...           ...    ...       ...        ...       ...       ...       ...       ...
        2995           15    200  0.586573   1.703662  0.302842  0.556116  0.826842       1.0
        2996           15    200  0.586573   1.703662  0.302842  0.556116  0.925842       1.0
        2997           15    200  0.586573   1.703662  0.302842  0.556116  0.832842       1.0
        2998           15    200  0.586573   1.703662  0.302842  0.556116  0.628842       1.0
        2999           15    200  0.586573   1.703662  0.302842  0.556116  0.856842       1.0

        [3000 rows x 8 columns]

    """

    data = pd.DataFrame([])

    drift_sbj = np.random.normal(gen_mu_drift, gen_sd_drift, n_participants)
    threshold_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_threshold, gen_sd_threshold, n_participants)))
    ndt_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_ndt, gen_sd_ndt, n_participants)))

    data['participant'] = np.repeat(np.arange(n_participants) + 1, n_trials)
    data['threshold'] = np.repeat(threshold_sbj, n_trials)
    data['ndt'] = np.repeat(ndt_sbj, n_trials)

    if gen_drift_trial_sd is None:
        data['drift'] = np.repeat(drift_sbj, n_trials)
    else:
        data['drift'] = np.random.normal(np.repeat(drift_sbj, n_trials), gen_drift_trial_sd)

    if gen_sd_rel_sp is None:
        rt, acc = random_ddm(data['drift'], data['threshold'], data['ndt'], rel_sp=.5, **kwargs)

    else:
        rel_sp_sbj = stats.norm.cdf(np.random.normal(gen_mu_rel_sp, gen_sd_rel_sp, n_participants))
        data['rel_sp'] = np.repeat(rel_sp_sbj, n_trials)
        rt, acc = random_ddm(data['drift'],
                             data['threshold'],
                             data['ndt'],
                             data['rel_sp'],
                             **kwargs)

    data['rt'] = rt
    data['accuracy'] = acc
    data['trial'] = np.tile(np.arange(1, n_trials + 1), n_participants)

    data = data.set_index(['participant', 'trial'])

    return data
