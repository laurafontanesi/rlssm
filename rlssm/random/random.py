import numpy as np
import pandas as pd
from scipy import stats
import random

## pure DDM
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

    Other Parameters
    ----------------

    rel_sp : numpy.ndarray or float, default .5
        When is an array , shape is usually (n_samples, n_trials).
        Relative starting point of the diffusion decision model.

    noise_constant : float, default 1
        Scaling factor of the diffusion decision model.
        If changed, drift and threshold would be scaled accordingly.
        Not to be changed in most applications.

    max_rt : float, default 10
        Controls the maximum rts that can be predicted.
        Making this higher might make the function a bit slower.

    dt : float, default 0.001
        Controls the time resolution of the diffusion decision model. Default is 1 msec.
        Lower values of dt make the function more precise but much slower.

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
    rel_sp = np.ones(shape)*rel_sp
    max_tsteps = max_rt/dt

    # initialize the diffusion process
    x = np.ones(shape)*rel_sp*threshold
    tstep = 0
    ongoing = np.array(np.ones(shape), dtype=bool)

    # start accumulation process
    while np.sum(ongoing) > 0 and tstep < max_tsteps:
        x[ongoing] += np.random.normal(drift[ongoing]*dt,
                                       noise_constant*np.sqrt(dt),
                                       np.sum(ongoing))
        tstep += 1

        # ended trials
        ended_correct = (x >= threshold)
        ended_incorrect = (x <= 0)

        # store results and filter out ended trials
        if np.sum(ended_correct) > 0:
            acc[np.logical_and(ended_correct, ongoing)] = 1
            rt[np.logical_and(ended_correct, ongoing)] = dt*tstep + ndt[np.logical_and(ended_correct,
                                                                                       ongoing)]
            ongoing[ended_correct] = False

        if np.sum(ended_incorrect) > 0:
            acc[np.logical_and(ended_incorrect, ongoing)] = 0
            rt[np.logical_and(ended_incorrect, ongoing)] = dt*tstep + ndt[np.logical_and(ended_incorrect,
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

    Other Parameters
    ----------------

    rel_sp : numpy.ndarray or float, default .5
        When is an array , shape is usually (n_samples, n_trials).
        Relative starting point of the diffusion decision model.

    noise_constant : float, default 1
        Scaling factor of the diffusion decision model.
        If changed, drift and threshold would be scaled accordingly.
        Not to be changed in most applications.

    max_rt : float, default 10
        Controls the maximum rts that can be predicted.
        Making this higher might make the function a bit slower.

    dt : float, default 0.001
        Controls the time resolution of the diffusion decision model. Default is 1 msec.
        Lower values of dt make the function more precise but much slower.

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

    n_tpoints = int(rt_max/dt)
    traces_size = (n_tpoints,) + drift.shape

    # initialize the diffusion process
    threshold = threshold[np.newaxis, :]
    starting_values = threshold*rel_sp

    # start accumulation process
    traces = np.random.normal(drift*dt, noise_constant*np.sqrt(dt), size=traces_size)
    traces = starting_values + np.cumsum(traces, axis=0)

    # look for threshold crossings
    up_passes = (traces >= threshold)
    down_passes = (traces <= 0)
    threshold_passes = np.argmax(np.logical_or(up_passes, down_passes), axis=0)

    # get rts
    rt = threshold_passes*dt
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
                 gen_drift_trialsd=None,
                 gen_rel_sp_trialsd=None,
                 **kwargs):
    """Simulates behavior (rt and accuracy) according to the diffusion decision model.

    This function is to simulate data for, for example, parameter recovery.

    Simulates data for one participant.

    In this parametrization, it is assumed that 0 is the lower threshold,
    and, when `rel_sp` = .5, the diffusion process starts halfway through the threshold value.

    Note
    ----
    When `gen_drift_trialsd` is not specified, there is no across-trial variability
    in the drift-rate.

    Instead, when `gen_drift_trialsd` is specified, the trial-by-trial drift-rate
    has the following distribution:

    - drift ~ normal(gen_drift, gen_drift_trialsd).

    Similarly, when `gen_rel_sp_trialsd` is not specified, there is no across-trial
    variability starting point.

    Instead, when `gen_rel_sp_trialsd` is specified, the trial-by-trial relative
    starting point has the following distribution:

    - rel_sp ~ Phi(normal(rel_sp, gen_rel_sp_trialsd)).

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

    Other Parameters
    ----------------

    gen_rel_sp : float, default .5
        Relative starting point of the diffusion decision model.
        Should be higher than 0 and smaller than 1.
        When is 0.5 (default), there is no bias.

    gen_drift_trialsd : float, optional
        Across trial variability in the drift-rate.
        Should be positive.

    gen_rel_sp_trialsd : float, optional
        Across trial variability in the realtive starting point.
        Should be positive.

    participant_label : string or float, default 1
        What will appear in the participant column of the output data.

    **kwargs
        Additional arguments to rlssm.random.random_ddm().

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
        >>> data = simulate_ddm(n_trials=1000,
                                gen_drift=.8,
                                gen_threshold=1.3,
                                gen_ndt=.23,
                                gen_rel_sp=.6)

        >>> print(data.head())
                participant  drift  rel_sp  threshold   ndt     rt  accuracy
        trial
        1                1    0.8     0.6        1.3  0.23  0.344       1.0
        2                1    0.8     0.6        1.3  0.23  0.376       0.0
        3                1    0.8     0.6        1.3  0.23  0.390       1.0
        4                1    0.8     0.6        1.3  0.23  0.434       0.0
        5                1    0.8     0.6        1.3  0.23  0.520       1.0

    To have trial number as a column::

        >>> print(data.reset_index())
            trial  participant  drift  rel_sp  threshold   ndt     rt  accuracy
        0        1            1    0.8     0.6        1.3  0.23  0.344       1.0
        1        2            1    0.8     0.6        1.3  0.23  0.376       0.0
        2        3            1    0.8     0.6        1.3  0.23  0.390       1.0
        3        4            1    0.8     0.6        1.3  0.23  0.434       0.0
        4        5            1    0.8     0.6        1.3  0.23  0.520       1.0
        ..     ...          ...    ...     ...        ...   ...    ...       ...
        995    996            1    0.8     0.6        1.3  0.23  0.423       1.0
        996    997            1    0.8     0.6        1.3  0.23  0.956       1.0
        997    998            1    0.8     0.6        1.3  0.23  0.347       1.0
        998    999            1    0.8     0.6        1.3  0.23  0.414       1.0
        999   1000            1    0.8     0.6        1.3  0.23  0.401       1.0

        [1000 rows x 8 columns]

    """
    data = pd.DataFrame({'participant': np.repeat(participant_label, n_trials)})

    if gen_drift_trialsd is None:
        data['drift'] = gen_drift
    else:
        data['drift'] = np.random.normal(gen_drift, gen_drift_trialsd, n_trials)
        data['drift_trialmu'] = gen_drift
        data['drift_trialsd'] = gen_drift_trialsd

    if gen_rel_sp_trialsd is None:
        data['rel_sp'] = gen_rel_sp
    else:
        gen_rel_sp = stats.norm.ppf(gen_rel_sp)
        data['rel_sp'] = stats.norm.cdf(np.random.normal(gen_rel_sp, gen_rel_sp_trialsd, n_trials))
        data['rel_sp_trialmu'] = gen_rel_sp
        data['transf_rel_sp_trialmu'] = stats.norm.cdf(gen_rel_sp)
        data['rel_sp_trialsd'] = gen_rel_sp_trialsd

    data['threshold'] = gen_threshold
    data['ndt'] = gen_ndt

    rt, acc = random_ddm(data['drift'], data['threshold'], data['ndt'], data['rel_sp'], **kwargs)

    data['rt'] = rt
    data['accuracy'] = acc
    data['trial'] = np.arange(1, n_trials+1)

    return data.set_index(['participant', 'trial'])

def simulate_hier_ddm(n_trials, n_participants,
                      gen_mu_drift, gen_sd_drift,
                      gen_mu_threshold, gen_sd_threshold,
                      gen_mu_ndt, gen_sd_ndt,
                      gen_mu_rel_sp=.5, gen_sd_rel_sp=None,
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

    When `gen_sd_rel_sp` is not specified, the relative starting point
    is assumed to be fixed across participants at `gen_mu_rel_sp`.

    Instead, when `gen_sd_rel_sp` is specified, the starting point
    has the following distribution:

    - rel_sp ~ Phi(normal(gen_mu_rel_sp, gen_sd_rel_sp))

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

    Other Parameters
    ----------------

    gen_mu_rel_sp : float, default .5
        Relative starting point of the diffusion decision model.
        When `gen_sd_rel_sp` is not specified, `gen_mu_rel_sp` is
        fixed across participants.
        When `gen_sd_rel_sp` is specified, `gen_mu_rel_sp` is the
        group-mean of the starting point.

    gen_sd_rel_sp : float, optional
        Group-standard deviation of the relative starting point
        of the diffusion decision model.

    **kwargs
        Additional arguments to `rlssm.random.random_ddm()`.

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

    data['participant'] = np.repeat(np.arange(n_participants)+1, n_trials)
    data['drift'] = np.repeat(drift_sbj, n_trials)
    data['threshold'] = np.repeat(threshold_sbj, n_trials)
    data['ndt'] = np.repeat(ndt_sbj, n_trials)

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
    data['trial'] = np.repeat(np.arange(1, n_trials+1), n_participants)

    return data.set_index(['participant', 'trial'])

# pure RL
def generate_task_design_fontanesi(n_trials_block,
                                   n_blocks,
                                   n_participants,
                                   trial_types,
                                   mean_options,
                                   sd_options):

    """Generates the RL stimuli as in the 2019 Fontanesi et al.'s paper.

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
        sequence = list(trial_types) * int(n_trials_block/n_trial_types)
        random.shuffle(sequence)

        count = 3
        while count < len(sequence):
            if sequence[count]==sequence[count-1]==sequence[count-2]==sequence[count-3]:
                random.shuffle(sequence)
                count = 2
            count += 1

        return(np.array(sequence))

    n_trials = n_trials_block*n_blocks

    task_design = pd.DataFrame({'participant': np.repeat(np.arange(1, n_participants+1), n_trials),
                                'block_label': np.tile(np.repeat(np.arange(1, n_blocks+1), n_trials_block), n_participants),
                                'trial_block': np.tile(np.arange(1, n_trials_block+1), n_blocks*n_participants),
                                'trial': np.tile(np.arange(1, n_trials+1), n_participants)})

    task_design['trial_type'] = np.concatenate([generate_trial_type_sequence(n_trials_block, trial_types) for i in range(n_blocks*n_participants)])
    task_design[['inc_option', 'cor_option']] = task_design.trial_type.str.split("-", expand=True).astype(int)

    options = pd.unique(task_design[['inc_option', 'cor_option']].values.ravel('K'))
    options = np.sort(options.astype(int))# sorted option numbers
    n_options = len(options)
    print("The task will be created with the following {} options: {}.".format(n_options, options))
    print("With mean (respectively): {} and SD: {}.".format(mean_options, sd_options))

    def reward_options(row):
        """Sample a reward from normal distribution for the cor/inc options in each row.
        """
        index_inc = int(row.inc_option - 1)
        f_inc = np.round(np.random.normal(mean_options[index_inc], sd_options[index_inc]))
        index_cor = int(row.cor_option - 1)
        f_cor = np.round(np.random.normal(mean_options[index_cor], sd_options[index_cor]))

        return pd.Series({'f_inc':int(f_inc), 'f_cor':int(f_cor)})

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

    alpha_pos : float, default None
        If a value for both alpha_pos and alpha_neg is provided,
        separate learning rates are estimated
        for positive and negative prediction errors.

    alpha_neg : float, default None
        If a value for both alpha_pos and alpha_neg is provided,
        separate learning rates are estimated
        for positive and negative prediction errors.

    initial_value_learning : float
        The initial value for Q learning.

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
            raise ValueError("Column {} should be included in the task_design.".format(label))
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
    n_options = len(options) # n Q values to be learned

    Q_cor = np.array([])
    Q_inc = np.array([])
    for n in range(n_trials):
        index_cor = int(task_design.cor_option.values[n]-1)
        index_inc = int(task_design.inc_option.values[n]-1)
        index_participant = np.where(participants == task_design.participant.values[n])[0][0]

        if task_design.trial_block.values[n] == 1:
            Q = np.ones(n_options)*initial_value_learning
        else:
            if separate_learning_rates:
                pe_cor = task_design.f_cor.values[n] - Q[index_cor]
                pe_inc = task_design.f_inc.values[n] - Q[index_inc]
                if pe_cor > 0:
                    Q[index_cor] += alpha_pos[index_participant]*(task_design.f_cor.values[n] - Q[index_cor])
                else:
                    Q[index_cor] += alpha_neg[index_participant]*(task_design.f_cor.values[n] - Q[index_cor])
                if pe_inc > 0:
                    Q[index_inc] += alpha_pos[index_participant]*(task_design.f_inc.values[n] - Q[index_inc])
                else:
                    Q[index_inc] += alpha_neg[index_participant]*(task_design.f_inc.values[n] - Q[index_inc])
            else:
                Q[index_cor] += alpha[index_participant]*(task_design.f_cor.values[n] - Q[index_cor])
                Q[index_inc] += alpha[index_participant]*(task_design.f_inc.values[n] - Q[index_inc])

        Q_cor = np.append(Q_cor, Q[index_cor])
        Q_inc = np.append(Q_inc, Q[index_inc])

    return pd.DataFrame({'Q_cor':Q_cor, 'Q_inc':Q_inc})

def _soft_max_2A(row):
    nom = np.exp(row.Q_cor*row.sensitivity)
    denom = np.sum([nom + np.exp(row.Q_inc*row.sensitivity)])
    return nom/denom

def simulate_rl_2A(task_design,
                              gen_alpha,
                              gen_sensitivity,
                              initial_value_learning=0):
    """Simulates behavior (accuracy) according to a RL model,

    where the learning component is the Q learning
    (delta learning rule) and the choice rule is the softmax.

    This function is to simulate data for, for example, parameter recovery.
    Simulates data for one participant.

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

    gen_sensitivity : float
        The generating sensitivity parameter for the soft_max choice rule.
        It should be a value higher than 0
        (the higher, the more sensitivity to value differences).

    initial_value_learning : float
        The initial value for Q learning.

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, that is the task_design, plus:
        'Q_cor', 'Q_inc', 'alpha', 'sensitivity',
        'p_cor', and 'accuracy'.

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
            pass # implement here Stefano's learning rule
        else:
            raise ValueError("The gen_alpha list should be of either length 2 or 3.")
    else:
        raise TypeError("The gen_alpha should be either a list or a float/int.")

    data['sensitivity'] = gen_sensitivity
    data['p_cor'] = data.apply(_soft_max_2A, axis=1)
    data['accuracy'] = stats.bernoulli.rvs(data['p_cor'].values) # simulate choices

    data = data.set_index(['participant', 'block_label', 'trial_block'])
    return data

def simulate_hier_rl_2A(task_design,
                                   gen_mu_alpha, gen_sd_alpha,
                                   gen_mu_sensitivity, gen_sd_sensitivity,
                                   initial_value_learning=0):
    """Simulates behavior (accuracy) according to a RL model,
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

    initial_value_learning : float
        The initial value for Q learning.

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, that is the task_design, plus:
        'Q_cor', 'Q_inc', 'alpha', 'sensitivity',
        'p_cor', and 'accuracy'.

    """
    data = task_design.copy()

    participants = pd.unique(data["participant"])
    n_participants = len(participants)
    if n_participants < 2:
        raise ValueError("You only have one participant. Use simulate_rl_2A instead.")

    if type(gen_mu_alpha) != type(gen_sd_alpha):
        raise TypeError("gen_mu_alpha and gen_sd_alpha should be of the same type.")

    if (type(gen_mu_alpha) == float) | (type(gen_mu_alpha) == int):
        parameters = pd.DataFrame({'alpha': stats.norm.cdf(np.random.normal(gen_mu_alpha, gen_sd_alpha, n_participants)),
                                   'sensitivity': np.log(1 + np.exp(np.random.normal(gen_mu_sensitivity, gen_sd_sensitivity, n_participants)))},
                                   index = participants)
        data = pd.concat([data.set_index('participant'), parameters], axis=1, ignore_index=False).reset_index().rename(columns={'index': 'participant'})

        data = pd.concat([data, _simulate_delta_rule_2A(task_design=task_design,
                                                                   alpha=parameters.alpha.values,
                                                                   initial_value_learning=initial_value_learning)],
                         axis=1)

    elif type(gen_mu_alpha) is list:
        if len(gen_mu_alpha) != len(gen_sd_alpha):
            raise ValueError("gen_mu_alpha and gen_sd_alpha should be of the same lenght.")
        if len(gen_mu_alpha) == 2:
            parameters = pd.DataFrame({'alpha_pos': stats.norm.cdf(np.random.normal(gen_mu_alpha[0], gen_sd_alpha[0], n_participants)),
                                       'alpha_neg': stats.norm.cdf(np.random.normal(gen_mu_alpha[1], gen_sd_alpha[1], n_participants)),
                                       'sensitivity': np.log(1 + np.exp(np.random.normal(gen_mu_sensitivity, gen_sd_sensitivity, n_participants)))},
                                       index = participants)
            data = pd.concat([data.set_index('participant'), parameters], axis=1, ignore_index=False).reset_index().rename(columns={'index': 'participant'})
            data = pd.concat([data, _simulate_delta_rule_2A(task_design=task_design,
                                                                       alpha=None,
                                                                       initial_value_learning=initial_value_learning,
                                                                       alpha_pos=parameters.alpha_pos.values,
                                                                       alpha_neg=parameters.alpha_neg.values)],
                             axis=1)

        elif len(gen_mu_alpha) == 3:
            pass # implement here Stefano's learning rule
        else:
            raise ValueError("The gen_mu_alpha list should be of either length 2 or 3.")
    else:
        raise TypeError("The gen_alpha should be either a list or a float/int.")

    data['p_cor'] = data.apply(_soft_max_2A, axis=1)
    data['accuracy'] = stats.bernoulli.rvs(data['p_cor'].values) # simulate choices

    data = data.set_index(['participant', 'block_label', 'trial_block'])
    return data

# RL + DDM
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

    initial_value_learning : float
        The initial value for Q learning.

    Other Parameters
    ----------------

    **kwargs
        Additional arguments to rlssm.random.random_ddm().

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, that is the task_design, plus:
        'Q_cor', 'Q_inc', 'drift', 'alpha', 'drift_scaling',
        'threshold', 'ndt', 'rt', and 'accuracy'.

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
            pass # implement here Stefano's learning rule
        else:
            raise ValueError("The gen_alpha list should be of either length 2 or 3.")
    else:
        raise TypeError("The gen_alpha should be either a list or a float/int.")

    data['drift_scaling'] = gen_drift_scaling
    data['threshold'] = gen_threshold
    data['ndt'] = gen_ndt
    data['drift'] = gen_drift_scaling*(data['Q_cor'] - data['Q_inc'])

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

    initial_value_learning : float
        The initial value for Q learning.

    Other Parameters
    ----------------

    **kwargs
        Additional arguments to rlssm.random.random_ddm().

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, that is the task_design, plus:
        'Q_cor', 'Q_inc', 'drift', 'alpha', 'drift_scaling',
        'threshold', 'ndt', 'rt', and 'accuracy'.

    """
    data = task_design.copy()
    participants = pd.unique(data["participant"])
    n_participants = len(participants)
    if n_participants < 2:
        raise ValueError("You only have one participant. Use simulate_rl_2A instead.")

    if type(gen_mu_alpha) != type(gen_sd_alpha):
        raise TypeError("gen_mu_alpha and gen_sd_alpha should be of the same type.")

    if (type(gen_mu_alpha) == float) | (type(gen_mu_alpha) == int):
        parameters = pd.DataFrame({'alpha': stats.norm.cdf(np.random.normal(gen_mu_alpha, gen_sd_alpha, n_participants)),
                                   'drift_scaling': np.log(1 + np.exp(np.random.normal(gen_mu_drift_scaling, gen_sd_drift_scaling, n_participants))),
                                   'threshold': np.log(1 + np.exp(np.random.normal(gen_mu_threshold, gen_sd_threshold, n_participants))),
                                   'ndt': np.log(1 + np.exp(np.random.normal(gen_mu_ndt, gen_sd_ndt, n_participants)))},
                                   index = participants)
        data = pd.concat([data.set_index('participant'), parameters], axis=1, ignore_index=False).reset_index().rename(columns={'index': 'participant'})
        data = pd.concat([data, _simulate_delta_rule_2A(task_design,
                                                                   parameters.alpha.values,
                                                                   initial_value_learning)],
                         axis=1)

    elif type(gen_mu_alpha) is list:
        if len(gen_mu_alpha) != len(gen_sd_alpha):
            raise ValueError("gen_mu_alpha and gen_sd_alpha should be of the same lenght.")
        if len(gen_mu_alpha) == 2:
            parameters = pd.DataFrame({'alpha_pos': stats.norm.cdf(np.random.normal(gen_mu_alpha[0], gen_sd_alpha[0], n_participants)),
                                       'alpha_neg': stats.norm.cdf(np.random.normal(gen_mu_alpha[1], gen_sd_alpha[1], n_participants)),
                                       'drift_scaling': np.log(1 + np.exp(np.random.normal(gen_mu_drift_scaling, gen_sd_drift_scaling, n_participants))),
                                       'threshold': np.log(1 + np.exp(np.random.normal(gen_mu_threshold, gen_sd_threshold, n_participants))),
                                       'ndt': np.log(1 + np.exp(np.random.normal(gen_mu_ndt, gen_sd_ndt, n_participants)))},
                                       index = participants)
            data = pd.concat([data.set_index('participant'), parameters], axis=1, ignore_index=False).reset_index().rename(columns={'index': 'participant'})
            data = pd.concat([data, _simulate_delta_rule_2A(task_design=task_design,
                                                                       alpha=None,
                                                                       initial_value_learning=initial_value_learning,
                                                                       alpha_pos=parameters.alpha_pos.values,
                                                                       alpha_neg=parameters.alpha_neg.values)],
                             axis=1)

        elif len(gen_mu_alpha) == 3:
            pass # implement here Stefano's learning rule
        else:
            raise ValueError("The gen_mu_alpha list should be of either length 2 or 3.")
    else:
        raise TypeError("The gen_alpha should be either a list or a float/int.")

    data['drift'] = data['drift_scaling']*(data['Q_cor'] - data['Q_inc'])

    # simulate responses
    rt, acc = random_ddm(data['drift'], data['threshold'], data['ndt'], .5, **kwargs)
    data['rt'] = rt
    data['accuracy'] = acc

    data = data.set_index(['participant', 'block_label', 'trial_block'])
    return data

# PURE RDM
def rdm_trial(I, threshold, non_decision_time, noise_constant=1, dt=0.001, max_rt=10):
    n_choice = len(I)
    x = np.zeros(n_choice)
    stop_race = False
    rt = 0

    while not stop_race:
        for i in range(n_choice):
            x[i] += np.random.normal(I[i]*dt, noise_constant*(dt**(1/2)), 1)[0]
        rt += dt
        not_reached = np.sum(x<threshold)
        if not_reached == n_choice:
            stop_race = False
            if rt > max_rt:
                x = np.zeros(n_choice)
                rt = 0
        elif not_reached == n_choice - 1:
            stop_race = True
        else:
            stop_race = False
            x = np.zeros(n_choice)
            rt = 0

    return rt+non_decision_time, np.where(x>=threshold)[0][0] + 1

def random_rdm_nA(drift, threshold, ndt, noise_constant=1, dt=0.001, max_rt=10):
    shape = ndt.shape
    n_options = drift.shape[1]
    choice = np.empty(shape)
    choice[:] = np.nan
    rt = np.empty(shape)
    rt[:] = np.nan

    max_tsteps = max_rt/dt

    x = np.zeros(drift.shape)
    tstep = 0
    ongoing = np.array(np.ones(drift.shape), dtype=bool)
    ended = np.array(np.ones(drift.shape), dtype=bool)

    stop_race = False

    while np.sum(ongoing) > 0 and tstep < max_tsteps:
        x[ongoing] += np.random.normal(drift[ongoing]*dt,
                                           noise_constant*np.sqrt(dt),
                                           np.sum(ongoing))
        tstep += 1

        for i in range(n_options):
            ended[:, i, :]= (x[:, i, :] >= threshold)

        # store results and filter out ended trials
        for i in range(n_options):
            if np.sum(ended[:, i, :]) > 0:
                choice[np.logical_and(ended[:, i, :], ongoing[:, i, :])] = i + 1
                rt[np.logical_and(ended[:, i, :], ongoing[:, i, :])] = dt*tstep + ndt[np.logical_and(ended[:, i, :], ongoing[:, i, :])]
                ongoing[:, i, :][ended[:, i, :]] = False

    return rt, choice

def random_rdm_2A(cor_drift, inc_drift, threshold, ndt, noise_constant=1, dt=0.001, max_rt=10):
    shape = cor_drift.shape
    acc = np.empty(shape)
    rt = np.empty(shape)
    acc[:] = np.nan
    rt[:] = np.nan

    max_tsteps = max_rt/dt

    x_cor = np.zeros(shape)
    x_inc = np.zeros(shape)

    tstep = 0
    ongoing = np.array(np.ones(shape), dtype=bool)

    stop_race = False

    while np.sum(ongoing) > 0 and tstep < max_tsteps:
        x_cor[ongoing] += np.random.normal(cor_drift[ongoing]*dt,
                                           noise_constant*np.sqrt(dt),
                                           np.sum(ongoing))
        x_inc[ongoing] += np.random.normal(inc_drift[ongoing]*dt,
                                           noise_constant*np.sqrt(dt),
                                           np.sum(ongoing))
        tstep += 1
        ended_correct = (x_cor >= threshold)
        ended_incorrect = (x_inc >= threshold)

        # store results and filter out ended trials
        if np.sum(ended_correct) > 0:
            acc[np.logical_and(ended_correct, ongoing)] = 1
            rt[np.logical_and(ended_correct, ongoing)] = dt*tstep + ndt[np.logical_and(ended_correct, ongoing)]
            ongoing[ended_correct] = False

        if np.sum(ended_incorrect) > 0:
            acc[np.logical_and(ended_incorrect, ongoing)] = 0
            rt[np.logical_and(ended_incorrect, ongoing)] = dt*tstep + ndt[np.logical_and(ended_incorrect, ongoing)]
            ongoing[ended_incorrect] = False
    return rt, acc

def simulate_rlrdm_2A(task_design,
                                 gen_alpha,
                                 gen_drift_scaling,
                                 gen_threshold,
                                 gen_ndt,
                                 initial_value_learning=0,
                                 **kwargs):
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
            pass # implement here Stefano's learning rule
        else:
            raise ValueError("The gen_alpha list should be of either length 2 or 3.")
    else:
        raise TypeError("The gen_alpha should be either a list or a float/int.")

    data['drift_scaling'] = gen_drift_scaling
    data['threshold'] = gen_threshold
    data['ndt'] = gen_ndt
    data['cor_drift'] = gen_drift_scaling*(data['Q_cor'])
    data['inc_drift'] = gen_drift_scaling*(data['Q_inc'])

    # simulate responses
    rt, acc = random_rdm_2A(data['cor_drift'],
                                       data['cor_drift'],
                                       data['threshold'],
                                       data['ndt'], **kwargs)
    data['rt'] = rt
    data['accuracy'] = acc

    data = data.set_index(['participant', 'block_label', 'trial_block'])
    return data

# PURE LBA
def random_lba_2A(k, A, tau, cor_drift, inc_drift):
    shape = cor_drift.shape
    acc = np.empty(shape)
    rt = np.empty(shape)
    acc[:] = np.nan
    rt[:] = np.nan

    b = k + A
    one_pose = True
    v_cor = np.random.normal(cor_drift, np.ones(cor_drift.shape))
    v_inc = np.random.normal(inc_drift, np.ones(inc_drift.shape))

    while one_pose:
        ind = np.logical_and(v_cor < 0, v_inc < 0)
        v_cor[ind] = np.random.normal(cor_drift[ind], np.ones(cor_drift[ind].shape))
        v_inc[ind] = np.random.normal(inc_drift[ind], np.ones(inc_drift[ind].shape))
        one_pose = np.sum(ind)>0

    start_cor = np.random.uniform(np.zeros(A.shape), A)
    start_inc = np.random.uniform(np.zeros(A.shape), A)

    ttf_cor = (b-start_cor)/v_cor
    ttf_inc = (b-start_inc)/v_inc

    ind = np.logical_and(ttf_cor <= ttf_inc, 0 < ttf_cor)
    acc[ind] = 1
    rt[ind] = ttf_cor[ind] + tau[ind]

    ind = np.logical_and(ttf_inc < 0, 0 < ttf_cor)
    acc[ind] = 1
    rt[ind] = ttf_cor[ind] + tau[ind]

    ind = np.logical_and(ttf_inc < ttf_cor, 0 < ttf_inc)
    acc[ind] = 0
    rt [ind] = ttf_inc[ind] + tau[ind]

    ind = np.logical_and(ttf_cor < 0, 0 < ttf_inc)
    acc[ind] = 0
    rt [ind] = ttf_inc[ind] + tau[ind]

    return rt, acc