import numpy as np
import pandas as pd


def random_rdm_2A(cor_drift, 
                  inc_drift, 
                  threshold, 
                  ndt, 
                  spvar=None,
                  starting_point_variability=False,
                  noise_constant=1, dt=0.001, max_rt=10):
    """ Simulates behavior (rt and accuracy) according to the Racing Diffusion Model.

    Parameters
    ----------

    cor_drift : numpy.ndarray
        Drift-rate of the Racing Diffusion Model - correct trials.

    inc_drift : numpy.ndarray
        Drift-rate of the Racing Diffusion Model - incorrect trials.

    threshold : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Threshold of the diffusion decision model.

    ndt : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Non decision time of the diffusion decision model, in seconds.

    noise_constant : float, default 1
        Scaling factor of the Racing Diffusion Model.
        If changed, drift and threshold would be scaled accordingly.
        Not to be changed in most applications.

    dt : float, default 0.001
        Controls the time resolution of the Racing Diffusion Model. Default is 1 msec.
        Lower values of dt make the function more precise but much slower.

    max_rt : float, default 10
        Controls the maximum rts that can be predicted.
        Making this higher might make the function a bit slower.

    Returns
    -------

    rt : numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated response times according to the Racing Diffusion Model.
        Every element corresponds to the set of parameters given as input with the same shape.

    acc: numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated accuracy according to the Racing Diffusion Model.
        Every element corresponds to the set of parameters given as input with the same shape.

    """
    # Based on the Wiener diffusion process
    shape = cor_drift.shape
    acc = np.empty(shape)
    rt = np.empty(shape)
    acc[:] = np.nan
    rt[:] = np.nan

    max_tsteps = max_rt / dt


    if starting_point_variability:
        x_cor = np.random.uniform(0, spvar)
        x_inc = np.random.uniform(0, spvar)
    else:
        x_cor = np.zeros(shape)
        x_inc = np.zeros(shape)

    tstep = 0
    ongoing = np.array(np.ones(shape), dtype=bool)

    stop_race = False

    while np.sum(ongoing) > 0 and tstep < max_tsteps:
        x_cor[ongoing] += np.random.normal(cor_drift[ongoing] * dt,
                                           noise_constant * np.sqrt(dt),
                                           np.sum(ongoing))
        x_inc[ongoing] += np.random.normal(inc_drift[ongoing] * dt,
                                           noise_constant * np.sqrt(dt),
                                           np.sum(ongoing))
        tstep += 1
        ended_correct = (x_cor >= threshold)
        ended_incorrect = (x_inc >= threshold)

        # store results and filter out ended trials
        if np.sum(ended_correct) > 0:
            acc[np.logical_and(ended_correct, ongoing)] = 1
            rt[np.logical_and(ended_correct, ongoing)] = dt * tstep + ndt[np.logical_and(ended_correct, ongoing)]
            ongoing[ended_correct] = False

        if np.sum(ended_incorrect) > 0:
            acc[np.logical_and(ended_incorrect, ongoing)] = 0
            rt[np.logical_and(ended_incorrect, ongoing)] = dt * tstep + ndt[np.logical_and(ended_incorrect, ongoing)]
            ongoing[ended_incorrect] = False
    return rt, acc


def simulate_rdm_2A(n_trials,
                    gen_cor_drift,
                    gen_inc_drift,
                    gen_threshold,
                    gen_ndt,
                    participant_label=1,
                    **kwargs):
    """Simulates behavior (rt and accuracy) according to the Racing Diffusion Model.

    This function is to simulate data for, for example, parameter recovery.

    Simulates data for one participant.

    Parameters
    ----------

    n_trials : int
        Number of trials to be simulated.

    gen_cor_drift : float
        Drift-rate of the Racing Diffusion Model - correct trials.

    gen_inc_drift : float
        Drift-rate of the Racing Diffusion Model - incorrect trials.

    gen_threshold : float
        Threshold of the Racing Diffusion Model.
        Should be positive.

    gen_ndt : float
        Non decision time of the Racing Diffusion Model, in seconds.
        Should be positive.

    participant_label : string or float, default 1
        What will appear in the participant column of the output data.

    Other Parameters
    ----------------

    **kwargs
        Additional arguments to rlssm.random.random_rdm_2A().

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, with n_trials rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters
        (both for each trial and across-trials when there is across-trial variability).

    Examples
    --------

        >>> data1 = simulate_rdm_2A(n_trials=1000,
                                    gen_cor_drift=.6,
                                    gen_inc_drift=.5,
                                    gen_threshold=1.4,
                                    gen_ndt=.23)
        >>> print(data.head())

                                   cor_drift  inc_drift  threshold   ndt      rt  accuracy
        participant trial
        1           1            0.6        0.5        1.4  0.23   1.056       0.0
                    2            0.6        0.5        1.4  0.23   1.812       1.0
                    3            0.6        0.5        1.4  0.23   1.204       1.0
                    4            0.6        0.5        1.4  0.23   5.283       1.0
                    5            0.6        0.5        1.4  0.23   0.713       1.0

    """
    # return a pandas dataframe with the following columns:
    # index: participant + trial, cor_drift, inc_drift, threshold, ndt, rt, accuracy
    data = pd.DataFrame({'participant': np.repeat(participant_label, n_trials)})

    data['cor_drift'] = gen_cor_drift
    data['inc_drift'] = gen_inc_drift

    data['threshold'] = gen_threshold
    data['ndt'] = gen_ndt

    rt, acc = random_rdm_2A(data['cor_drift'],
                            data['inc_drift'],
                            data['threshold'],
                            data['ndt'],
                            **kwargs)

    data['rt'] = rt
    data['accuracy'] = acc
    data['trial'] = np.arange(1, n_trials + 1)

    data = data.set_index(['participant', 'trial'])

    return data


def simulate_hier_rdm(n_trials, n_participants,
                      gen_mu_drift_cor, gen_sd_drift_cor,
                      gen_mu_drift_inc, gen_sd_drift_inc,
                      gen_mu_threshold, gen_sd_threshold,
                      gen_mu_ndt, gen_sd_ndt,
                      **kwargs):
    """Simulates behavior (rt and accuracy) according to the Racing Difussion Model.

    Parameters
    ----------

    n_trials : int
        Number of trials to simulate.

    n_participants : int
        Number of participants to simulate.

    gen_mu_drift_cor : float
        Group-mean of the drift-rate of the RDM for the correct responses.

    gen_sd_drift_cor : float
        Group-standard deviation of the drift-rate of the RDM for the correct responses.

    gen_mu_drift_inc : float
        Group-mean of the drift-rate of the RDM for the incorrect responses.

    gen_sd_drift_inc : float
        Group-standard deviation of the drift-rate of the RDM for the incorrect responses.

    gen_mu_threshold : float
        Group-mean of the threshold of the RDM.

    gen_sd_threshold : float
        Group-standard deviation of the threshold of the RDM.

    gen_mu_ndt : float
        Group-mean of the non-decision time of the RDM.

    gen_sd_ndt : float
        Group-standard deviation of the non-decision time of the RDM.

    Other Parameters
    ----------------

    **kwargs : dict
        Keyword arguments to be passed to `random_rdm_2A`.

    Returns
    -------

    data : pandas.DataFrame
        `pandas.DataFrame`, with n_trials*n_participants rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters (at the participant level).

    Example
    -------
    Simulate data for 30 participants with 100 trials each.

        >>> hier_data = simulate_hier_rdm(n_trials=100,
                                      n_participants=30,
                                      gen_mu_drift=1,
                                      gen_sd_drift=.5,
                                      gen_mu_threshold=1,
                                      gen_sd_threshold=.1,
                                      gen_mu_ndt=.23,
                                      gen_sd_ndt=.1)
        >>> print(hier_data.head())
                            cor_drift  inc_drift  threshold  ndt        rt  accuracy
        participant trial
        1           1       0.703100   1.557855   1.281720  0.684340  1.160340  0.0
                    1       0.703100   1.557855   1.281720  0.684340  0.986340  0.0
                    1       0.703100   1.557855   1.281720  0.684340  1.838340  1.0
                    1       0.703100   1.557855   1.281720  0.684340  1.083340  0.0
                    1       0.703100   1.557855   1.281720  0.684340  2.045340  1.0
    """
    data = pd.DataFrame([])

    cor_drift_sbj = np.random.normal(gen_mu_drift_cor, gen_sd_drift_cor, n_participants)
    inc_drift_sbj = np.random.normal(gen_mu_drift_inc, gen_sd_drift_inc, n_participants)

    threshold_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_threshold, gen_sd_threshold, n_participants)))
    ndt_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_ndt, gen_sd_ndt, n_participants)))

    data['participant'] = np.repeat(np.arange(n_participants) + 1, n_trials)
    data['cor_drift'] = np.repeat(cor_drift_sbj, n_trials)
    data['inc_drift'] = np.repeat(inc_drift_sbj, n_trials)
    data['threshold'] = np.repeat(threshold_sbj, n_trials)
    data['ndt'] = np.repeat(ndt_sbj, n_trials)

    rt, acc = random_rdm_2A(data['cor_drift'],
                            data['inc_drift'],
                            data['threshold'],
                            data['ndt'],
                            **kwargs)

    data['rt'] = rt
    data['accuracy'] = acc
    data['trial'] = np.tile(np.arange(1, n_trials + 1), n_participants)

    data = data.set_index(['participant', 'trial'])

    return data
