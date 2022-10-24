import numpy as np
import pandas as pd

from rlssm.random.random_RDM import random_rdm_2A


def simulate_ardm_2A(gen_S_cor,
                     gen_S_inc,
                     gen_threshold,
                     gen_ndt,
                     gen_v0,
                     gen_ws,
                     gen_wd,
                     gen_sp_trial_var=None,
                     participant_label=1,
                     **kwargs):
    """Simulates behavior (rt and accuracy) according to the Advantage Racing Diffusion Model.

    Parameters
    ----------

    gen_S_cor : float
        Strongness of correct option.

    gen_S_inc : float
        Strongness of incorrect option.

    gen_threshold : float
        Threshold parameter.

    gen_ndt : float
        Non-decision time parameter, in seconds.

    gen_v0 : float
        The Bias parameter; ensures each accumulator has a positive drift rate.

    gen_ws : float
        Sum weight parameter.

    gen_wd : float
        Difference weight parameter.

    gen_sp_trial_var : float, default None
        Across trial starting point variability parameter.

    participant_label : string or float, default 1
        What will appear in the participant column of the output data.

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
    """
    n_trials = np.shape(gen_S_cor)[0]

    data = pd.DataFrame({'participant': np.repeat(participant_label, n_trials)})

    data['S_cor'] = gen_S_cor
    data['S_inc'] = gen_S_inc
    data['threshold'] = gen_threshold
    data['ndt'] = gen_ndt

    gen_cor_drift = gen_v0 + gen_wd * (gen_S_cor - gen_S_inc) + gen_ws * (gen_S_cor + gen_S_inc)
    gen_inc_drift = gen_v0 + gen_wd * (gen_S_inc - gen_S_cor) + gen_ws * (gen_S_cor + gen_S_inc)

    data['cor_drift'] = gen_cor_drift
    data['inc_drift'] = gen_inc_drift
    
    if gen_sp_trial_var is None:
        rt, acc = random_rdm_2A(data['cor_drift'],
                                data['inc_drift'],
                                data['threshold'], 
                                data['ndt'])
    else:
        data['sp_trial_var'] = gen_sp_trial_var

        rt, acc = random_rdm_2A(data['cor_drift'],
                                data['inc_drift'],
                                data['threshold'], 
                                data['ndt'],
                                spvar=data['sp_trial_var'],
                                starting_point_variability=True, **kwargs)

    data['rt'] = rt
    data['accuracy'] = acc
    data['trial'] = np.arange(1, n_trials + 1)

    data = data.set_index(['participant', 'trial'])

    return data


def simulate_hier_ardm(n_trials, n_participants,
                       gen_S_cor, gen_S_inc,
                       gen_mu_v0, gen_sd_v0, 
                       gen_mu_ws, gen_sd_ws,
                       gen_mu_wd, gen_sd_wd,
                       gen_mu_threshold, gen_sd_threshold,
                       gen_mu_ndt, gen_sd_ndt,
                       gen_mu_sp_trial_var=None, gen_sd_sp_trial_var=None,
                       **kwargs):
    """Simulate behavior (rt and accuracy) according to a hierarchical Advantage Racing Diffusion Model.

    Parameters
    ---------
    n_trials : int
        Number of trials to simulate.

    n_participants : int
        Number of participants to simulate.

    gen_S_cor: np.array
        Strongness of correct options.
    
    gen_S_inc: np.array
        Strongness of incorrect options.

   gen_mu_v0 : float
        Group-mean of the bias parameter; ensures each accumulator has a positive drift rate.

    gen_sd_v0 : float
        Group-standard deviation of the bias parameter.

    gen_mu_ws : float
        Group-mean of sum weight parameter. 

    gen_sd_ws : float
        Group-standard deviation of sum weight parameter. 

    gen_mu_wd : float
        Group-mean of difference weight parameter. 

    gen_sd_wd : float
        Group-standard deviation of difference weight parameter.

    gen_mu_threshold : float
        Group-mean threshold of the advantage Racing Diffusion Model.

    gen_sd_threshold : float
        Group-standard deviation of the threshold of the advantage Racing Diffusion Model.

    gen_mu_ndt : float
        Group-mean non-decision time of the advantage Racing Diffusion Model.

    gen_sd_ndt : float
        Group-standard deviation of the non-decision time of the advantage Racing Diffusion Model.

    gen_mu_sp_trial_var : float, default None
        Group-mean of across trial varibility for starting point parameter.

    gen_sd_sp_trial_var : float, default None
        Group-standard deviation of across trial varibility for starting point parameter.

    Other Parameters
    ----------------

    **kwargs : dict
        Additional parameters to be passed to `random_rdm_2A`.

    Returns
    -------
    data : DataFrame
        `pandas.DataFrame`, with n_trials*n_participants rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters (at the participant level).
    """
    data = pd.DataFrame([])

    v0_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_v0, gen_sd_v0, n_participants)))
    ws_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_ws, gen_sd_ws, n_participants)))
    wd_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_wd, gen_sd_wd, n_participants)))

    threshold_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_threshold, gen_sd_threshold, n_participants)))
    ndt_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_ndt, gen_sd_ndt, n_participants)))
    
    cor_drift_sbj = v0_sbj + wd_sbj * (gen_S_cor - gen_S_inc) + ws_sbj * (gen_S_cor + gen_S_inc)
    inc_drift_sbj = v0_sbj + wd_sbj * (gen_S_inc - gen_S_cor) + ws_sbj * (gen_S_cor + gen_S_inc)

    data['participant'] = np.repeat(np.arange(n_participants) + 1, n_trials)
    
    data['cor_drift'] = np.repeat(cor_drift_sbj, n_trials)
    data['inc_drift'] = np.repeat(inc_drift_sbj, n_trials)
    data['threshold'] = np.repeat(threshold_sbj, n_trials)
    data['ndt'] = np.repeat(ndt_sbj, n_trials)
    
    if gen_mu_sp_trial_var is None or gen_sd_sp_trial_var is None:
        rt, acc = random_rdm_2A(data['cor_drift'],
                                data['inc_drift'], 
                                data['threshold'],
                                data['ndt'])
    else:
        sp_trial_var_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_sp_trial_var, 
                                                              gen_sd_sp_trial_var, 
                                                              n_participants)))
        data['sp_trial_var'] = np.repeat(sp_trial_var_sbj, n_trials)
        rt, acc = random_rdm_2A(data['cor_drift'],
                                data['inc_drift'], 
                                data['threshold'],
                                data['ndt'],
                                data['sp_trial_var'], True, **kwargs)

    data['rt'] = rt
    data['accuracy'] = acc
    data['trial'] = np.tile(np.arange(1, n_trials + 1), n_participants)

    data = data.set_index(['participant', 'trial'])

    return data
