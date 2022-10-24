import numpy as np
import pandas as pd
from scipy import stats

from rlssm.random.random_LBA import random_lba_2A
from rlssm.random.random_common import _simulate_delta_rule_2A

def simulate_rllba_2A(task_design,
                       gen_alpha,
                       gen_sp_trial_var,
                       gen_ndt,
                       gen_k,
                       gen_drift_scaling,
                       gen_slop=None,
                       gen_drift_asym=None,
                       gen_drift_trial_var=None,
                       nonlinear_mapping=False,
                       initial_value_learning=0,
                       **kwargs):
    """Simulates behavior (rt and accuracy) according to the RL-LBA model.

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

    gen_sp_trial_var : float
        sp_trial_var of the rlalba model. Should be positive.

    gen_ndt : float
        Non decision time of the rlalba model, in seconds. Should be positive.

    gen_k : float, list, or numpy.ndarray
        Distance between starting point variability and threshold.

    gen_drift_scaling : float
        The drift scaling parameter.

    gen_slop : float, default None
        The slop parameter.

    gen_drift_asym : float, default None
        The drift asymmetry parameter.

    nonlinear_mapping : bool, default False
        Whether to use a nonlinear mapping or not.

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

    if nonlinear_mapping:
        if gen_slop == None or gen_drift_asym == None:
            raise ValueError("The gen_slop and gen_drift_asym can not get \'None\' with nonlinear_mapping mechanism! ")

    data['k'] = gen_k
    data['ndt'] = gen_ndt
    data['sp_trial_var'] = gen_sp_trial_var
    data['drift_scaling'] = gen_drift_scaling
    data['drift_trial_var'] = gen_drift_trial_var

    if nonlinear_mapping:
        data['cor_drift'] = (gen_drift_scaling + gen_drift_asym * (data['Q_mean_t'] - data['Q_min'])) / (
                1 + np.exp(gen_slop * (data['Q_mean_t'] + data['Q_max_t'] - 2 * data['Q_cor'])))
        data['inc_drift'] = (gen_drift_scaling + gen_drift_asym * (data['Q_mean_t'] - data['Q_min'])) / (
                1 + np.exp(gen_slop * (data['Q_mean_t'] + data['Q_max_t'] - 2 * data['Q_inc'])))
    else:
        data['cor_drift'] = gen_drift_scaling * (data['Q_cor'])
        data['inc_drift'] = gen_drift_scaling * (data['Q_inc'])

    rt, acc = random_lba_2A(data['cor_drift'],
                            data['inc_drift'],
                            data['sp_trial_var'], 
                            data['ndt'], 
                            data['k'], 
                            data['drift_trial_var'], **kwargs)

    data['rt'] = rt
    data['accuracy'] = acc

    data = data.set_index(['participant', 'block_label', 'trial_block'])

    return data


def simulate_hier_rlalba(task_design,
                         gen_mu_alpha, gen_sd_alpha,
                         gen_mu_sp_trial_var, gen_sd_sp_trial_var,
                         gen_mu_ndt, gen_sd_ndt,
                         gen_mu_k, gen_sd_k,
                         gen_mu_drift_trial_var, gen_sd_drift_trial_var,
                         gen_mu_drift_scaling, gen_sd_drift_scaling,
                         gen_mu_slop=None, gen_sd_slop=None,
                         gen_mu_drift_asym=None, gen_sd_drift_asym=None,
                         nonlinear_mapping=False,
                         initial_value_learning=0,
                         **kwargs):
    """Simulate behavior (rt and accuracy) according to a hierarchical RL-ALBA model.

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

    gen_mu_sp_trial_var : float
        Group-mean of sp_trial_var parameter.

    gen_sd_sp_trial_var : float
        Group-standard deviation of the sp_trial_var parameter.

    gen_mu_ndt : float
        Group-mean of non-decision time parameter.

    gen_sd_ndt : float
        Group-standard deviation of the non-decision time parameter.

    gen_mu_k : float
        Group-mean of of the distance between starting point variability and threshold.

    gen_sd_k : float
        Group-standard deviation of the distance between starting point variability and threshold.

    gen_mu_drift_trial_var : float
        Group-mean of across trial variability in the drift-rate.
        Should be positive.

    gen_sd_drift_trial_var : float
        Group-standard deviation of across trial variability in the drift-rate.
        Should be positive.

    gen_mu_slop : float
        Group-mean of the slop of the drift-rate.

    gen_sd_slop : float
        Group-standard deviation of the slop of the drift-reate.

    gen_mu_drift_asym : float
        Group-mean of the drift asym of the drift-rate.

    gen_sd_drift_asym : float
        Group-standard deviation of the drift asym of the drift-rate.

    nonlinear_mapping : bool, default False
        Whether to use the nonlinear mapping mechanism.

    initial_value_learning : float, default 0
        The initial value for Q learning.

    Other parameters
    ----------------

    **kwargs : dict
        Additional parameters to be passed to `random_lba_2A`.

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, with n_trials*n_participants rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters (at the participant level).

    """
    data = task_design.copy()
    participants = pd.unique(data["participant"])
    n_participants = len(participants)

    if n_participants < 2:
        raise ValueError("You only have one participant. Use simulate_rl_2A instead.")

    if type(gen_mu_alpha) != type(gen_sd_alpha):
        raise TypeError("gen_mu_alpha and gen_sd_alpha should be of the same type.")

    if nonlinear_mapping:
        if gen_mu_slop == None or gen_sd_slop == None or gen_mu_drift_asym == None or gen_sd_drift_asym == None:
            raise ValueError("The gen_mu_slop and gen_mu_drift_asym can not be None with nonlinear_mapping mechanism! ")

    parameters_dic = {'drift_scaling': np.log(1 + np.exp(np.random.normal(gen_mu_drift_scaling, 
                                                                          gen_sd_drift_scaling, 
                                                                          n_participants))),
                       'sp_trial_var': np.log(1 + np.exp(np.random.normal(gen_mu_sp_trial_var, 
                                                                          gen_sd_sp_trial_var, 
                                                                          n_participants))),
                       'ndt': np.log(1 + np.exp(np.random.normal(gen_mu_ndt, gen_sd_ndt, n_participants))),
                       'k': np.log(1 + np.exp(np.random.normal(gen_mu_k, gen_sd_k, n_participants))),
                       'drift_trial_var': np.log(1+np.exp(np.random.normal(gen_mu_drift_trial_var,
                                                                           gen_sd_drift_trial_var,
                                                                           n_participants)))}
    
    if nonlinear_mapping:
        parameters_dic['drift_asym'] = np.log(1 + np.exp(np.random.normal(gen_mu_drift_asym, 
                                                                          gen_sd_drift_asym, 
                                                                          n_participants)))
        parameters_dic['slop'] = np.log(1 + np.exp(np.random.normal(gen_mu_slop, 
                                                                    gen_sd_slop, 
                                                                    n_participants)))
                                                                    
    if (type(gen_mu_alpha) == float) | (type(gen_mu_alpha) == int):
        parameters_dic['alpha'] = stats.norm.cdf(np.random.normal(gen_mu_alpha, gen_sd_alpha, n_participants))
        parameters = pd.DataFrame(parameters_dic, index=participants)

        data = pd.concat([data.set_index('participant'),
                          parameters], axis=1, ignore_index=False).reset_index().rename(
            columns={'index': 'participant'})
        data = pd.concat([data, _simulate_delta_rule_2A(task_design,
                                                        parameters.alpha.values,
                                                        initial_value_learning)], axis=1)

    elif type(gen_mu_alpha) is list:
        if len(gen_mu_alpha) != len(gen_sd_alpha):
            raise ValueError("gen_mu_alpha and gen_sd_alpha should be of the same length.")
        if len(gen_mu_alpha) == 2:
            parameters_dic['alpha_pos'] = stats.norm.cdf(np.random.normal(gen_mu_alpha[0], gen_sd_alpha[0], n_participants))
            parameters_dic['alpha_neg'] = stats.norm.cdf(np.random.normal(gen_mu_alpha[1], gen_sd_alpha[1], n_participants))
            
            parameters = pd.DataFrame(parameters_dic, index=participants)
            data = pd.concat([data.set_index('participant'), parameters], axis=1,
                             ignore_index=False).reset_index().rename(columns={'index': 'participant'})
            data = pd.concat([data, _simulate_delta_rule_2A(task_design=task_design,
                                                            alpha=None,
                                                            initial_value_learning=initial_value_learning,
                                                            alpha_pos=parameters.alpha_pos.values,
                                                            alpha_neg=parameters.alpha_neg.values)], axis=1)

        elif len(gen_mu_alpha) == 3:
            pass  # implement here Stefano's learning rule
        else:
            raise ValueError("The gen_mu_alpha list should be of either length 2 or 3.")
    else:
        raise TypeError("The gen_alpha should be either a list or a float/int.")

    if nonlinear_mapping:
        data['cor_drift'] = (data['drift_scaling'] + data['drift_asym'] * (data['Q_mean_t'] - data['Q_min'])) / (
                1 + np.exp(data['slop'] * (data['Q_mean_t'] + data['Q_max_t'] - 2 * data['Q_cor'])))
        data['inc_drift'] = (data['drift_scaling'] + data['drift_asym'] * (data['Q_mean_t'] - data['Q_min'])) / (
                1 + np.exp(data['slop'] * (data['Q_mean_t'] + data['Q_max_t'] - 2 * data['Q_inc'])))
    else:
        data['cor_drift'] = data['drift_scaling'] * (data['Q_cor'])
        data['inc_drift'] = data['drift_scaling'] * (data['Q_inc'])

    rt, acc = random_lba_2A(data['cor_drift'], 
                            data['inc_drift'], 
                            data['sp_trial_var'],
                            data['ndt'], 
                            data['k'],
                            data['drift_trial_var'], **kwargs)

    data['rt'] = rt
    data['accuracy'] = acc

    data = data.set_index(['participant', 'block_label', 'trial_block'])

    return data
