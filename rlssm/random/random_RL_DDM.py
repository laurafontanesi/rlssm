import numpy as np
import pandas as pd
from scipy import stats

# RL_2A + DDM
from rlssm.random.random_DDM import random_ddm
from rlssm.random.random_RL import _simulate_delta_rule_2A


def simulate_rlddm_2A(task_design,
                      gen_alpha,
                      gen_drift_scaling,
                      gen_threshold,
                      gen_ndt,
                      gen_drift_asymptote=None,
                      gen_threshold_modulation=None,
                      initial_value_learning=0,
                      **kwargs):
    """Simulates behavior (rt and accuracy) according to a RLDDM model,

    Parameters
    ----------

    task_design : pandas.DataFrame
        A pandas DataFrame containing the task design.

    gen_alpha : float, listx
        The learning rate parameter.

    gen_drift_scaling : float, listx
        The drift scaling parameter.

    gen_threshold : float, listx
        The threshold parameter.

    gen_ndt : float, listx
        The non-decision time parameter.

    gen_drift_asymptote : float, listx, default: None
        The drift asymptote parameter.

    gen_threshold_modulation : float, listx, default: None
        The threshold modulation parameter.

    initial_value_learning : float, default: 0
        The initial value of the learning parameter.

    Other Parameters
    ----------------

    **kwargs : dict
        Additional keyword arguments to be further passed.

    Returns
    -------

    data : pandas.DataFrame

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
    data['ndt'] = gen_ndt

    if gen_threshold_modulation == None:
        data['threshold'] = gen_threshold
    else:
        Q_mean = (data['Q_cor'] + data['Q_inc'])/2
        data['threshold_fix'] = gen_threshold
        data['threshold_modulation'] = gen_threshold_modulation
        data['threshold'] = np.log(1 + np.exp(gen_threshold + gen_threshold_modulation*Q_mean))
        
    
    if gen_drift_asymptote == None:
        data['drift'] = gen_drift_scaling * (data['Q_cor'] - data['Q_inc'])
    else:
        data['drift_asymptote'] = gen_drift_asymptote
        z = gen_drift_scaling * (data['Q_cor'] - data['Q_inc'])
        data['drift'] = (gen_drift_asymptote)/(1+np.exp(-z)) - gen_drift_asymptote/2

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
    """Simulates behavior (rt and accuracy) according to a RLDDM hierarchical model.

    Parameters
    ----------

    task_design : pandas.DataFrame
        A pandas DataFrame containing the task design.

    gen_mu_alpha : list, float, int
        The mean of the learning rate parameter.

    gen_sd_alpha : list, float, int
        The standard deviation of the learning rate parameter.

    gen_mu_drift_scaling : list, float, int
        The mean of the drift scaling parameter.

    gen_sd_drift_scaling : list, float, int
        The standard deviation of the drift scaling parameter.

    gen_mu_threshold : list, float, int
        The mean of the threshold parameter.

    gen_sd_threshold : list, float, int
        The standard deviation of the threshold parameter.

    gen_mu_ndt : list, float, int
        The mean of the non-decision time parameter.

    gen_sd_ndt : list, float, int
        The standard deviation of the non-decision time parameter.

    initial_value_learning : float, default: 0
        The initial value of the learning parameter.

    Other Parameters
    ----------------

    **kwargs : dict
        Additional keyword arguments to be further passed.

    Returns
    -------

    data : pandas.DataFrame

    """
    data = task_design.copy()
    participants = pd.unique(data["participant"])
    n_participants = len(participants)
    if n_participants < 2:
        raise ValueError("You only have one participant. Use simulate_rl_2A instead.")

    if type(gen_mu_alpha) != type(gen_sd_alpha):
        raise TypeError("gen_mu_alpha and gen_sd_alpha should be of the same type.")

    if (type(gen_mu_alpha) == float) | (type(gen_mu_alpha) == int):
        parameters = pd.DataFrame(
            {'alpha': stats.norm.cdf(np.random.normal(gen_mu_alpha, gen_sd_alpha, n_participants)),
             'drift_scaling': np.log(
                 1 + np.exp(np.random.normal(gen_mu_drift_scaling, gen_sd_drift_scaling, n_participants))),
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
            raise ValueError("gen_mu_alpha and gen_sd_alpha should be of the same length.")
        if len(gen_mu_alpha) == 2:
            parameters = pd.DataFrame(
                {'alpha_pos': stats.norm.cdf(np.random.normal(gen_mu_alpha[0], gen_sd_alpha[0], n_participants)),
                 'alpha_neg': stats.norm.cdf(np.random.normal(gen_mu_alpha[1], gen_sd_alpha[1], n_participants)),
                 'drift_scaling': np.log(
                     1 + np.exp(np.random.normal(gen_mu_drift_scaling, gen_sd_drift_scaling, n_participants))),
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

    data['drift'] = data['drift_scaling'] * (data['Q_cor'] - data['Q_inc'])

    # simulate responses
    rt, acc = random_ddm(data['drift'], data['threshold'], data['ndt'], .5, **kwargs)
    data['rt'] = rt
    data['accuracy'] = acc

    data = data.set_index(['participant', 'block_label', 'trial_block'])
    return data
