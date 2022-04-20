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
