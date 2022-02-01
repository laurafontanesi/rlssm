import numpy as np

from rlssm.random.random_RL import simulate_rl_2A, simulate_hier_rl_2A
from rlssm.random.random_common import generate_task_design_fontanesi


# Reinforcement learning data
def test_random_RL(print_results=True):
    dm_non_hier = generate_task_design_fontanesi(n_trials_block=80,
                                                 n_blocks=3,
                                                 n_participants=1,
                                                 trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                 mean_options=[34, 38, 50, 54],
                                                 sd_options=[5, 5, 5, 5])

    data1 = simulate_rl_2A(task_design=dm_non_hier,
                           gen_alpha=.1,
                           gen_sensitivity=.5,
                           initial_value_learning=20)

    data2 = simulate_rl_2A(task_design=dm_non_hier,
                           gen_alpha=.1,
                           gen_sensitivity=.01,
                           initial_value_learning=20)

    data3 = simulate_rl_2A(task_design=dm_non_hier,
                           gen_alpha=.9,
                           gen_sensitivity=.5,
                           initial_value_learning=20)

    data4 = simulate_rl_2A(task_design=dm_non_hier,
                           gen_alpha=.1,
                           gen_sensitivity=.5,
                           initial_value_learning=0)

    data_non_hier_2alpha = simulate_rl_2A(task_design=dm_non_hier,
                                          gen_alpha=[.6, .02],
                                          gen_sensitivity=.5,
                                          initial_value_learning=44)

    dm_hier = generate_task_design_fontanesi(n_trials_block=80,
                                             n_blocks=3,
                                             n_participants=30,
                                             trial_types=['1-2', '1-3', '2-4', '3-4'],
                                             mean_options=[34, 38, 50, 54],
                                             sd_options=[5, 5, 5, 5])

    data_hier = simulate_hier_rl_2A(task_design=dm_hier,
                                    gen_mu_alpha=-.5,
                                    gen_sd_alpha=.1,
                                    gen_mu_sensitivity=.5,
                                    gen_sd_sensitivity=.1,
                                    initial_value_learning=20)

    data_hier_2alpha = simulate_hier_rl_2A(task_design=dm_hier,
                                           gen_mu_alpha=[-.5, -1],
                                           gen_sd_alpha=[.1, .1],
                                           gen_mu_sensitivity=.5,
                                           gen_sd_sensitivity=.1,
                                           initial_value_learning=20)

    # TEST: assure the correct number of participants
    assert data1.index[-1][0] == 1, f"Number of participants should be 1"
    assert data2.index[-1][0] == 1, f"Number of participants should be 1"
    assert data3.index[-1][0] == 1, f"Number of participants should be 1"
    assert data4.index[-1][0] == 1, f"Number of participants should be 1"
    assert data_non_hier_2alpha.index[-1][0] == 1, f"Number of participants should be 1"
    assert data_hier_2alpha.index[-1][0] == 30, f"Number of participants should be 30"
    assert data_hier.index[-1][0] == 30, f"Number of participants should be 30"

    # TEST non-hier 1alpha: mean accuracy(data1) > accuracy(data2)
    assert np.mean(data1['accuracy']) > np.mean(
        data2['accuracy']), f"Accuracy(data1) is not better than accuracy(data2)"

    # TEST non-hier 1alpha: check mean p_cor(data1) > p_cor(data2)
    assert np.mean(data1['p_cor']) > np.mean(
        data2['p_cor']), f"Mean p_cor(data1) is not better than mean p_cor(data2)"

    # TEST non-hier 1alpha: variance p_cor(data3) > p_cor(data1) - tests learning rate
    assert np.var(data3['p_cor']) > np.var(data1['p_cor']), \
        f"variance p_cor(data3) is not better than variance p_cor(data1)"

    # TEST non-hier 1alpha: check initial values for data1 of Q_cor,Q_inc is 20 (row 0)
    assert data1['Q_cor'].iloc[0] == 20, f"initial Q_cor for data1 should be 20 but it is {data4['Q_cor'].iloc(0)}"
    assert data1['Q_inc'].iloc[0] == 20, f"initial Q_inc for data1 should be 20 but it is {data4['Q_inc'].iloc(0)}"

    # TEST non-hier 1alpha: check initial values for data4 of Q_cor,Q_inc is 0 (row 0)
    assert data4['Q_cor'].iloc[0] == 0, f"initial Q_cor for data4 should be 0 but it is {data4['Q_cor'].iloc(0)}"
    assert data4['Q_inc'].iloc[0] == 0, f"initial Q_inc for data4 should be 0 but it is {data4['Q_inc'].iloc(0)}"

    # TEST non-hier 2alpha: variance of q_cor > q_inc
    assert np.var(data_non_hier_2alpha['Q_cor']) > np.var(data_non_hier_2alpha['Q_inc']), \
        f"variance Q_cor(data) is not better than variance Q_inc(data)"
