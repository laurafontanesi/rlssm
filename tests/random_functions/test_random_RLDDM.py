import os
import pandas as pd

from rlssm.random.random_RL_DDM import simulate_hier_rlddm_2A, simulate_rlddm_2A
from rlssm.random.random_common import generate_task_design_fontanesi


def test_random_RLDDM(print_results=True):
    dm_non_hier = generate_task_design_fontanesi(n_trials_block=80,
                                                 n_blocks=3,
                                                 n_participants=30,
                                                 trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                 mean_options=[34, 38, 50, 54],
                                                 sd_options=[5, 5, 5, 5])

    dm_2__non_hier_alpha = generate_task_design_fontanesi(n_trials_block=80,
                                                          n_blocks=3,
                                                          n_participants=1,
                                                          trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                          mean_options=[34, 38, 50, 54],
                                                          sd_options=[5, 5, 5, 5])

    data_non_hier = simulate_hier_rlddm_2A(task_design=dm_non_hier,
                                           gen_mu_alpha=-.5,
                                           gen_sd_alpha=.1,
                                           gen_mu_drift_scaling=.1,
                                           gen_sd_drift_scaling=.5,
                                           gen_mu_threshold=1,
                                           gen_sd_threshold=.1,
                                           gen_mu_ndt=.23,
                                           gen_sd_ndt=.05,
                                           initial_value_learning=20)

    data_non_hier_2alpha = simulate_rlddm_2A(task_design=dm_2__non_hier_alpha,
                                             gen_alpha=[.1, .01],
                                             gen_drift_scaling=.1,
                                             gen_threshold=1,
                                             gen_ndt=.23,
                                             initial_value_learning=0)

    data_hier_2alpha = simulate_hier_rlddm_2A(task_design=dm_non_hier,
                                              gen_mu_alpha=[-.5, -1],
                                              gen_sd_alpha=[.1, .1],
                                              gen_mu_drift_scaling=.1,
                                              gen_sd_drift_scaling=.5,
                                              gen_mu_threshold=1,
                                              gen_sd_threshold=.1,
                                              gen_mu_ndt=.23,
                                              gen_sd_ndt=.05,
                                              initial_value_learning=20)

    # TEST: assure that there are 30 participants
    assert data_non_hier.index[-1][0] == 30, f"Number of participants should be 30"
    assert data_non_hier_2alpha.index[-1][0] == 1, f"Number of participants should be 1"
    assert data_hier_2alpha.index[-1][0] == 30, f"Number of participants should be 30"

    # Test data produced against reference data; test to be created
    reference_path = os.path.join(os.path.dirname(__file__), 'reference_data', 'RL_DDM_2A.csv')
    reference_path = os.path.join(os.path.dirname(__file__), 'reference_data', 'RL_DDM_2_alpha.csv')
    reference_path = os.path.join(os.path.dirname(__file__), 'reference_data', 'RL_DDM_hier_2alpha.csv')
