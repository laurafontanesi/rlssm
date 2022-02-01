import os
import pandas as pd

from rlssm.random.random_RL_RDM import simulate_rlrdm_2A
from rlssm.random.random_common import generate_task_design_fontanesi


def test_random_RLRDM(print_results=True):
    dm = generate_task_design_fontanesi(n_trials_block=80,
                                        n_blocks=3,
                                        n_participants=1,
                                        trial_types=['1-2', '1-3', '2-4', '3-4'],
                                        mean_options=[34, 38, 50, 54],
                                        sd_options=[5, 5, 5, 5])

    data = simulate_rlrdm_2A(task_design=dm,
                             gen_alpha=0.1,
                             gen_drift_scaling=.1,
                             gen_threshold=1,
                             gen_ndt=.23,
                             initial_value_learning=0)

    # TEST: assure there is only 1 participant
    assert data.index[-1][0] == 1, f"Number of participants should be 1"

    # Test data produced against reference data
    reference_path = os.path.join(os.path.dirname(__file__), 'reference_data', 'RL_RDM_non_hier.csv')
