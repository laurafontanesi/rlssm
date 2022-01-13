import os
import pandas as pd

from rlssm.random.random_RL import simulate_hier_rl_2A
from rlssm.random.random_common import generate_task_design_fontanesi


# Reinforcement learning data
def test_hier_RL_2A(print_results=True):
    print("Test - simple RL_2A; simulate_hier_rl_2A")

    dm = generate_task_design_fontanesi(n_trials_block=80,
                                        n_blocks=3,
                                        n_participants=30,
                                        trial_types=['1-2', '1-3', '2-4', '3-4'],
                                        mean_options=[34, 38, 50, 54],
                                        sd_options=[5, 5, 5, 5])

    data = simulate_hier_rl_2A(task_design=dm,
                               gen_mu_alpha=-.5,
                               gen_sd_alpha=.1,
                               gen_mu_sensitivity=.5,
                               gen_sd_sensitivity=.1,
                               initial_value_learning=20)

    # TEST: assure that there are 30 participants
    assert data.index[-1][0] == 30, f"Number of participants should be 30"

    # Test data produced against reference data
    reference_path = os.path.join(os.path.dirname(__file__), 'reference_data', 'hier_RL_2A.csv')
    # data.to_csv(reference_path)
    reference_data = pd.read_csv(reference_path, index_col=0)
    # assert data.equals(reference_data)

    if print_results:
        print(data)
