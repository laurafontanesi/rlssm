import os
import pandas as pd

from rlssm.random.random_common import generate_task_design_fontanesi


# HIER DATA, random 2 alternatives
def test_generate_dm_hier(print_results=True):
    print("Test - DM simulate hier data, 2 alternatives per trial (4 per block)")

    dm = generate_task_design_fontanesi(n_trials_block=80,
                                        n_blocks=3,
                                        n_participants=1,
                                        trial_types=['1-2', '1-3', '2-4', '3-4'],
                                        mean_options=[34, 38, 50, 54],
                                        sd_options=[5, 5, 5, 5])

    dm2 = generate_task_design_fontanesi(n_trials_block=100,
                                         n_blocks=20,
                                         n_participants=1,
                                         trial_types=['1-2', '1-3', '2-4', '3-4'],
                                         mean_options=[34, 38, 50, 54],
                                         sd_options=[5, 5, 5, 5])

    # dm3 = generate_task_design_fontanesi(n_trials_block=15,
    #                                      n_blocks=20,
    #                                      n_participants=1,
    #                                      trial_types=['1-2', '1-3', '2-4', '3-4'],
    #                                      mean_options=[34, 38, 50, 54],
    #                                      sd_options=[5, 5, 5, 5])

    # TEST changing n_trials_block = 100, n_blocks = 20 in dm and it still works;
    # Now see in the data that trial_block=100 and 20

    # TEST: dm3 has 15 trial blocks; check if we really need it to be a multiple of 4, in this case

    # Test data produced against reference data
    reference_path = os.path.join(os.path.dirname(__file__), 'reference_data', 'generate_dm_hier.csv')
    # dm.to_csv(reference_path)
    reference_data = pd.read_csv(reference_path, index_col=0)
    # assert dm.equals(reference_data)

    if print_results:
        print(dm)
