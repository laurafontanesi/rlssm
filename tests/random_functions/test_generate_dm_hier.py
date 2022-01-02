# HIER DATA, random2 alternatives
from rlssm.random.random_common import generate_task_design_fontanesi


def test_generate_dm_hier(print_results=True):
    print("Test - DM simulate hier data, 2 alternatives per trial (4 per block)")

    dm = generate_task_design_fontanesi(n_trials_block=80,
                                        n_blocks=3,
                                        n_participants=30,
                                        trial_types=['1-2', '1-3', '2-4', '3-4'],
                                        mean_options=[34, 38, 50, 54],
                                        sd_options=[5, 5, 5, 5])

    if print_results:
        print(dm)
