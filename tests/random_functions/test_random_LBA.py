from rlssm import LBAModel_2A
from rlssm.random.random_LBA import simulate_lba_2A
from rlssm.random.random_common import generate_task_design_fontanesi


def test_random_LBA(print_results=True):
    dm_non_hier = generate_task_design_fontanesi(n_trials_block=80,
                                                 n_blocks=3,
                                                 n_participants=1,
                                                 trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                 mean_options=[34, 38, 50, 54],
                                                 sd_options=[5, 5, 5, 5])

    model = LBAModel_2A(hierarchical_levels=1)

    # function to be created
    # data_hier = simulate_lba_2A(task_design=dm_non_hier,
    #                             gen_mu_alpha=-.5,
    #                             gen_sd_alpha=.1,
    #                             gen_mu_sensitivity=.5,
    #                             gen_sd_sensitivity=.1,
    #                             initial_value_learning=20)
