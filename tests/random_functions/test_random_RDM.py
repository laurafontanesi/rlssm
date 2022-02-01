from rlssm import RDModel_2A
from rlssm.random.random_common import generate_task_design_fontanesi


def test_random_RDM(print_results=True):
    dm_non_hier = generate_task_design_fontanesi(n_trials_block=80,
                                                 n_blocks=3,
                                                 n_participants=1,
                                                 trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                 mean_options=[34, 38, 50, 54],
                                                 sd_options=[5, 5, 5, 5])

    model = RDModel_2A(hierarchical_levels=1)

    # function to be created
    # simulate_rdm_2A(n_trials,
    #                 gen_cor_drift,
    #                 gen_inc_drift,
    #                 gen_threshold,
    #                 gen_ndt,
    #                 gen_rel_sp=.5,
    #                 participant_label=1,
    #                 gen_drift_trialsd=None,
    #                 gen_rel_sp_trialsd=None)

