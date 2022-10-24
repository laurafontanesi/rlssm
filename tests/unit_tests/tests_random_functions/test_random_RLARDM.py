import unittest
from rlssm.random.random_RL_ARDM import simulate_rlardm_2A, simulate_hier_rlardm
from rlssm.random.random_common import generate_task_design_fontanesi


class TestRandomRLARDM(unittest.TestCase):
    def test_random_RLARDM_test1(self):
        self.dm_non_hier = generate_task_design_fontanesi(n_trials_block=80,
                                                          n_blocks=3,
                                                          n_participants=1,
                                                          trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                          mean_options=[34, 38, 50, 54],
                                                          sd_options=[5, 5, 5, 5])

        self.data_non_hier = simulate_rlardm_2A(task_design=self.dm_non_hier,
                                                gen_alpha=0.1,
                                                gen_threshold=3,
                                                gen_ndt=.2,
                                                gen_v0=2,
                                                gen_ws=.01,
                                                gen_wd=.05,
                                                gen_drift_trial_sd=None)

        # TEST: assure there is only 1 participant
        assert self.data_non_hier.index[-1][0] == 1, f"Number of participants should be 1"

    def test_random_RLARDM_hier(self):
        # TEST hierarchical version
        self.dm_hier = generate_task_design_fontanesi(n_trials_block=80,
                                                      n_blocks=3,
                                                      n_participants=30,
                                                      trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                      mean_options=[34, 38, 50, 54],
                                                      sd_options=[5, 5, 5, 5])

        self.data_hier = simulate_hier_rlardm(task_design=self.dm_hier,
                                              gen_mu_alpha=[-.5, -1], gen_sd_alpha=[.1, .1],
                                              gen_mu_threshold=2, gen_sd_threshold=.1,
                                              gen_mu_ndt=.23, gen_sd_ndt=.1,
                                              gen_mu_v0=3, gen_sd_v0=.1,
                                              gen_mu_ws=-4, gen_sd_ws=.01,
                                              gen_mu_wd=-2, gen_sd_wd=.01,
                                              initial_value_learning=0,
                                              gen_drift_trial_sd=None)
        # TEST: assure that there are 30 participants
        assert self.data_hier.index[-1][0] == 30, f"Number of participants should be 30"
