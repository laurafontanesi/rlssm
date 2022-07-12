import unittest

from rlssm.random.random_RL_ALBA import simulate_rlalba_2A, simulate_hier_rlalba
from rlssm.random.random_common import generate_task_design_fontanesi


class TestRandomRLALBA(unittest.TestCase):
    def setUp(self):
        self.dm_non_hier = generate_task_design_fontanesi(n_trials_block=80,
                                                          n_blocks=3,
                                                          n_participants=1,
                                                          trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                          mean_options=[34, 38, 50, 54],
                                                          sd_options=[5, 5, 5, 5])

        self.dm_hier = generate_task_design_fontanesi(n_trials_block=80,
                                                      n_blocks=3,
                                                      n_participants=30,
                                                      trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                      mean_options=[34, 38, 50, 54],
                                                      sd_options=[5, 5, 5, 5])

        self.data_non_hier = simulate_rlalba_2A(task_design=self.dm_non_hier,
                                                gen_alpha=.1,
                                                gen_sp_trial_var=.2,
                                                gen_ndt=.2,
                                                gen_k=3,
                                                gen_v0=2,
                                                gen_ws=.01,
                                                gen_wd=.05,
                                                gen_drift_trial_sd=None)

        self.data_hier = simulate_hier_rlalba(task_design=self.dm_hier,
                                              gen_mu_alpha=[-.5, -1], gen_sd_alpha=[.1, .1],
                                              gen_mu_sp_trial_var=2, gen_sd_sp_trial_var=.1,
                                              gen_mu_ndt=.23, gen_sd_ndt=.1,
                                              gen_mu_k=2, gen_sd_k=.1,
                                              gen_mu_v0=3, gen_sd_v0=.1,
                                              gen_mu_ws=-4, gen_sd_ws=.01,
                                              gen_mu_wd=-2, gen_sd_wd=.01,
                                              initial_value_learning=0,
                                              gen_drift_trial_sd=None)

    def test_random_RLALBA_test1(self):
        # TEST: assure there is only 1 participant
        assert self.data_non_hier.index[-1][0] == 1, f"Number of participants should be 1"

    def test_random_RLALBA_hier(self):
        # TEST hierarchical version
        # TEST: assure that there are 30 participants
        assert self.data_hier.index[-1][0] == 30, f"Number of participants should be 30"
