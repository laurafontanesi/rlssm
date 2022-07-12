import os
import unittest

from rlssm.random.random_RL_DDM import simulate_hier_rlddm_2A, simulate_rlddm_2A
from rlssm.random.random_common import generate_task_design_fontanesi


class TestRandomRLDDM(unittest.TestCase):
    def setUp(self):
        self.dm_non_hier = generate_task_design_fontanesi(n_trials_block=80,
                                                          n_blocks=3,
                                                          n_participants=30,
                                                          trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                          mean_options=[34, 38, 50, 54],
                                                          sd_options=[5, 5, 5, 5])

        self.dm_2_non_hier_alpha = generate_task_design_fontanesi(n_trials_block=80,
                                                                  n_blocks=3,
                                                                  n_participants=1,
                                                                  trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                                  mean_options=[34, 38, 50, 54],
                                                                  sd_options=[5, 5, 5, 5])

        self.data_non_hier = simulate_hier_rlddm_2A(task_design=self.dm_non_hier,
                                                    gen_mu_alpha=-.5,
                                                    gen_sd_alpha=.1,
                                                    gen_mu_drift_scaling=.1,
                                                    gen_sd_drift_scaling=.5,
                                                    gen_mu_threshold=1,
                                                    gen_sd_threshold=.1,
                                                    gen_mu_ndt=.23,
                                                    gen_sd_ndt=.05,
                                                    initial_value_learning=20)

        self.data_non_hier_2alpha = simulate_rlddm_2A(task_design=self.dm_2_non_hier_alpha,
                                                      gen_alpha=[.1, .01],
                                                      gen_drift_scaling=.1,
                                                      gen_threshold=1,
                                                      gen_ndt=.23,
                                                      initial_value_learning=0)

        self.data_hier_2alpha = simulate_hier_rlddm_2A(task_design=self.dm_non_hier,
                                                       gen_mu_alpha=[-.5, -1],
                                                       gen_sd_alpha=[.1, .1],
                                                       gen_mu_drift_scaling=.1,
                                                       gen_sd_drift_scaling=.5,
                                                       gen_mu_threshold=1,
                                                       gen_sd_threshold=.1,
                                                       gen_mu_ndt=.23,
                                                       gen_sd_ndt=.05,
                                                       initial_value_learning=20)

    def test_random_RLDDM_test1(self):
        # TEST: assure that there are 30 participants
        assert self.data_non_hier.index[-1][0] == 30, f"Number of participants should be 30"
        assert self.data_non_hier_2alpha.index[-1][0] == 1, f"Number of participants should be 1"

    def test_random_RLDDM_hier(self):
        assert self.data_hier_2alpha.index[-1][0] == 30, f"Number of participants should be 30"
