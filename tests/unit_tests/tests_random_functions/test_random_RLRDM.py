import unittest

from rlssm.random.random_RL_RDM import simulate_rlrdm_2A, simulate_hier_rlrdm_2A
from rlssm.random.random_common import generate_task_design_fontanesi


class TestRandomRLRDM(unittest.TestCase):
    def setUp(self):
        self.dm = generate_task_design_fontanesi(n_trials_block=80,
                                                 n_blocks=3,
                                                 n_participants=1,
                                                 trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                 mean_options=[34, 38, 50, 54],
                                                 sd_options=[5, 5, 5, 5])

    def test_random_RLRDM(self):
        self.data_non_hier = simulate_rlrdm_2A(task_design=self.dm,
                                               gen_alpha=0.1,
                                               gen_drift_scaling=.1,
                                               gen_threshold=1,
                                               gen_ndt=.23,
                                               initial_value_learning=0)
        
        # TEST: assure there is only 1 participant
        assert self.data_non_hier.index[-1][0] == 1, f"Number of participants should be 1"

    def test_random_RLRDM_hier(self):
        self.data_hier_2alpha = simulate_hier_rlrdm_2A(task_design=self.dm,
                                                       gen_mu_alpha=[-.5, -1],
                                                       gen_sd_alpha=[.1, .1],
                                                       gen_mu_drift_scaling=.1,
                                                       gen_sd_drift_scaling=.5,
                                                       gen_mu_threshold=1,
                                                       gen_sd_threshold=.1,
                                                       gen_mu_ndt=.23,
                                                       gen_sd_ndt=.05,
                                                       initial_value_learning=20)

        assert self.data_hier_2alpha.index[-1][0] == 30, f"Number of participants should be 30"
