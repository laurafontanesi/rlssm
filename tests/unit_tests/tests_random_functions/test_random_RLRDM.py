import unittest

from rlssm.random.random_RL_RDM import simulate_rlrdm_2A
from rlssm.random.random_common import generate_task_design_fontanesi


class TestRandomRLRDM(unittest.TestCase):
    def setUp(self):
        self.dm = generate_task_design_fontanesi(n_trials_block=80,
                                                 n_blocks=3,
                                                 n_participants=1,
                                                 trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                 mean_options=[34, 38, 50, 54],
                                                 sd_options=[5, 5, 5, 5])

        self.data = simulate_rlrdm_2A(task_design=self.dm,
                                      gen_alpha=0.1,
                                      gen_drift_scaling=.1,
                                      gen_threshold=1,
                                      gen_ndt=.23,
                                      initial_value_learning=0)

    def test_random_RLRDM(self):
        # TEST: assure there is only 1 participant
        assert self.data.index[-1][0] == 1, f"Number of participants should be 1"
