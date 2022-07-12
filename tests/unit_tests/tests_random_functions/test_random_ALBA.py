import unittest

import numpy as np

from rlssm.random.random_ALBA import simulate_alba_2A, simulate_hier_alba


class TestRandomALBA(unittest.TestCase):
    def setUp(self):
        self.data1 = simulate_alba_2A(gen_S_cor=np.random.normal(.4, 0.01, 100),
                                      gen_S_inc=np.random.normal(.3, 0.01, 100),
                                      gen_sp_trial_var=2,
                                      gen_ndt=.2,
                                      gen_k=.2,
                                      gen_v0=1,
                                      gen_ws=.7,
                                      gen_wd=1,
                                      gen_drift_trial_sd=.1)

    def test_random_ALBA_test1(self):
        # TEST: assure there is only 1 participant
        assert self.data1.index[-1][0] == 1, f"Number of participants should be 1"

    def test_random_ALBA_hier(self):
        # TEST hierarchical version
        self.data_hier = simulate_hier_alba(n_trials=100, n_participants=30, gen_v0=1,
                                            gen_ws=.7, gen_wd=1,
                                            gen_mu_drift_cor=.4, gen_sd_drift_cor=0.01,
                                            gen_mu_drift_inc=.3, gen_sd_drift_inc=0.01,
                                            gen_mu_sp_trial_var=1, gen_sd_sp_trial_var=.1,
                                            gen_mu_ndt=.23, gen_sd_ndt=.1,
                                            gen_mu_k=.5, gen_sd_k=None,
                                            gen_drift_trial_sd=None)
        # TEST: assure that there are 30 participants
        assert self.data_hier.index[-1][0] == 30, f"Number of participants should be 30"
