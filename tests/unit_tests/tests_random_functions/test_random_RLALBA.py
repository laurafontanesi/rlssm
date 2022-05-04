import unittest


class TestRandomRLALBA(unittest.TestCase):
    def setUp(self):
        self.data1 = simulate_RLALBA_2A(n_trials=1000,
                                        gen_cor_drift=.6,
                                        gen_inc_drift=.4,
                                        gen_threshold=1.5,
                                        gen_ndt=.23,
                                        gen_rel_sp=.8,
                                        gen_drift_trial_sd=.5)

    def test_random_RLALBA_test1(self):
        # TEST: assure there is only 1 participant
        assert self.data1.index[-1][0] == 1, f"Number of participants should be 1"

    def test_random_RLALBA_hier(self):
        # TEST hierarchical version
        self.data_hier = simulate_hier_RLALBA(n_trials=100, n_participants=30,
                                              gen_mu_drift_cor=1, gen_sd_drift_cor=.5,
                                              gen_mu_drift_inc=1, gen_sd_drift_inc=.5,
                                              gen_mu_threshold=1, gen_sd_threshold=.1,
                                              gen_mu_ndt=.23, gen_sd_ndt=.1,
                                              gen_mu_rel_sp=.1, gen_sd_rel_sp=.05)

        # TEST: assure that there are 30 participants
        assert self.data_hier.index[-1][0] == 30, f"Number of participants should be 30"
