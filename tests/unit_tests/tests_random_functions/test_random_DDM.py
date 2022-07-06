import unittest

import numpy as np

from rlssm.random.random_DDM import simulate_ddm, simulate_hier_ddm


class TestRandomDDM(unittest.TestCase):
    def setUp(self):
        self.data1 = simulate_ddm(
            n_trials=1000,
            gen_drift=.6,
            gen_drift_trial_sd=.1,
            gen_threshold=1.4,
            gen_ndt=.23)

        self.data2 = simulate_ddm(
            n_trials=1000,
            gen_drift=-.6,
            gen_drift_trial_sd=.1,
            gen_threshold=1.4,
            gen_ndt=.23)

        self.data3 = simulate_ddm(
            n_trials=1000,
            gen_drift=.6,
            gen_drift_trial_sd=.1,
            gen_threshold=2.4,
            gen_ndt=.23)

        self.data4 = simulate_ddm(
            n_trials=1000,
            gen_drift=.6,
            gen_drift_trial_sd=.1,
            gen_rel_sp=.8,
            gen_threshold=1.4,
            gen_ndt=.23)

        self.data5 = simulate_ddm(
            n_trials=1000,
            gen_drift=.6,
            gen_drift_trial_sd=.1,
            gen_threshold=1.4,
            gen_ndt=.5)

    def test_random_DDM_test1(self):
        # TEST: assure there is only 1 participant
        assert self.data1.index[-1][0] == 1, f"Number of participants should be 1"

    def test_random_DDM_test2(self):
        # TEST: check that accuracy of data1 > accuracy of data2
        assert np.mean(self.data1['accuracy']) > np.mean(
            self.data2['accuracy']), f"Accuracy(data1) is not better than accuracy(data2)"

    def test_random_DDM_test3(self):
        # TEST: check if data3 has slower response time than data1 and data3 has higher accuracy than data1
        assert np.mean(self.data3['rt']) > np.mean(self.data1['rt']), f"RT data3 is not slower than RT data1"

    def test_random_DDM_test4(self):
        assert np.mean(self.data3['accuracy']) > np.mean(
            self.data1['accuracy']), f"Accuracy(data3) is not better than accuracy(data1)"

    def test_random_DDM_test5(self):
        # TEST: threshold should be positive
        assert all(i > 0 for i in self.data1['threshold']), f"Threshold should be positive"

    def test_random_DDM_test6(self):
        # TEST: rel_sp should be positive
        assert all(i > 0 for i in self.data1['rel_sp']), f"rel_sp should be positive"

    def test_random_DDM_test7(self):
        # TEST: data4 response time for accurate decision is faster than for non accurate decision
        assert np.mean(self.data4.loc[self.data4['accuracy'] == 1.0]['rt']) <= np.mean(
            self.data4.loc[self.data4['accuracy'] == 0][
                'rt']), f"rt(accurate decision) should be faster than rt(non accurate decision)"

    def test_random_DDM_test8(self):
        # TEST: rt for data 5 > rt for data 1
        assert np.mean(self.data5['rt']) > np.mean(
            self.data1['rt']), f"min rt(data5) should be greater than min rt(data1)"

    def test_random_DDM_hier(self):
        # TEST hierarchical version
        data_hier = simulate_hier_ddm(n_trials=100,
                                      n_participants=30,
                                      gen_mu_drift=1,
                                      gen_sd_drift=.5,
                                      gen_mu_threshold=1,
                                      gen_sd_threshold=.1,
                                      gen_mu_ndt=.23,
                                      gen_sd_ndt=.1,
                                      gen_drift_trial_sd=.01)

        # TEST: assure that there are 30 participants
        assert data_hier.index[-1][0] == 30, f"Number of participants should be 30"
