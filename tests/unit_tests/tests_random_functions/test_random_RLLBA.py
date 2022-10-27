import unittest

from rlssm.random.random_RL_LBA import simulate_rllba_2A, simulate_hier_rllba
from rlssm.random.random_common import generate_task_design_fontanesi


class TestRandomRLDDM(unittest.TestCase):
    def test_random_RLLBA(self):
        self.dm_non_hier = generate_task_design_fontanesi(n_trials_block=80,
                                                          n_blocks=3,
                                                          n_participants=1,
                                                          trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                          mean_options=[34, 38, 50, 54],
                                                          sd_options=[5, 5, 5, 5])

        self.data_non_hier = simulate_rllba_2A(task_design=self.dm_non_hier,
                                               gen_alpha=.1,
                                               gen_sp_trial_var=.2,
                                               gen_ndt=.2,
                                               gen_k=3,
                                               gen_drift_scaling=.1)

        # TEST: assure there is only 1 participant
        assert self.data_non_hier.index[-1][0] == 1, f"Number of participants should be 1"

    def test_random_RLLBA_hier(self):
        self.dm_hier = generate_task_design_fontanesi(n_trials_block=80,
                                                      n_blocks=3,
                                                      n_participants=30,
                                                      trial_types=['1-2', '1-3', '2-4', '3-4'],
                                                      mean_options=[34, 38, 50, 54],
                                                      sd_options=[5, 5, 5, 5])

        self.data_hier_2alpha = simulate_hier_rllba(task_design=self.dm_hier,
                                                    gen_mu_alpha=[-.5, -1],
                                                    gen_sd_alpha=[.1, .1],
                                                    gen_mu_sp_trial_var=1,
                                                    gen_sd_sp_trial_var=.1,
                                                    gen_mu_ndt=.23,
                                                    gen_sd_ndt=.05,
                                                    gen_mu_k=.5,
                                                    gen_sd_k=.05,
                                                    gen_mu_drift_trial_var=.1,
                                                    gen_sd_drift_trial_var=.1,
                                                    gen_mu_drift_scaling=.1,
                                                    gen_sd_drift_scaling=.5)

        # TEST: assure that there are 30 participants in data_hier_2alpha
        assert self.data_hier_2alpha.index[-1][0] == 30, f"Number of participants should be 30"
