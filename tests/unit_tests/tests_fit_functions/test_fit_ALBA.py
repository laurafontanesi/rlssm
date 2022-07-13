import unittest

import numpy as np

from rlssm.model.models_ALBA import ALBAModel_2A
from rlssm.utility.load_data import load_example_dataset


class TestFitALBA(unittest.TestCase):
    def test_fit_ALBA(self):
        hier_levels = 1

        model = ALBAModel_2A(hierarchical_levels=hier_levels)

        data = load_example_dataset(hierarchical_levels=hier_levels)

        data['S_cor'] = np.random.normal(.4, 0.01, data.shape[0])
        data['S_inc'] = np.random.normal(.3, 0.01, data.shape[0])

        model_fit = model.fit(data,
                              iter_sampling=500,
                              iter_warmup=500,
                              chains=2,
                              parallel_chains=2,
                              ndt_priors={'mu': 0, 'sd': .1})

    def test_fit_ALBA_hier(self):
        hier_levels = 2

        model = ALBAModel_2A(hierarchical_levels=hier_levels)

        data = load_example_dataset(hierarchical_levels=hier_levels)

        # to make the hier test work faster, only take the first 10 participants into consideration
        data_hier = data[data['participant'] <= 10]

        data_hier['S_cor'] = np.random.normal(.4, 0.01, data_hier.shape[0])
        data_hier['S_inc'] = np.random.normal(.3, 0.01, data_hier.shape[0])

        sp_trial_var_priors = {'mu_mu': -1, 'sd_mu': .5, 'mu_sd': 0, 'sd_sd': 1}

        model_fit = model.fit(data_hier,
                              sp_trial_var_priors=sp_trial_var_priors,
                              iter_sampling=500,
                              iter_warmup=500,
                              chains=2,
                              parallel_chains=2,
                              ndt_priors={'mu_mu': -1, 'sd_mu': .5, 'mu_sd': 0, 'sd_sd': 1})
