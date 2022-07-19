import unittest

import numpy as np

from rlssm.utility.load_data import load_example_dataset
from rlssm.model.models_RL import RLModel_2A


class TestFitRL(unittest.TestCase):
    def test_fit_RL(self):
        hier_levels = 1

        model = RLModel_2A(hierarchical_levels=hier_levels)

        data = load_example_dataset(hierarchical_levels=hier_levels)
        data['feedback_type'] = np.array(1)

        model_fit = model.fit(data,
                              K=4,
                              initial_value_learning=27.5,
                              sensitivity_priors={'mu': 0, 'sd': 5},
                              iter_sampling=500,
                              iter_warmup=500,
                              chains=2,
                              parallel_chains=2)

        assert not model_fit.get_posterior_predictives_df(100).empty, "Posterior predictives could not be retrieved"

    def test_fit_RL_hier(self):
        hier_levels = 2

        model = RLModel_2A(hierarchical_levels=hier_levels)

        data = load_example_dataset(hierarchical_levels=hier_levels)
        data['feedback_type'] = np.array(1)

        # to make the hier test work faster, only take the first 10 participants into consideration
        data_hier = data[data['participant'] <= 10]

        model_fit = model.fit(data_hier,
                              K=4,
                              initial_value_learning=27.5,
                              iter_sampling=500,
                              iter_warmup=500,
                              chains=2,
                              parallel_chains=2)

        assert not model_fit.get_posterior_predictives_df(100).empty, "Posterior predictives could not be retrieved"
