import unittest

from rlssm.utility.load_data import load_example_dataset
from rlssm.model.models_RL import RLModel_2A


class TestFitRL(unittest.TestCase):
    def test_fit_RL(self):
        hier_levels = 1

        model = RLModel_2A(hierarchical_levels=hier_levels)

        data = load_example_dataset(hierarchical_levels=hier_levels)

        model_fit = model.fit(data,
                              K=4,
                              initial_value_learning=27.5,
                              sensitivity_priors={'mu': 0, 'sd': 5},
                              iter=1000,
                              chains=2,
                              verbose=False)

    def test_fit_RL_hier(self):
        hier_levels = 2

        model = RLModel_2A(hierarchical_levels=hier_levels)

        data = load_example_dataset(hierarchical_levels=hier_levels)

        # to make the hier test work faster, only take the first 10 participants into consideration
        data_hier = data[data['participant'] <= 10]

        model_fit = model.fit(data_hier,
                              K=4,
                              initial_value_learning=27.5,
                              warmup=50,
                              iter=200,
                              chains=2)
