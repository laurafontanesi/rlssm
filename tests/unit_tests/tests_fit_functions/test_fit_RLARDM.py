import unittest

from rlssm.model.models_ARDM import RLARDModel_2A
from rlssm.utility.load_data import load_example_dataset


class TestFitRLARDM(unittest.TestCase):
    def test_fit_RLARDM(self):
        hier_levels = 1

        model = RLARDModel_2A(hierarchical_levels=hier_levels)

        data = load_example_dataset(hierarchical_levels=hier_levels)

        model_fit = model.fit(data,
                              K=4,
                              initial_value_learning=27.5,
                              # iter=1000,
                              chains=2)

    def test_fit_RLARDM_hier(self):
        hier_levels = 2

        model = RLARDModel_2A(hierarchical_levels=hier_levels)

        data = load_example_dataset(hierarchical_levels=hier_levels)

        # to make the hier test work faster, only take the first 10 participants into consideration
        data_hier = data[data['participant'] <= 10]

        model_fit = model.fit(data_hier,
                              K=4,
                              initial_value_learning=27.5,
                              # warmup=50,
                              # iter=200,
                              chains=2)
