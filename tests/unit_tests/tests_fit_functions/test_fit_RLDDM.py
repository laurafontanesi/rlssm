import unittest

from rlssm.model.models_DDM import RLDDModel
from rlssm.utility.load_data import load_example_dataset


class TestFitRLDDM(unittest.TestCase):
    def test_fit_RLDDM(self):
        hier_levels = 1

        model = RLDDModel(hierarchical_levels=hier_levels)

        data = load_example_dataset(hierarchical_levels=hier_levels)

        model_fit = model.fit(data,
                              K=4,
                              initial_value_learning=27.5,
                              iter=1000,
                              chains=2,
                              pointwise_waic=False,
                              verbose=False)

        pred = model_fit.get_posterior_predictives_df(100)

    def test_fit_RLDDM_hier(self):
        hier_levels = 2

        model = RLDDModel(hierarchical_levels=hier_levels)

        data = load_example_dataset(hierarchical_levels=hier_levels)

        # to make the hier test work faster, only take the first 10 participants into consideration
        data_hier = data[data['participant'] <= 10]

        model_fit = model.fit(data_hier,
                              K=4,
                              initial_value_learning=27.5,
                              warmup=50,
                              iter=200,
                              chains=2,
                              verbose=False)
