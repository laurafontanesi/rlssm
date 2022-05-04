import unittest

from rlssm.model.models_ARDM import ARDModel_2A
from rlssm.utility.load_data import load_example_dataset


class TestFitARDM(unittest.TestCase):
    def test_fit_ARDM(self):
        hier_levels = 1

        model = ARDModel_2A(hierarchical_levels=hier_levels)

        data = load_example_dataset(hierarchical_levels=hier_levels)

        threshold_priors = {'mu': 0, 'sd': 5}
        drift_priors = {'mu': 0, 'sd': 5}
        ndt_priors = {'mu': 0, 'sd': 5}

        model_fit = model.fit(data,
                              threshold_priors=threshold_priors,
                              ndt_priors=drift_priors,
                              drift_priors=ndt_priors,
                              iter=1000,
                              chains=2,
                              pointwise_waic=False,
                              verbose=False)

    def test_fit_ARDM_hier(self):
        hier_levels = 2

        model = ARDModel_2A(hierarchical_levels=hier_levels)

        data = load_example_dataset(hierarchical_levels=hier_levels)

        # to make the hier test work faster, only take the first 10 participants into consideration
        data_hier = data[data['participant'] <= 10]

        drift_priors = {'mu_mu': 1, 'sd_mu': 1, 'mu_sd': 0, 'sd_sd': 1}
        threshold_priors = {'mu_mu': -1, 'sd_mu': .5, 'mu_sd': 0, 'sd_sd': 1}
        ndt_priors = {'mu_mu': 1, 'sd_mu': 1, 'mu_sd': 0, 'sd_sd': 1}

        model_fit = model.fit(data_hier,
                              drift_priors=drift_priors,
                              threshold_priors=threshold_priors,
                              ndt_priors=ndt_priors,
                              warmup=50,
                              iter=200,
                              chains=2,
                              verbose=False)
