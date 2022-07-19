import unittest

from rlssm.model.models_LBA import LBAModel_2A
from rlssm.utility.load_data import load_example_dataset


class TestFitLBA(unittest.TestCase):
    def test_fit_LBA(self):
        hier_levels = 1

        model = LBAModel_2A(hierarchical_levels=hier_levels)

        data = load_example_dataset(hierarchical_levels=hier_levels)

        model_fit = model.fit(data,
                              iter_sampling=500,
                              iter_warmup=500,
                              chains=2,
                              parallel_chains=2)

        assert not model_fit.get_posterior_predictives_df(100).empty, "Posterior predictives could not be retrieved"

    def test_fit_LBA_hier(self):
        hier_levels = 2

        model = LBAModel_2A(hierarchical_levels=hier_levels)

        data = load_example_dataset(hierarchical_levels=hier_levels)

        # to make the hier test work faster, only take the first 10 participants into consideration
        data_hier = data[data['participant'] <= 10]

        drift_priors = {'mu_mu': 1, 'sd_mu': 1, 'mu_sd': 0, 'sd_sd': 1}
        threshold_priors = {'mu_mu': -1, 'sd_mu': .5, 'mu_sd': 0, 'sd_sd': 1}

        model_fit = model.fit(data_hier,
                              drift_priors=drift_priors,
                              iter_sampling=500,
                              iter_warmup=500,
                              chains=2,
                              parallel_chains=2)

        assert not model_fit.get_posterior_predictives_df(100).empty, "Posterior predictives could not be retrieved"
