import os
import pandas as pd

from rlssm import RDModel_2A, load_example_dataset
from tests.fit_functions.common_methods import get_experiment_data


def test_fit_RDM(hier_levels=1, print_results=True, test_hier=False):
    model = RDModel_2A(hierarchical_levels=hier_levels)

    data = load_example_dataset(hierarchical_levels=hier_levels)

    if hier_levels == 1:
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
                              verbose=False,
                              print_diagnostics=print_results)

    if test_hier and hier_levels > 1:
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
