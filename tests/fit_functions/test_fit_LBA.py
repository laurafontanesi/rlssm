from rlssm import LBAModel_2A, load_example_dataset


def test_fit_LBA(hier_levels=1, print_results=True, test_hier=False):
    model = LBAModel_2A(hierarchical_levels=hier_levels)

    data = load_example_dataset(hierarchical_levels=hier_levels)

    if hier_levels == 1:
        model_fit = model.fit(
            data,
            iter=1000,
            chains=2)

    if test_hier and hier_levels > 1:
        # to make the hier test work faster, only take the first 10 participants into consideration
        data_hier = data[data['participant'] <= 10]

        drift_priors = {'mu_mu': 1, 'sd_mu': 1, 'mu_sd': 0, 'sd_sd': 1}
        threshold_priors = {'mu_mu': -1, 'sd_mu': .5, 'mu_sd': 0, 'sd_sd': 1}

        model_fit = model.fit(data_hier,
                              drift_priors=drift_priors,
                              warmup=50,
                              iter=200,
                              chains=2,
                              verbose=False)
