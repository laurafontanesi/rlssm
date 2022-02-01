from rlssm import RLModel_2A, load_example_dataset


def test_fit_RL(hier_levels=1, print_results=True, test_hier=False):
    model = RLModel_2A(hierarchical_levels=hier_levels)

    data = load_example_dataset(hierarchical_levels=hier_levels)

    if hier_levels == 1:
        model_fit = model.fit(data,
                              K=4,
                              initial_value_learning=27.5,
                              sensitivity_priors={'mu': 0, 'sd': 5},
                              iter=1000,
                              chains=2,
                              verbose=False)

    if test_hier and hier_levels > 1:
        # to make the hier test work faster, only take the first 10 participants into consideration
        data_hier = data[data['participant'] <= 10]

        model_fit = model.fit(data_hier,
                              K=4,
                              initial_value_learning=27.5,
                              warmup=50,
                              iter=200,
                              chains=2)
