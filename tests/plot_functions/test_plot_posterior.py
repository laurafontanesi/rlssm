import os

import pandas as pd

from rlssm import DDModel, ALBAModel_2A


def test_plot_posterior(print_results=True):
    # load non-hierarchical DDM fit:
    model = DDModel(hierarchical_levels=1)

    data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'data_experiment.csv')
    data = pd.read_csv(data_path, index_col=0)
    data = data[data.participant == 20].reset_index(drop=True)
    data['block_label'] += 1

    # sampling parameters
    n_iter, n_chains, n_thin = 1000, 2, 1

    model_fit = model.fit(
        data,
        thin=n_thin,
        iter=n_iter,
        chains=n_chains,
        pointwise_waic=False,
        verbose=False)

    print(model_fit.samples.describe())
