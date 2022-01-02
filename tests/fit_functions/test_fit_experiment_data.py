import os
import pandas as pd

from rlssm import DDModel


def test_fit_experiment_data(print_results=True):
    data_path = os.path.join(os.getcwd(), os.pardir, 'data', 'data_experiment.csv')
    data = pd.read_csv(data_path, index_col=0)

    data = data[data.participant == 20].reset_index(drop=True)

    data['block_label'] += 1

    if print_results:
        print(data)

    model = DDModel(hierarchical_levels=1)

    # sampling parameters
    n_iter, n_chains, n_thin = 1000, 2, 1

    model_fit = model.fit(
        data,
        thin=n_thin,
        iter=n_iter,
        chains=n_chains,
        pointwise_waic=False,
        verbose=False,
        print_diagnostics=print_results)

    if print_results:
        print(model_fit.data_info['data'])
