import os
import pandas as pd

from rlssm import DDModel


def test_fit_experiment_data(print_results=True):
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'data_experiment.csv')
    data = pd.read_csv(data_path, index_col=0)
    data = data[data.participant == 20].reset_index(drop=True)
    data['block_label'] += 1

    if print_results:
        print(data)

    # Test that data imported is same as reference data
    reference_path = os.path.join(os.path.dirname(__file__), 'reference_data', 'data.csv')
    reference_data = pd.read_csv(reference_path, index_col=0)
    assert data.equals(reference_data)

    model = DDModel(hierarchical_levels=1)
    n_iter, n_chains, n_thin = 1000, 2, 1
    model_fit = model.fit(
        data,
        thin=n_thin,
        iter=n_iter,
        chains=n_chains,
        pointwise_waic=False,
        verbose=False,
        print_diagnostics=print_results)

    # Test data produced against reference data
    reference_path = os.path.join(os.path.dirname(__file__), 'reference_data', 'model_fit.csv')
    reference_data = pd.read_csv(reference_path, index_col=0)
    assert model_fit.data_info['data'].equals(reference_data)

    if print_results:
        print(model_fit.data_info['data'])
