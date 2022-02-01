import os

import pandas as pd


def get_experiment_data():
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'data_experiment.csv')
    data = pd.read_csv(data_path, index_col=0)
    data = data[data.participant == 20].reset_index(drop=True)
    data['block_label'] += 1

    # Test that data imported is same as reference data
    reference_path = os.path.join(os.path.dirname(__file__), 'reference_data', 'data.csv')
    reference_data = pd.read_csv(reference_path, index_col=0)
    assert data.equals(reference_data)

    return data
