import os
import numpy as np
import pandas as pd

__dir__ = os.path.abspath(os.path.dirname(__file__))

def load_example_dataset(hierarchical_levels, n_alternatives=2):
    """Load example dataset for testing and tutorials.

    Parameters
    ----------
    n_alternatives : int
        When 2, the dataset of https://doi.org/10.3758/s13423-018-1554-2
        is loaded.

    """
    if not isinstance(hierarchical_levels, int):
        raise TypeError("hierarchical_levels must be integer")
    if np.sum(hierarchical_levels == np.array([1, 2])) < 1:
        raise ValueError("set to 1 for individual data, and to 2 for grouped data")

    if n_alternatives == 2:
        data_path = os.path.join(__dir__, os.pardir, 'data', 'data_experiment.csv')
        data = pd.read_csv(data_path, index_col=0)

        if hierarchical_levels == 1:
            # Select 1 random participant
            pp = np.random.choice(pd.unique(data.participant))
            data = data[data.participant == pp].reset_index(drop=True)
            
        return data

    else:
        raise ValueError("For now there is only a decision between 2 alternatives dataset")