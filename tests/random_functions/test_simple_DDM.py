import os
import pandas as pd

from rlssm.random.random_DDM import simulate_ddm


# Sequential sampling models data
# NON HIER DATA, 2 alternatives
def test_simple_DDM(print_results=True):
    print("Test - simple simulate_ddm")

    data = simulate_ddm(
        n_trials=300,
        gen_drift=.6,
        gen_drift_trialsd=.1,
        gen_threshold=1.4,
        gen_ndt=.23)

    # Test data produced against reference data
    reference_path = os.path.join(os.path.dirname(__file__), 'reference_data', 'simple_DDM.csv')
    # data.to_csv(reference_path)
    reference_data = pd.read_csv(reference_path, index_col=0)
    # assert data.equals(reference_data)

    if print_results:
        print(data)
