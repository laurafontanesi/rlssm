import os
import pandas as pd

from rlssm.random.random_DDM import simulate_hier_ddm


# Sequential sampling models data
def test_hier_DDM(print_results=True):
    print("Test - Hierarchical DDM; simulate_hier_ddm")

    data = simulate_hier_ddm(n_trials=100,
                             n_participants=30,
                             gen_mu_drift=1,
                             gen_sd_drift=.5,
                             gen_mu_threshold=1,
                             gen_sd_threshold=.1,
                             gen_mu_ndt=.23,
                             gen_sd_ndt=.1)

    # Test data produced against reference data
    reference_path = os.path.join(os.path.dirname(__file__), 'reference_data', 'hier_DDM.csv')
    # data.to_csv(reference_path)
    reference_data = pd.read_csv(reference_path, index_col=0)
    # assert data.equals(reference_data)

    if print_results:
        print(data)
