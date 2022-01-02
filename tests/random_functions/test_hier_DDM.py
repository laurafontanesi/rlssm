# Sequential sampling models data
from rlssm.random.random_DDM import simulate_hier_ddm


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

    if print_results:
        print(data)
