# Sequential sampling models data
# NON HIER DATA, 2 alternatives
from rlssm.random.random_DDM import simulate_ddm


def test_simple_DDM(print_results=True):
    print("Test - simple simulate_ddm")

    data = simulate_ddm(
        n_trials=300,
        gen_drift=.6,
        gen_drift_trialsd=.1,
        gen_threshold=1.4,
        gen_ndt=.23)

    if print_results:
        print(data)
