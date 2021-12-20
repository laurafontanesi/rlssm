# Sequential sampling models data
# NON HIER DATA, 2 alternatives

from rlssm.random import simulate_ddm

data = simulate_ddm(
    n_trials=300,
    gen_drift=.6,
    gen_drift_trialsd=.1,
    gen_threshold=1.4,
    gen_ndt=.23)

print("Simple DDM; simulate_ddm works")
print(data)
