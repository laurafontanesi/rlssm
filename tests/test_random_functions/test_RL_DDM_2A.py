# Reinforcement learning data
# HIER DATA, 2 alternatives
from rlssm.random.random_RL_DDM import simulate_hier_rlddm_2A
from rlssm.random.random_common import generate_task_design_fontanesi

dm = generate_task_design_fontanesi(n_trials_block=80,
                                    n_blocks=3,
                                    n_participants=30,
                                    trial_types=['1-2', '1-3', '2-4', '3-4'],
                                    mean_options=[34, 38, 50, 54],
                                    sd_options=[5, 5, 5, 5])

data = simulate_hier_rlddm_2A(task_design=dm,
                              gen_mu_alpha=-.5,
                              gen_sd_alpha=.1,
                              gen_mu_drift_scaling=.1,
                              gen_sd_drift_scaling=.5,
                              gen_mu_threshold=1,
                              gen_sd_threshold=.1,
                              gen_mu_ndt=.23,
                              gen_sd_ndt=.05,
                              initial_value_learning=20)
print("RL_2A + DDM; simulate_hier_rlddm_2A works")
print(data)