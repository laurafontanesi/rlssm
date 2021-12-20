# Reinforcement learning data
from rlssm.random import simulate_rlddm_2A, generate_task_design_fontanesi

dm = generate_task_design_fontanesi(n_trials_block=80,
                                    n_blocks=3,
                                    n_participants=1,
                                    trial_types=['1-2', '1-3', '2-4', '3-4'],
                                    mean_options=[34, 38, 50, 54],
                                    sd_options=[5, 5, 5, 5])

data = simulate_rlddm_2A(task_design=dm,
                         gen_alpha=[.1, .01],
                         gen_drift_scaling=.1,
                         gen_threshold=1,
                         gen_ndt=.23,
                         initial_value_learning=0)

print("RL + DDM with 2 alphas; simulate_rlddm_2A works")
print(data)