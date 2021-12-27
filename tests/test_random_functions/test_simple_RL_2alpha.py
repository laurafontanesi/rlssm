# Reinforcement learning data
## HIER DATA, 2 alternatives
from rlssm.random.random_RL import simulate_rl_2A
from rlssm.random.random_common import generate_task_design_fontanesi

dm = generate_task_design_fontanesi(n_trials_block=80,
                                    n_blocks=3,
                                    n_participants=1,
                                    trial_types=['1-2', '1-3', '2-4', '3-4'],
                                    mean_options=[34, 38, 50, 54],
                                    sd_options=[5, 5, 5, 5])

data = simulate_rl_2A(task_design=dm,
                      gen_alpha=[.2, .05],
                      gen_sensitivity=.5,
                      initial_value_learning=20)
print("simple RL_2A with 2 alphas; simulate_rl_2A works")
print(data)