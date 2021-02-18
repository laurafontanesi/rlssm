import rlssm
import pandas as pd
import pandas as pd
import numpy as np

## Reinforcement learning data
from rlssm.random import generate_task_design_fontanesi, simulate_rl_2alternatives, simulate_hier_rl_2alternatives
from rlssm.random import simulate_rlddm_2alternatives, simulate_hier_rlddm_2alternatives
from rlssm.random import simulate_rlrdm_2alternatives
## NON HIER DATA, 2 alternatives
dm = generate_task_design_fontanesi(n_trials_block=80,
                                    n_blocks=3,
                                    n_participants=1,
                                    trial_types=['1-2', '1-3', '2-4', '3-4'],
                                    mean_options=[34, 38, 50, 54],
                                    sd_options=[5, 5, 5, 5])
print("SIMULATE NON HIER DATA, 2 alternatives per trial (4 per block), design matrix:")
print(dm)

data = simulate_rl_2alternatives(task_design=dm,
                                 gen_alpha=.1,
                                 gen_sensitivity=.5,
                                 initial_value_learning=20)
print("simple RL")
print(data)

data = simulate_rl_2alternatives(task_design=dm,
                                 gen_alpha=[.2, .05],
                                 gen_sensitivity=.5,
                                 initial_value_learning=20)
print("simple RL with 2 alphas")
print(data)

data = simulate_rlrdm_2alternatives(task_design=dm,
                                     gen_alpha=0.1,
                                     gen_drift_scaling=.1,
                                     gen_threshold=1,
                                     gen_ndt=.23,
                                     initial_value_learning=0)
print("RL + RDM")
print(data)

data = simulate_rlddm_2alternatives(task_design=dm,
                                     gen_alpha=.1,
                                     gen_drift_scaling=.1,
                                     gen_threshold=1,
                                     gen_ndt=.23,
                                     initial_value_learning=0)
print("RL + DDM")
print(data)

data = simulate_rlddm_2alternatives(task_design=dm,
                                     gen_alpha=[.1, .01],
                                     gen_drift_scaling=.1,
                                     gen_threshold=1,
                                     gen_ndt=.23,
                                     initial_value_learning=0)
print("RL + DDM with 2 alphas")
print(data)

## HIER DATA, 2 alternatives
dm = generate_task_design_fontanesi(n_trials_block=80,
                                    n_blocks=3,
                                    n_participants=30,
                                    trial_types=['1-2', '1-3', '2-4', '3-4'],
                                    mean_options=[34, 38, 50, 54],
                                    sd_options=[5, 5, 5, 5])
print("SIMULATE HIER DATA, 2 alternatives per trial (4 per block), design matrix:")
print(dm)

data = simulate_hier_rl_2alternatives(task_design=dm,
                                      gen_mu_alpha=-.5,
                                      gen_sd_alpha=.1,
                                      gen_mu_sensitivity=.5,
                                      gen_sd_sensitivity=.1,
                                      initial_value_learning=20)
print("simple RL")
print(data)

data = simulate_hier_rl_2alternatives(task_design=dm,
                                      gen_mu_alpha=[-.5, -1],
                                      gen_sd_alpha=[.1, .1],
                                      gen_mu_sensitivity=.5,
                                      gen_sd_sensitivity=.1,
                                      initial_value_learning=20)
print("simple RL with 2 alphas")
print(data)

data = simulate_hier_rlddm_2alternatives(task_design=dm,
                                          gen_mu_alpha=-.5,
                                          gen_sd_alpha=.1,
                                          gen_mu_drift_scaling=.1,
                                          gen_sd_drift_scaling=.5,
                                          gen_mu_threshold=1,
                                          gen_sd_threshold=.1,
                                          gen_mu_ndt=.23,
                                          gen_sd_ndt=.05,
                                          initial_value_learning=20)
print("RL + DDM")
print(data)

data = simulate_hier_rlddm_2alternatives(task_design=dm,
                                          gen_mu_alpha=[-.5, -1],
                                          gen_sd_alpha=[.1, .1],
                                          gen_mu_drift_scaling=.1,
                                          gen_sd_drift_scaling=.5,
                                          gen_mu_threshold=1,
                                          gen_sd_threshold=.1,
                                          gen_mu_ndt=.23,
                                          gen_sd_ndt=.05,
                                          initial_value_learning=20)
print("RL + DDM with 2 alphas")
print(data)
print("individual alphas:")
print(pd.unique(data.alpha_pos))
print(pd.unique(data.alpha_neg))

## Sequential sampling models data
from rlssm.random import simulate_ddm, simulate_hier_ddm
## NON HIER DATA, 2 alternatives
data = simulate_ddm(
    n_trials=300, 
    gen_drift=.6, 
    gen_drift_trialsd=.1,
    gen_threshold=1.4, 
    gen_ndt=.23)
print("Simple DDM")
print(data)

data = simulate_hier_ddm(n_trials=100,
    n_participants=30,
    gen_mu_drift=1,
    gen_sd_drift=.5,
    gen_mu_threshold=1,
    gen_sd_threshold=.1,
    gen_mu_ndt=.23,
    gen_sd_ndt=.1)
print("Hierarchical DDM")
print(data)