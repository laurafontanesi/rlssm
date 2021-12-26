# HIER DATA, random2 alternatives
from rlssm.random.random import generate_task_design_fontanesi

dm = generate_task_design_fontanesi(n_trials_block=80,
                                    n_blocks=3,
                                    n_participants=30,
                                    trial_types=['1-2', '1-3', '2-4', '3-4'],
                                    mean_options=[34, 38, 50, 54],
                                    sd_options=[5, 5, 5, 5])

print("SIMULATE HIER DATA, 2 alternatives per trial (4 per block), design matrix:")
print(dm)
print("generating HIER DATA dm works")
