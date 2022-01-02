# Reinforcement learning data
from rlssm.random.random_RL_DDM import simulate_rlddm_2A
from rlssm.random.random_common import generate_task_design_fontanesi


def test_RL_DDM_2_alpha(print_results=True):
    print("Test - RL + DDM 2 alpha; simulate_rlddm_2A")

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
    if print_results:
        print(data)
