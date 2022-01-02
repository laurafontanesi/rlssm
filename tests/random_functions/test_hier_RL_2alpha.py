# Reinforcement learning data
from rlssm.random.random_RL import simulate_hier_rl_2A
from rlssm.random.random_common import generate_task_design_fontanesi


def test_hier_RL_2alpha(print_results=True):
    print("Test - simple RL_2A with 2 alphas; simulate_hier_rl_2A")

    dm = generate_task_design_fontanesi(n_trials_block=80,
                                        n_blocks=3,
                                        n_participants=30,
                                        trial_types=['1-2', '1-3', '2-4', '3-4'],
                                        mean_options=[34, 38, 50, 54],
                                        sd_options=[5, 5, 5, 5])

    data = simulate_hier_rl_2A(task_design=dm,
                               gen_mu_alpha=[-.5, -1],
                               gen_sd_alpha=[.1, .1],
                               gen_mu_sensitivity=.5,
                               gen_sd_sensitivity=.1,
                               initial_value_learning=20)
    if print_results:
        print(data)
