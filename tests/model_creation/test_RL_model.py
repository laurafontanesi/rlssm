import os
import re
from rlssm import RLModel_2A


def test_RL_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    rl_model = RLModel_2A(hierarchical_levels=2)
    if len([file for file in os.listdir(pkl_path) if re.search('RL', file)]):
        print("Success - Test RL pkl model existence")
    else:
        print("Failure - Test RL pkl model existence")
