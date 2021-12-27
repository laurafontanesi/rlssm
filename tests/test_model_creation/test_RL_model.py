import os
import re

from rlssm import RLModel_2A

__dir__ = os.path.abspath(os.path.dirname(__file__))
path_pkl_fldr = os.path.join(os.path.dirname(os.path.dirname(__dir__)), "pkl_files")

rl_model = RLModel_2A(hierarchical_levels=2)
if len([file for file in os.listdir(path_pkl_fldr) if re.search('RL', file)]):
    print("Success - Test RL pkl model existence")
else:
    print("Failure - Test RL pkl model existence")