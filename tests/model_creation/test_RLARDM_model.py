import os
import re

from rlssm import RLARDModel_2A


def test_RLARDM_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    rlardm_model = RLARDModel_2A(hierarchical_levels=1)
    if len([file for file in os.listdir(pkl_path) if re.search('-RLARDM', file)]):
        print("Success - Test RLARDM pkl model existence")
    else:
        print("Failure - Test RLARDM pkl model existence")


def test_hierRLARDM_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    hier_rlardm_model = RLARDModel_2A(hierarchical_levels=2)
    if len([file for file in os.listdir(pkl_path) if re.search('hierRLARDM', file)]):
        print("Success - Test hierRLARDM pkl model existence")
    else:
        print("Failure - Test hierRLARDM pkl model existence")
