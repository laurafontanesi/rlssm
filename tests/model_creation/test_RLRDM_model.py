import os
import re

from rlssm import RLRDModel_2A


def test_RLRDM_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    rlrdm_model = RLRDModel_2A(hierarchical_levels=1)
    if len([file for file in os.listdir(pkl_path) if re.search('-RLRDM', file)]):
        print("Success - Test RLRDM pkl model existence")
    else:
        print("Failure - Test RLRDM pkl model existence")


def test_hierRLRDM_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    hier_rlrdm_model = RLRDModel_2A(hierarchical_levels=2)
    if len([file for file in os.listdir(pkl_path) if re.search('hierRLRDM', file)]):
        print("Success - Test hierRLRDM pkl model existence")
    else:
        print("Failure - Test hierRLRDM pkl model existence")
