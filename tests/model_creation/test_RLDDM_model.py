import os
import re

from rlssm import RLDDModel


def test_RLDDM_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    rlddm_model = RLDDModel(hierarchical_levels=1)
    if len([file for file in os.listdir(pkl_path) if re.search('-RLDDM', file)]):
        print("Success - Test hierRLDDM pkl model existence")
    else:
        print("Failure - Test hierRLDDM pkl model existence")


def test_hierRLDDM_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    hier_rlddm_model = RLDDModel(hierarchical_levels=2)
    if len([file for file in os.listdir(pkl_path) if re.search('hierRLDDM', file)]):
        print("Success - Test RLDDM pkl model existence")
    else:
        print("Failure - Test RLDDM pkl model existence")
