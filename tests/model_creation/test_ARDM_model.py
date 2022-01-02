import os
import re
from rlssm import ARDModel_2A


def test_ARDM_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    ardm_model = ARDModel_2A(hierarchical_levels=1)
    if len([file for file in os.listdir(pkl_path) if re.search('-ARDM', file)]):
        print("Success - Test ARDM pkl model existence")
    else:
        print("Failure - Test ARDM pkl model existence")


def test_hierARDM_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    hier_ardm_model = ARDModel_2A(hierarchical_levels=2)
    if len([file for file in os.listdir(pkl_path) if re.search('hierARDM', file)]):
        print("Success - Test hierARDM pkl model existence")
    else:
        print("Failure - Test hierARDM pkl model existence")