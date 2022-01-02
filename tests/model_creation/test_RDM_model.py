import os
import re
from rlssm import RDModel_2A


def test_RDM_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    rdm_model = RDModel_2A(hierarchical_levels=1)
    if len([file for file in os.listdir(pkl_path) if re.search('-RDM', file)]):
        print("Success - Test RDM pkl model existence")
    else:
        print("Failure - Test RDM pkl model existence")


def test_hierRDM_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    hier_rdm_model = RDModel_2A(hierarchical_levels=2)
    if len([file for file in os.listdir(pkl_path) if re.search('hierRDM', file)]):
        print("Success - Test hierRDM pkl model existence")
    else:
        print("Failure - Test hierRDM pkl model existence")
