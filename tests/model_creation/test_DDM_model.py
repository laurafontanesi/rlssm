import os
import re
from rlssm import DDModel


def test_DDM_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    ddm_model = DDModel(hierarchical_levels=1)
    if len([file for file in os.listdir(pkl_path) if re.search('-DDM', file)]):
        print("Success - Test DDM pkl model existence")
    else:
        print("Failure - Test DDM pkl model existence")


def test_hierDDM_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    hier_ddm_model = DDModel(hierarchical_levels=2)
    if len([file for file in os.listdir(pkl_path) if re.search('hierDDM', file)]):
        print("Success - Test hierDDM pkl model existence")
    else:
        print("Failure - Test hierDDM pkl model existence")
