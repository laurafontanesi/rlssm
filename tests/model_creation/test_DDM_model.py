import os
import re
from rlssm import DDModel


def test_DDM_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    ddm_model = DDModel(hierarchical_levels=2)
    if len([file for file in os.listdir(pkl_path) if re.search('DDM', file)]):
        print("Success - Test DDM pkl model existence")
    else:
        print("Failure - Test DDM pkl model existence")
