import os
import re
from rlssm import ALBAModel_2A


def test_ALBA_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    alba_model = ALBAModel_2A(hierarchical_levels=1)
    if len([file for file in os.listdir(pkl_path) if re.search('-ALBA', file)]):
        print("Success - Test ALBA pkl model existence")
    else:
        print("Failure - Test ALBA pkl model existence")


def test_hierALBA_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    hier_alba_model = ALBAModel_2A(hierarchical_levels=2)
    if len([file for file in os.listdir(pkl_path) if re.search('hierALBA', file)]):
        print("Success - Test hierALBA pkl model existence")
    else:
        print("Failure - Test hierALBA pkl model existence")
