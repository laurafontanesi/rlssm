import os
import re
from rlssm import RLLBAModel_2A


def test_RLLBA_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    rllba_model = RLLBAModel_2A(hierarchical_levels=1)
    if len([file for file in os.listdir(pkl_path) if re.search('-RLLBA', file)]):
        print("Success - Test RLLBA pkl model existence")
    else:
        print("Failure - Test RLLBA pkl model existence")


def test_hierRLLBA_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    hier_rllba_model = RLLBAModel_2A(hierarchical_levels=2)
    if len([file for file in os.listdir(pkl_path) if re.search('hierRLLBA', file)]):
        print("Success - Test hierRLLBA pkl model existence")
    else:
        print("Failure - Test hierRLLBA pkl model existence")
