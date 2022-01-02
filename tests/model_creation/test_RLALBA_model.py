import os
import re

from rlssm import RLALBAModel_2A


def test_RLALBA_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    rlalba_model = RLALBAModel_2A(hierarchical_levels=1)
    if len([file for file in os.listdir(pkl_path) if re.search('-RLALBA', file)]):
        print("Success - Test RLALBA pkl model existence")
    else:
        print("Failure - Test RLALBA pkl model existence")


def test_hierRLALBA_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    hier_rlalba_model = RLALBAModel_2A(hierarchical_levels=2)
    if len([file for file in os.listdir(pkl_path) if re.search('hierRLALBA', file)]):
        print("Success - Test hierRLALBA pkl model existence")
    else:
        print("Failure - Test hierRLALBA pkl model existence")
