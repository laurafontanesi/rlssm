import os
import re
from rlssm import LBAModel_2A


def test_LBA_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    lba_model = LBAModel_2A(hierarchical_levels=1)
    if len([file for file in os.listdir(pkl_path) if re.search('-LBA', file)]):
        print("Success - Test LBA pkl model existence")
    else:
        print("Failure - Test LBA pkl model existence")


def test_hierLBA_model(print_results=True):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')

    hier_lba_model = LBAModel_2A(hierarchical_levels=2)
    if len([file for file in os.listdir(pkl_path) if re.search('hierLBA', file)]):
        print("Success - Test hierLBA pkl model existence")
    else:
        print("Failure - Test hierLBA pkl model existence")