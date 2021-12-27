import os
import re

__dir__ = os.path.abspath(os.path.dirname(__file__))

from rlssm import ALBAModel_2A

path_pkl_fldr = os.path.join(os.path.dirname(os.path.dirname(__dir__)), "pkl_files")
if not os.path.exists(path_pkl_fldr):
    os.makedirs(path_pkl_fldr)

alba_model = ALBAModel_2A(hierarchical_levels=2)
if len([file for file in os.listdir(path_pkl_fldr) if re.search('ALBA', file)]):
    print("Success - Test ALBA pkl model existence")
else:
    print("Failure - Test ALBA pkl model existence")
