import os
import re

from rlssm import ARDModel_2A

__dir__ = os.path.abspath(os.path.dirname(__file__))
path_pkl_fldr = os.path.join(os.path.dirname(os.path.dirname(__dir__)), "pkl_files")
if not os.path.exists(path_pkl_fldr):
    os.makedirs(path_pkl_fldr)

ardm_model = ARDModel_2A(hierarchical_levels=2)
if len([file for file in os.listdir(path_pkl_fldr) if re.search('ARDM', file)]):
    print("Success - Test ARDM pkl model existence")
else:
    print("Failure - Test ARDM pkl model existence")
