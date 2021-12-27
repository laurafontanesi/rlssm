import os
import re

from rlssm import DDModel

__dir__ = os.path.abspath(os.path.dirname(__file__))
path_pkl_fldr = os.path.join(os.path.dirname(os.path.dirname(__dir__)), "pkl_files")
if not os.path.exists(path_pkl_fldr):
    os.makedirs(path_pkl_fldr)

ddm_model = DDModel(hierarchical_levels=2)
if len([file for file in os.listdir(path_pkl_fldr) if re.search('DDM', file)]):
    print("Success - Test DDM pkl model existence")
else:
    print("Failure - Test DDM pkl model existence")
