import os
import re

from rlssm import LBAModel_2A

__dir__ = os.path.abspath(os.path.dirname(__file__))
path_pkl_fldr = os.path.join(os.path.dirname(os.path.dirname(__dir__)), "pkl_files")

lba_model = LBAModel_2A(hierarchical_levels=2)
if len([file for file in os.listdir(path_pkl_fldr) if re.search('hierLBA', file) or re.search('-LBA', file)]):
    print("Success - Test LBA pkl model existence")
else:
    print("Failure - Test LBA pkl model existence")
