import os
import re
import sys


def check_pkl_file_existence(model_name, hier_levels):
    # this path should be rlssm/pkl_files
    pkl_path = os.path.join(sys.path[1], 'pkl_files')

    if hier_levels == 1:
        pkl_file_counter = len([file for file in os.listdir(pkl_path) if re.search(f'-{model_name}', file)])
    else:
        pkl_file_counter = len([file for file in os.listdir(pkl_path) if re.search(f'hier{model_name}', file)])

    assert pkl_file_counter > 0, f"Failure - Test {model_name} hier_levels={hier_levels} model creation"
    print(f"Success - Test {model_name} hier_levels = {hier_levels} model creation")
