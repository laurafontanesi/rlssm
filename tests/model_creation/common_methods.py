import os
import re


def check_pkl_file_existence(model_name, hier_levels):
    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')
    if hier_levels == 1:
        pkl_file_counter = len([file for file in os.listdir(pkl_path) if re.search(f'-{model_name}', file)])
    else:
        pkl_file_counter = len([file for file in os.listdir(pkl_path) if re.search(f'hier{model_name}', file)])

    assert pkl_file_counter > 0, f"Failure - Test {model_name} hier_levels={hier_levels} model creation"
    print(f"Success - Test {model_name} hier_levels = {hier_levels} model creation")
