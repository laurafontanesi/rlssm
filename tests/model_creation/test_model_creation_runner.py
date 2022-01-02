import os

from tests.model_creation.test_ALBA_model import test_ALBA_model
from tests.model_creation.test_ARDM_model import test_ARDM_model
from tests.model_creation.test_DDM_model import test_DDM_model
from tests.model_creation.test_LBA_model import test_LBA_model
from tests.model_creation.test_RDM_model import test_RDM_model
from tests.model_creation.test_RL_model import test_RL_model


def test_model_creation(print_results=True):
    print("Running the model creation tests")
    print("----------------------------------")

    testing_dir_path = os.path.dirname(os.path.realpath(__file__))
    success_tests_ran = total_tests = len([item for item in os.listdir(testing_dir_path) if '.py' in item]) - 1

    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)

    tests_to_run = [
        test_ALBA_model,
        test_ARDM_model,
        test_DDM_model,
        test_LBA_model,
        test_RDM_model,
        test_RL_model
    ]

    for t in tests_to_run:
        try:
            t(print_results)
        except Exception as exc:
            print(f"Model creation tests: Exception occurred: {exc}")
            success_tests_ran -= 1

    print(f"Model creation tests: Succesfully ran {success_tests_ran}/{total_tests} tests")
    print("----------------------------------")
