import os

from tests.model_creation.test_ALBA_model import test_ALBA_model
from tests.model_creation.test_ARDM_model import test_ARDM_model
from tests.model_creation.test_DDM_model import test_DDM_model
from tests.model_creation.test_LBA_model import test_LBA_model
from tests.model_creation.test_RDM_model import test_RDM_model
from tests.model_creation.test_RLALBA_model import test_RLALBA_model
from tests.model_creation.test_RLARDM_model import test_RLARDM_model
from tests.model_creation.test_RLDDM_model import test_RLDDM_model
from tests.model_creation.test_RLLBA_model import test_RLLBA_model
from tests.model_creation.test_RLRDM_model import test_RLRDM_model
from tests.model_creation.test_RL_model import test_RL_model


def test_model_creation(print_results=True):
    print("Running the model creation tests")
    print("----------------------------------")

    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)

    tests_to_run = [
        test_ALBA_model,
        test_ARDM_model,
        test_DDM_model,
        test_LBA_model,
        test_RDM_model,
        test_RL_model,
        test_RLALBA_model,
        test_RLARDM_model,
        test_RLDDM_model,
        test_RLLBA_model,
        test_RLRDM_model
    ]

    success_tests_ran = 0
    total_tests = 2 * len(tests_to_run)

    for t in tests_to_run:
        # Test creating the non-hier models
        try:
            t(hier_levels=1, print_results=print_results)
            success_tests_ran += 1
        except Exception as exc:
            print(f"Non hier model creation failed: {exc}")

        # Test creating the hier models
        try:
            t(hier_levels=2, print_results=print_results)
            success_tests_ran += 1
        except Exception as exc:
            print(f"Hier model creation failed: {exc}")

    print(f"Model creation tests: Successfully ran {success_tests_ran}/{total_tests} tests")
    print("----------------------------------")
