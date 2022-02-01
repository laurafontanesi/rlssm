import os

from tests.model_creation.test_model_ALBA import test_model_ALBA
from tests.model_creation.test_model_ARDM import test_model_ARDM
from tests.model_creation.test_model_DDM import test_model_DDM
from tests.model_creation.test_model_LBA import test_model_LBA
from tests.model_creation.test_model_RDM import test_model_RDM
from tests.model_creation.test_model_RL import test_model_RL
from tests.model_creation.test_model_RLALBA import test_model_RLALBA
from tests.model_creation.test_model_RLARDM import test_model_RLARDM
from tests.model_creation.test_model_RLDDM import test_model_RLDDM
from tests.model_creation.test_model_RLLBA import test_model_RLLBA
from tests.model_creation.test_model_RLRDM import test_model_RLRDM


def test_model_creation(print_results=True):
    print("Running the model creation tests")
    print("----------------------------------")

    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)

    tests_to_run = [
        test_model_ALBA,
        test_model_ARDM,
        test_model_DDM,
        test_model_LBA,
        test_model_RDM,
        test_model_RL,
        test_model_RLALBA,
        test_model_RLARDM,
        test_model_RLDDM,
        test_model_RLLBA,
        test_model_RLRDM
    ]

    success_tests_ran = 0
    total_tests = 2 * len(tests_to_run)

    failing_tests = []

    for t in tests_to_run:
        # Test creating the non-hier models
        try:
            t(hier_levels=1, print_results=print_results)
            success_tests_ran += 1
        except Exception as exc:
            failing_tests.append(t.__name__)
            print(f"{t.__name__}: Non hier model creation failed: {exc}")

        # Test creating the hier models
        try:
            t(hier_levels=2, print_results=print_results)
            success_tests_ran += 1
        except Exception as exc:
            failing_tests.append(t.__name__)
            print(f"{t.__name__}: Hier model creation failed: {exc}")

    print(f"Model creation tests: Successfully ran {success_tests_ran}/{total_tests} tests")
    print("----------------------------------")

    return failing_tests
