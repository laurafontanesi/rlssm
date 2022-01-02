import os

from tests.random_functions.test_RL_DDM_2A import test_RL_DDM_2A
from tests.random_functions.test_RL_DDM_2_alpha import test_RL_DDM_2_alpha
from tests.random_functions.test_RL_DDM_hier_2alpha import test_RL_DDM_hier_2alpha
from tests.random_functions.test_RL_RDM_non_hier import test_RL_RDM_non_hier
from tests.random_functions.test_generate_dm_hier import test_generate_dm_hier
from tests.random_functions.test_generate_dm_non_hier import test_generate_dm_non_hier
from tests.random_functions.test_hier_DDM import test_hier_DDM
from tests.random_functions.test_hier_RL_2A import test_hier_RL_2A
from tests.random_functions.test_hier_RL_2alpha import test_hier_RL_2alpha
from tests.random_functions.test_simple_DDM import test_simple_DDM
from tests.random_functions.test_simple_RL import test_simple_RL
from tests.random_functions.test_simple_RL_2alpha import test_simple_RL_2alpha


def test_random_functions(print_results=True):
    print("Running the random functions tests")
    print("----------------------------------")

    testing_dir_path = os.path.dirname(os.path.realpath(__file__))
    success_tests_ran = total_tests = len([item for item in os.listdir(testing_dir_path) if '.py' in item]) - 1

    tests_to_run = [
        test_generate_dm_hier,
        test_generate_dm_non_hier,
        test_hier_DDM,
        test_hier_RL_2A,
        test_hier_RL_2alpha,
        test_RL_DDM_2_alpha,
        test_RL_DDM_2A,
        test_RL_DDM_hier_2alpha,
        test_RL_RDM_non_hier,
        test_simple_DDM,
        test_simple_RL,
        test_simple_RL_2alpha
    ]

    for t in tests_to_run:
        try:
            t(print_results)
        except AssertionError as aerr:
            print(f"{t.__name__}: AssertionError occurred!!! {aerr}")
            success_tests_ran -= 1

    print(f"Random functions tests: Succesfully ran {success_tests_ran}/{total_tests} tests")
    print("----------------------------------")
