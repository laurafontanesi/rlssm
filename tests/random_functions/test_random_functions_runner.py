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

    try:
        test_generate_dm_hier(print_results)
        test_generate_dm_non_hier(print_results)
        test_hier_DDM(print_results)
        test_hier_RL_2A(print_results)
        test_hier_RL_2alpha(print_results)
        test_RL_DDM_2_alpha(print_results)
        test_RL_DDM_2A(print_results)
        test_RL_DDM_hier_2alpha(print_results)
        test_RL_RDM_non_hier(print_results)
        test_simple_DDM(print_results)
        test_simple_RL(print_results)
        test_simple_RL_2alpha(print_results)
    except Exception as exc:
        print(f"Random functions tests: Exception occurred: {exc}")
        success_tests_ran -= 1
    finally:
        print(f"Random functions tests: Succesfully ran {success_tests_ran}/{total_tests} tests")
        print("----------------------------------")
