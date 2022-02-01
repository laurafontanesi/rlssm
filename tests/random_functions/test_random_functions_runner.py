from tests.random_functions.test_random_DDM import test_random_DDM
from tests.random_functions.test_random_LBA import test_random_LBA
from tests.random_functions.test_random_RDM import test_random_RDM
from tests.random_functions.test_random_RL import test_random_RL
from tests.random_functions.test_random_RLDDM import test_random_RLDDM
from tests.random_functions.test_random_RLRDM import test_random_RLRDM


def test_random_functions(print_results=True):
    print("Running the random functions tests")
    print("----------------------------------")

    tests_to_run = [
        test_random_DDM,
        test_random_LBA,  # needs work
        test_random_RDM,  # needs work
        test_random_RL,
        test_random_RLDDM,
        test_random_RLRDM
    ]

    total_tests = len(tests_to_run)
    success_tests_ran = 0

    failing_tests = []

    for t in tests_to_run:
        try:
            t(print_results)
            print(f"Successfully ran the test: {t.__name__}")
            success_tests_ran += 1
        except AssertionError as aerr:
            failing_tests.append(t.__name__)
            print(f"{t.__name__}: AssertionError occurred!!! {aerr}")

    print(f"Random functions tests: Successfully ran {success_tests_ran}/{total_tests} tests")
    print("----------------------------------")

    return failing_tests
