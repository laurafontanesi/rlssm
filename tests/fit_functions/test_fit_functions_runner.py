import os
from tests.fit_functions.test_fit_experiment_data import test_fit_experiment_data


def test_fit_functions(print_results=True):
    print("Running the fit functions tests")
    print("----------------------------------")

    testing_dir_path = os.path.dirname(os.path.realpath(__file__))
    success_tests_ran = total_tests = len([item for item in os.listdir(testing_dir_path) if '.py' in item]) - 1

    tests_to_run = [
        test_fit_experiment_data
    ]

    for t in tests_to_run:
        try:
            t(print_results)
        except AssertionError as aerr:
            print(f"{t.__name__}: AssertionError occurred!!! {aerr}")
            success_tests_ran -= 1

    print(f"Fit functions tests: Succesfully ran {success_tests_ran}/{total_tests} tests")
    print("----------------------------------")
