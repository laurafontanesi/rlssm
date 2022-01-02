import os
from tests.fit_functions.test_fit_experiment_data import test_fit_experiment_data


def test_fit_functions(print_results=True):
    print("Running the fit functions tests")
    print("----------------------------------")

    testing_dir_path = os.path.dirname(os.path.realpath(__file__))
    success_tests_ran = total_tests = len([item for item in os.listdir(testing_dir_path) if '.py' in item]) - 1

    try:
        test_fit_experiment_data(print_results)
    except Exception as exc:
        print(f"Fit functions tests: Exception occurred: {exc}")
        success_tests_ran -= 1
    finally:
        print(f"Fit functions tests: Succesfully ran {success_tests_ran}/{total_tests} tests")
        print("----------------------------------")
