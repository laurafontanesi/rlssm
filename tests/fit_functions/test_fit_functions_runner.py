import os
from tests.fit_functions.test_fit_experiment_data import test_fit_experiment_data


def test_fit_functions(print_results=True):
    print("Running the fit functions tests")
    print("----------------------------------")

    tests_to_run = [
        test_fit_experiment_data
    ]

    total_tests = len(tests_to_run)
    success_tests_ran = 0

    for t in tests_to_run:
        try:
            t(print_results)
            success_tests_ran += 1
        except AssertionError as aerr:
            print(f"{t.__name__}: AssertionError occurred!!! {aerr}")

    print(f"Fit functions tests: Succesfully ran {success_tests_ran}/{total_tests} tests")
    print("----------------------------------")
