from tests.plot_functions.test_plot_posterior import test_plot_posterior


def test_plot_functions(print_results=True):
    print("Running the plot functions tests")
    print("----------------------------------")

    tests_to_run = [
        test_plot_posterior
    ]

    total_tests = len(tests_to_run)
    success_tests_ran = 0

    for t in tests_to_run:
        try:
            t(print_results)
            success_tests_ran += 1
        except AssertionError as aerr:
            print(f"{t.__name__}: AssertionError occurred!!! {aerr}")

    print(f"Plot functions tests: Successfully ran {success_tests_ran}/{total_tests} tests")
    print("----------------------------------")
