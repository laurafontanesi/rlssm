from tests.plot_functions.test_plot_DDM import test_DDM_plot_posterior
from tests.plot_functions.test_plot_LBA import test_LBA_plot_posterior
from tests.plot_functions.test_plot_RDM import test_RDM_plot_posterior
from tests.plot_functions.test_plot_RL import test_RL_plot_posterior


def test_plot_functions(print_results=True):
    print("Running the plot functions tests")
    print("----------------------------------")

    tests_to_run = [
        test_DDM_plot_posterior,
        test_RL_plot_posterior,
        test_LBA_plot_posterior,
        test_RDM_plot_posterior
    ]

    total_tests = len(tests_to_run)
    success_tests_ran = 0

    failing_tests = []

    for t in tests_to_run:
        try:
            t(print_results)
            success_tests_ran += 1
        except AssertionError as aerr:
            failing_tests.append(t.__name__)
            print(f"{t.__name__}: AssertionError occurred!!! {aerr}")

    print(f"Plot functions tests: Successfully ran {success_tests_ran}/{total_tests} tests")
    print("----------------------------------")

    return failing_tests
