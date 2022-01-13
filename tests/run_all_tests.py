from tests.fit_functions.test_fit_functions_runner import test_fit_functions
from tests.model_creation.test_model_creation_runner import test_model_creation
from tests.plot_functions.test_plot_functions_runner import test_plot_functions
from tests.random_functions.test_random_functions_runner import test_random_functions


def run_all_tests():
    print_results = True

    test_fit_functions(print_results)
    test_model_creation(print_results)
    test_random_functions(print_results)
    test_plot_functions(print_results)


if __name__ == '__main__':
    run_all_tests()
