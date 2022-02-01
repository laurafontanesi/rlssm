import numpy as np

from tests.fit_functions.test_fit_functions_runner import test_fit_functions
from tests.model_creation.test_model_creation_runner import test_model_creation
from tests.plot_functions.test_plot_functions_runner import test_plot_functions
from tests.random_functions.test_random_functions_runner import test_random_functions


def run_all_tests():
    print_results = True
    test_hier = False

    failing_tests = []

    failing_tests.append(test_fit_functions(print_results=print_results, test_hier=test_hier))
    failing_tests.append(test_model_creation(print_results=print_results))
    failing_tests.append(test_random_functions(print_results=print_results))
    failing_tests.append(test_plot_functions(print_results=print_results))

    print(f"The failing tests are: {np.array(failing_tests).flatten()}")


if __name__ == '__main__':
    run_all_tests()
