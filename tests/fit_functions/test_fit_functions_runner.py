from tests.fit_functions.test_fit_DDM import test_fit_DDM
from tests.fit_functions.test_fit_LBA import test_fit_LBA
from tests.fit_functions.test_fit_RDM import test_fit_RDM
from tests.fit_functions.test_fit_RL import test_fit_RL


def test_fit_functions(print_results=True, test_hier=True):
    print("Running the fit functions tests")
    print("----------------------------------")

    tests_to_run = [
        test_fit_DDM,
        test_fit_LBA,
        test_fit_RDM,
        test_fit_RL
    ]

    total_tests = len(tests_to_run)
    if test_hier:
        total_tests *= 2
    success_tests_ran = 0

    failing_tests = []

    for t in tests_to_run:
        # Test fitting the non-hier models
        try:
            hier_levels = 1
            t(hier_levels=hier_levels, print_results=print_results)
            print(f"Successfully ran the test: {t.__name__}, hier_levels={hier_levels}")
            success_tests_ran += 1
        except Exception as exc:
            failing_tests.append(t.__name__)
            print(f"{t.__name__}: Non hier model fitting failed: {exc}")

        if test_hier:
            # Test fitting the hier models
            try:
                hier_levels = 2
                t(hier_levels=hier_levels, print_results=print_results, test_hier=test_hier)
                print(f"Successfully ran the test: {t.__name__}, hier_levels={hier_levels}")
                success_tests_ran += 1
            except Exception as exc:
                failing_tests.append(t.__name__)
                print(f"{t.__name__}: Hier model fitting failed: {exc}")

    print(f"Fit functions tests: Successfully ran {success_tests_ran}/{total_tests} tests")
    print("----------------------------------")

    return failing_tests
