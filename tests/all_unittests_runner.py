import time
import unittest

passedTests = set()
failedTests = set()


def run_tests(ignore_hier_tests=False):
    test_dir = 'unit_tests/'
    discovered_tests = unittest.defaultTestLoader.discover(test_dir)
    print(f"Running tests in {test_dir}; : Number of test discovered: {discovered_tests.countTestCases()}")
    print(f"Ignoring hierarchical version of all tests: {ignore_hier_tests}")

    for allUnitTests in discovered_tests:
        for tests in allUnitTests:
            for test in tests:
                test_name = str(test).split(' ')[0]
                if ignore_hier_tests and test_name.endswith("hier"):
                    print(f"Ignore test: {test_name}")
                    continue
                else:
                    print(f"Running test: {test_name}")
                    start_time = time.time()
                    result = test.run()
                    print(f"Ran {test_name} in {(time.time() - start_time):.2f} sec. Result: {result}")
                    if len(result.failures) == 0:
                        passedTests.add(test_name)
                    else:
                        failedTests.add(test_name)


start_total_time = time.time()
run_tests(ignore_hier_tests=True)
print(f"Finished running the tests in {(time.time() - start_total_time):.2f} sec. See summary below...")
print(f"\tPASSING tests: {len(passedTests)} ({passedTests})")
print(f"\tFAILING tests: {len(failedTests)} ({failedTests})")
