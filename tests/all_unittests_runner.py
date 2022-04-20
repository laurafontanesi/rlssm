import unittest

passedTests = set()
failedTests = set()


def run_tests(run_hier_tests=False):
    for allUnitTests in unittest.defaultTestLoader.discover('unit_tests'):
        for tests in allUnitTests:
            for test in tests:
                test_name = str(test).split(' ')[0]
                if run_hier_tests:
                    result = test.run()
                    if result.failures == 0:
                        passedTests.add(test_name)
                    else:
                        failedTests.add(test_name)
                else:
                    if not test_name.endswith("hier"):
                        result = test.run()
                        print(f"{test_name}; results: {result}")
                        if result.failures == 0:
                            passedTests.add(test_name)
                        else:
                            failedTests.add(test_name)


run_tests(run_hier_tests=False)
print(f"Tests passed: {len(passedTests)} ({passedTests})")
print(f"Tests failed: {len(failedTests)} ({failedTests})")
