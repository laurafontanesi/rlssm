import os
import glob

print("Running the random functions tests")
print("----------------------------------")

no_tests_ran = 0

# Run all tests in the current folder
for file in glob.iglob("*.py"):
    if file != os.path.basename(__file__):
        print(f"Currently running the test: {file}")
        os.system("python3 " + file)
        no_tests_ran += 1

print(f"Successfully ran {no_tests_ran} tests for testing random functions")
