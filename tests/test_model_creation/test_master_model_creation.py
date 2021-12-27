import os

print("Running the model creation tests")
print("----------------------------------")

no_tests_ran = 0

curr_dir = os.path.dirname(os.path.realpath(__file__))

# Run all tests in the current folder
for file in os.listdir(curr_dir):
    if file.endswith(".py"):
        if file != os.path.basename(__file__):
            print(f"Currently running the test: {file}")
            os.system("python " + os.path.join(curr_dir, file))
            no_tests_ran += 1

print(f"Successfully ran {no_tests_ran} tests for model creation")
print("----------------------------------")
