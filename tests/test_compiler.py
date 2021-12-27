from rlssm import DDModel, load_example_dataset

model = DDModel(hierarchical_levels=2)
data = load_example_dataset(hierarchical_levels=1)

print("done")