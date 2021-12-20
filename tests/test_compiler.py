import rlssm

model = rlssm.DDModel(hierarchical_levels=2)
data = rlssm.load_example_dataset(hierarchical_levels=1)

print("done")