import rlssm
import pandas as pd
import os

par_path = os.getcwd()
data_path = os.path.join(par_path, 'data/data_experiment.csv')

data = pd.read_csv(data_path, index_col=0)

data = data[data.participant == 20].reset_index(drop=True)

data['block_label'] += 1

print(data)

model = rlssm.DDModel(hierarchical_levels = 1)

# sampling parameters
n_iter = 1000
n_chains = 2
n_thin = 1

model_fit = model.fit(
    data,
    thin = n_thin,
    iter = n_iter,
    chains = n_chains,
    pointwise_waic=False,
    verbose = False)

print(model_fit.data_info['data'])