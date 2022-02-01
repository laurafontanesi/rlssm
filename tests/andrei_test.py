from rlssm import load_example_dataset, RDModel_2A

model = RDModel_2A(hierarchical_levels=2)

data = load_example_dataset(hierarchical_levels=2)

# to make the hier test work faster, only take the first 10 participants into consideration
data_hier = data[data['participant'] <= 10]

drift_priors = {'mu_mu': 1, 'sd_mu': 1, 'mu_sd': 0, 'sd_sd': 1}
threshold_priors = {'mu_mu': -1, 'sd_mu': .5, 'mu_sd': 0, 'sd_sd': 1}
ndt_priors = {'mu_mu': 1, 'sd_mu': 1, 'mu_sd': 0, 'sd_sd': 1}

model_fit = model.fit(data_hier,
                      drift_priors=drift_priors,
                      threshold_priors=threshold_priors,
                      ndt_priors=ndt_priors,
                      warmup=50,
                      iter=200,
                      chains=2,
                      verbose=False)
