from rlssm.model.models_DDM import DDModel

"""
This is a simple testing script; run it to check if the installation is successful
If installed correctly, the output should have the form: 
    {'drift_priors': {'mu_mu': 1, 'sd_mu': 5, 'mu_sd': 0, 'sd_sd': 5}, 
    'threshold_priors': {'mu_mu': 1, 'sd_mu': 3, 'mu_sd': 0, 'sd_sd': 3}, 
    'ndt_priors': {'mu_mu': 1, 'sd_mu': 1, 'mu_sd': 0, 'sd_sd': 1}}

"""
model = DDModel(hierarchical_levels=2)
priors = model.priors
assert model.priors is not None, "Priors of the model cannot be retrieved"
print(priors)
