import rlssm

"""
This is a simple testing script; run it to check if the installation is successful
"""
model = rlssm.DDModel(hierarchical_levels=2)
priors = model.priors
print(priors)

if (i in str(priors) for i in ["drift_priors", "threshold_priors", "ndt_priors", "mu_mu", "sd_sd"]):
    print("Compiler test results: SUCCESS")
else:
    print("Compiler test results: FAIL")