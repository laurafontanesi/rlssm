data {
	int<lower=1> N;									// number of data items

	int<lower=-1,upper=1> accuracy[N];				// accuracy (-1, 1)
	int<lower=0,upper=1> accuracy_flipped[N];		// flipped accuracy (1, 0)
	real<lower=0> rt[N];							// rt
	real beta[N];									// regression coefficients from neuroimaging

	vector[2] drift_trialmu_priors;					// mean and sd of the prior
	vector[2] threshold_priors;						// mean and sd of the prior
	vector[2] ndt_priors;							// mean and sd of the prior
	vector[2] rel_sp_trialmu_priors;				// mean and sd of the prior
	vector[2] drift_trialsd_priors;					// mean and sd of the prior
	vector[2] rel_sp_trialsd_priors;				// mean and sd of the prior
	vector[2] regression_coefficients_priors;		// mean and sd of the prior
}
parameters {
	real drift_trialmu;
	real threshold;
	real ndt;
	real rel_sp_trialmu;
	real<lower=0> drift_trialsd;
	real<lower=0> rel_sp_trialsd;
	real z_drift_trial[N];
	real z_rel_sp_trial[N];
	real drift_coefficient;
	real rel_sp_coefficient;
}
transformed parameters {
	real drift_ll[N];								// trial-by-trial drift rate for likelihood (incorporates accuracy)
	real drift_t[N];								// trial-by-trial drift rate for predictions
	real<lower=0> threshold_t[N];					// trial-by-trial threshold   
	real<lower=0> ndt_t[N];							// trial-by-trial ndt
	real<lower=0, upper=1> rel_sp_ll[N];			// trial-by-trial relative starting point for likelihood (incorporates accuracy)
	real<lower=0, upper=1> rel_sp_t[N];				// trial-by-trial relative starting point
	real beta_t[N];

	real transf_drift_trialmu;
	real transf_threshold;
	real transf_ndt;
	real transf_rel_sp_trialmu;
	real<lower=0> transf_drift_trialsd;
	real<lower=0> transf_rel_sp_trialsd;
	real transf_drift_coefficient;
	real transf_rel_sp_coefficient;

	transf_drift_trialmu = drift_trialmu;			// for the output
	transf_threshold = log(1 + exp(threshold));
	transf_ndt = log(1 + exp(ndt));
	transf_rel_sp_trialmu = Phi(rel_sp_trialmu);
	transf_drift_trialsd = drift_trialsd;
	transf_rel_sp_trialsd = rel_sp_trialsd;
	transf_drift_coefficient = drift_coefficient;
	transf_rel_sp_coefficient = rel_sp_coefficient;

	for (n in 1:N) {
		drift_t[n] = drift_trialmu + z_drift_trial[n]*drift_trialsd + drift_coefficient*beta[n];
		drift_ll[n] = drift_t[n]*accuracy[n];
		threshold_t[n] = transf_threshold;
		ndt_t[n] = transf_ndt;
		rel_sp_t[n] = Phi(rel_sp_trialmu + z_rel_sp_trial[n]*rel_sp_trialsd + rel_sp_coefficient*beta[n]);
		rel_sp_ll[n] = accuracy_flipped[n] + accuracy[n]*rel_sp_t[n];
		beta_t[n] = beta[n];
	}
}
model {
	drift_trialmu ~ normal(drift_trialmu_priors[1], drift_trialmu_priors[2]);
	threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	ndt ~ normal(ndt_priors[1], ndt_priors[2]);
	rel_sp_trialmu ~ normal(rel_sp_trialmu_priors[1], rel_sp_trialmu_priors[2]);

	drift_trialsd ~ normal(drift_trialsd_priors[1], drift_trialsd_priors[2]);
	z_drift_trial ~ normal(0, 1);

	rel_sp_trialsd ~ normal(rel_sp_trialsd_priors[1], rel_sp_trialsd_priors[2]);
	z_rel_sp_trial ~ normal(0, 1);
	
	drift_coefficient ~ normal(regression_coefficients_priors[1], regression_coefficients_priors[2]);
	rel_sp_coefficient ~ normal(regression_coefficients_priors[1], regression_coefficients_priors[2]);

	rt ~ wiener(threshold_t, ndt_t, rel_sp_ll, drift_ll);
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = wiener_lpdf(rt[n] | threshold_t[n], ndt_t[n], rel_sp_ll[n], drift_ll[n]);
	}
}
}