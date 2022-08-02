data {
	int<lower=1> N;									// number of data items

	int<lower=-1,upper=1> accuracy[N];				// accuracy (-1, 1)
	int<lower=0,upper=1> accuracy_flipped[N];		// flipped accuracy (1, 0)
	real<lower=0> rt[N];							// rt

	vector[2] drift_priors;							// mean and sd of the prior
	vector[2] threshold_priors;						// mean and sd of the prior
	vector[2] ndt_priors;							// mean and sd of the prior
	vector[2] rel_sp_trialsd_priors;				// mean and sd of the prior

	real<lower=0, upper=1> starting_point;			// starting point diffusion model not to estimate
}
parameters {
	real drift;
	real threshold;
	real ndt;
	real<lower=0> rel_sp_trialsd;
	real z_rel_sp_trial[N];
}
transformed parameters {
	real drift_ll[N];								// trial-by-trial drift rate for likelihood (incorporates accuracy)
	real drift_t[N];								// trial-by-trial drift rate for predictions
	real<lower=0> threshold_t[N];					// trial-by-trial threshold
	real<lower=0> ndt_t[N];							// trial-by-trial ndt
	real<lower=0, upper=1> rel_sp_ll[N];			// trial-by-trial relative starting point for likelihood (incorporates accuracy)
	real<lower=0, upper=1> rel_sp_t[N];				// trial-by-trial relative starting point

	real transf_drift;
	real transf_threshold;
	real transf_ndt;
	real<lower=0> transf_rel_sp_trialsd;

	transf_drift = drift;							// for the output
	transf_threshold = log(1 + exp(threshold));
	transf_ndt = log(1 + exp(ndt));
	transf_rel_sp_trialsd = rel_sp_trialsd;

	for (n in 1:N) {
		drift_t[n] = drift;
		drift_ll[n] = drift_t[n]*accuracy[n];
		threshold_t[n] = transf_threshold;
		ndt_t[n] = transf_ndt;
		rel_sp_t[n] = Phi(starting_point + z_rel_sp_trial[n]*rel_sp_trialsd);
		rel_sp_ll[n] = accuracy_flipped[n] + accuracy[n]*rel_sp_t[n];
	}
}
model {
	drift ~ normal(drift_priors[1], drift_priors[2]);
	threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	ndt ~ normal(ndt_priors[1], ndt_priors[2]);

	rel_sp_trialsd ~ normal(rel_sp_trialsd_priors[1], rel_sp_trialsd_priors[2]);
	z_rel_sp_trial ~ normal(0, 1);

	rt ~ wiener(threshold_t, ndt_t, rel_sp_ll, drift_ll);
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = wiener_lpdf(rt[n] | threshold_t[n], ndt_t[n], rel_sp_ll, drift_ll[n]);
	}
}
}