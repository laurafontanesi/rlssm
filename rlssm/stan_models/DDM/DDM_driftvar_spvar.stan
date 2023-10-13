data {
	int<lower=1> N;									// number of data items

	array[N] int<lower=-1,upper=1> accuracy;			// accuracy (-1, 1)
	array[N] int<lower=0,upper=1> accuracy_flipped;		// flipped accuracy (1, 0)
	array[N] real<lower=0> rt;							// rt

	vector[2] drift_trial_mu_priors;				// mean and sd of the prior
	vector[2] threshold_priors;						// mean and sd of the prior
	vector[2] ndt_priors;							// mean and sd of the prior
	vector[2] drift_trial_sd_priors;				// mean and sd of the prior
	vector[2] rel_sp_trial_sd_priors;				// mean and sd of the prior
	
	real<lower=0, upper=1> starting_point;			// starting point diffusion model not to estimate
}
parameters {
	real drift_trial_mu;
	real threshold;
	real ndt;
	real<lower=0> drift_trial_sd;
	real<lower=0> rel_sp_trial_sd;
	array[N] real z_drift_trial;
	array[N] real z_rel_sp_trial;
}
transformed parameters {
	array[N] real drift_ll;								// trial-by-trial drift rate for likelihood (incorporates accuracy)
	array[N] real drift_t;								// trial-by-trial drift rate for predictions
	array[N] real<lower=0> threshold_t;					// trial-by-trial threshold   
	array[N] real<lower=0> ndt_t;							// trial-by-trial ndt
	array[N] real<lower=0, upper=1> rel_sp_ll;			// trial-by-trial relative starting point for likelihood (incorporates accuracy)
	array[N] real<lower=0, upper=1> rel_sp_t;				// trial-by-trial relative starting point

	real transf_drift_trial_mu;
	real transf_threshold;
	real transf_ndt;
	real<lower=0> transf_drift_trial_sd;
	real<lower=0> transf_rel_sp_trial_sd;

	transf_drift_trial_mu = drift_trial_mu;			// for the output
	transf_threshold = log(1 + exp(threshold));
	transf_ndt = log(1 + exp(ndt));
	transf_drift_trial_sd = drift_trial_sd;
	transf_rel_sp_trial_sd = rel_sp_trial_sd;

	for (n in 1:N) {
		drift_t[n] = drift_trial_mu + z_drift_trial[n]*drift_trial_sd;
		drift_ll[n] = drift_t[n]*accuracy[n];
		threshold_t[n] = transf_threshold;
		ndt_t[n] = transf_ndt;
		rel_sp_t[n] = Phi(inv_Phi(starting_point) + z_rel_sp_trial[n]*rel_sp_trial_sd);
		rel_sp_ll[n] = accuracy_flipped[n] + accuracy[n]*rel_sp_t[n];
	}
}
model {
	drift_trial_mu ~ normal(drift_trial_mu_priors[1], drift_trial_mu_priors[2]);
	threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	ndt ~ normal(ndt_priors[1], ndt_priors[2]);

	drift_trial_sd ~ normal(drift_trial_sd_priors[1], drift_trial_sd_priors[2]);
	z_drift_trial ~ normal(0, 1);

	rel_sp_trial_sd ~ normal(rel_sp_trial_sd_priors[1], rel_sp_trial_sd_priors[2]);
	z_rel_sp_trial ~ normal(0, 1);

	rt ~ wiener(threshold_t, ndt_t, rel_sp_ll, drift_ll);
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = wiener_lpdf(rt[n] | threshold_t[n], ndt_t[n], rel_sp_ll[n], drift_ll[n]);
	}
}
}