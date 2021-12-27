data {
	int<lower=1> N;									// number of data items
	int<lower=1> L;									// number of levels
	int<lower=1, upper=L> participant[N];			// level (participant)

	int<lower=-1,upper=1> accuracy[N];				// accuracy (-1, 1)
	real<lower=0> rt[N];							// rt

	vector[4] drift_trialmu_priors;					// mean and sd of the prior
	vector[4] threshold_priors;						// mean and sd of the prior
	vector[4] ndt_priors;							// mean and sd of the prior
	vector[4] drift_trialsd_priors;					// mean and sd of the prior
	real<lower=0, upper=1> starting_point;			// starting point diffusion model not to estimate
}
parameters {
	real mu_drift_trialmu;
	real mu_drift_trialsd;
	real mu_threshold;
	real mu_ndt;

	real<lower=0> sd_drift_trialmu;
	real<lower=0> sd_drift_trialsd;
	real<lower=0> sd_threshold;
	real<lower=0> sd_ndt;

	real z_drift_trialmu[L];
	real z_drift_trialsd[L];
	real z_threshold[L];
	real z_ndt[L];

	real z_drift_trial[N];
}
transformed parameters {
	real drift_ll[N];								// trial-by-trial drift rate for likelihood (incorporates accuracy)
	real drift_t[N];								// trial-by-trial drift rate for predictions
	real<lower=0> threshold_t[N];					// trial-by-trial threshold
	real<lower=0> ndt_t[N];							// trial-by-trial ndt

	real drift_trialmu_sbj[L];
	real<lower=0> drift_trialsd_sbj[L];
	real<lower=0> threshold_sbj[L];
	real<lower=0> ndt_sbj[L];

	real transf_mu_drift_trialmu;
	real transf_mu_drift_trialsd;
	real transf_mu_threshold;
	real transf_mu_ndt;

	transf_mu_drift_trialmu = mu_drift_trialmu;		// for the output
	transf_mu_drift_trialsd = log(1 + exp(mu_drift_trialsd));
	transf_mu_threshold = log(1 + exp(mu_threshold));
	transf_mu_ndt = log(1 + exp(mu_ndt));

	for (l in 1:L) {
		drift_trialmu_sbj[l] = mu_drift_trialmu + z_drift_trialmu[l]*sd_drift_trialmu;
		drift_trialsd_sbj[l] = log(1 + exp(mu_drift_trialsd + z_drift_trialsd[l]*sd_drift_trialsd));
		threshold_sbj[l] = log(1 + exp(mu_threhsold + z_threshold[l]*sd_threshold));
		ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l]*sd_ndt));
	}

	for (n in 1:N) {
		drift_t[n] = drift_trialmu_sbj[participant[n]] + z_drift_trial[n]*drift_trialsd_sbj[participant[n]];
		drift_ll[n] = drift_t[n]*accuracy[n];
		threshold_t[n] = threshold_sbj[participant[n]];
		ndt_t[n] = ndt_sbj[participant[n]];
	}
}
model {
	mu_drift_trialmu ~ normal(drift_trialmu_priors[1], drift_trialmu_priors[2]);
	mu_drift_trialsd ~ normal(drift_trialsd_priors[1], drift_trialsd_priors[2]);
	mu_threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	mu_ndt ~ normal(ndt_priors[1], ndt_priors[2]);

	sd_drift_trialmu ~ normal(drift_trialmu_priors[3], drift_trialmu_priors[4]);
	sd_drift_trialsd ~ normal(drift_trialsd_priors[3], drift_trialsd_priors[4]);
	sd_threshold ~ normal(threshold_priors[3], threshold_priors[4]);
	sd_ndt ~ normal(ndt_priors[3], ndt_priors[4]);

	z_drift_trialmu ~ normal(0, 1);
	z_drift_trialsd ~ normal(0, 1);
	z_threshold ~ normal(0, 1);
	z_ndt ~ normal(0, 1);

	z_drift_trial ~ normal(0, 1);

	rt ~ wiener(threshold_t, ndt_t, starting_point, drift_ll);
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = wiener_lpdf(rt[n] | threshold_t[n], ndt_t[n], starting_point, drift_ll[n]);
	}
}
}