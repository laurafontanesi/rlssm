data {
	int<lower=1> N;									// number of data items
	int<lower=1> L;									// number of levels
	array[N] int<lower=1, upper=L> participant;			// level (participant)

	array[N] int<lower=-1,upper=1> accuracy;			// accuracy (-1, 1)
	array[N] real<lower=0> rt;							// rt

	vector[4] drift_trial_mu_priors;					// mean and sd of the prior
	vector[4] threshold_priors;						// mean and sd of the prior
	vector[4] ndt_priors;							// mean and sd of the prior
	vector[4] drift_trial_sd_priors;					// mean and sd of the prior
	real<lower=0, upper=1> starting_point;			// starting point diffusion model not to estimate
}
parameters {
	real mu_drift_trial_mu;
	real mu_drift_trial_sd;
	real mu_threshold;
	real mu_ndt;

	real<lower=0> sd_drift_trial_mu;
	real<lower=0> sd_drift_trial_sd;
	real<lower=0> sd_threshold;
	real<lower=0> sd_ndt;

	array[L] real z_drift_trial_mu;
	array[L] real z_drift_trial_sd;
	array[L] real z_threshold;
	array[L] real z_ndt;

	array[N] real z_drift_trial;
}
transformed parameters {
	array[N] real drift_ll;								// trial-by-trial drift rate for likelihood (incorporates accuracy)
	array[N] real drift_t;								// trial-by-trial drift rate for predictions
	array[N] real<lower=0> threshold_t;					// trial-by-trial threshold
	array[N] real<lower=0> ndt_t;						// trial-by-trial ndt

	array[L] real drift_trial_mu_sbj;
	array[L] real<lower=0> drift_trial_sd_sbj;
	array[L] real<lower=0> threshold_sbj;
	array[L] real<lower=0> ndt_sbj;

	real transf_mu_drift_trial_mu;
	real transf_mu_drift_trial_sd;
	real transf_mu_threshold;
	real transf_mu_ndt;

	transf_mu_drift_trial_mu = mu_drift_trial_mu;		// for the output
	transf_mu_drift_trial_sd = log(1 + exp(mu_drift_trial_sd));
	transf_mu_threshold = log(1 + exp(mu_threshold));
	transf_mu_ndt = log(1 + exp(mu_ndt));

	for (l in 1:L) {
		drift_trial_mu_sbj[l] = mu_drift_trial_mu + z_drift_trial_mu[l]*sd_drift_trial_mu;
		drift_trial_sd_sbj[l] = log(1 + exp(mu_drift_trial_sd + z_drift_trial_sd[l]*sd_drift_trial_sd));
		threshold_sbj[l] = log(1 + exp(mu_threhsold + z_threshold[l]*sd_threshold));
		ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l]*sd_ndt));
	}

	for (n in 1:N) {
		drift_t[n] = drift_trial_mu_sbj[participant[n]] + z_drift_trial[n]*drift_trial_sd_sbj[participant[n]];
		drift_ll[n] = drift_t[n]*accuracy[n];
		threshold_t[n] = threshold_sbj[participant[n]];
		ndt_t[n] = ndt_sbj[participant[n]];
	}
}
model {
	mu_drift_trial_mu ~ normal(drift_trial_mu_priors[1], drift_trial_mu_priors[2]);
	mu_drift_trial_sd ~ normal(drift_trial_sd_priors[1], drift_trial_sd_priors[2]);
	mu_threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	mu_ndt ~ normal(ndt_priors[1], ndt_priors[2]);

	sd_drift_trial_mu ~ normal(drift_trial_mu_priors[3], drift_trial_mu_priors[4]);
	sd_drift_trial_sd ~ normal(drift_trial_sd_priors[3], drift_trial_sd_priors[4]);
	sd_threshold ~ normal(threshold_priors[3], threshold_priors[4]);
	sd_ndt ~ normal(ndt_priors[3], ndt_priors[4]);

	z_drift_trial_mu ~ normal(0, 1);
	z_drift_trial_sd ~ normal(0, 1);
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