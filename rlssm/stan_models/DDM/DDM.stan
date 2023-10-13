data {
	int<lower=1> N;									// number of data items

	array[N] int<lower=-1,upper=1> accuracy;		// accuracy (-1, 1)
	array[N] real<lower=0> rt;						// rt

	vector[2] drift_priors;							// mean and sd of the prior
	vector[2] threshold_priors;						// mean and sd of the prior
	vector[2] ndt_priors;							// mean and sd of the prior
	real<lower=0, upper=1> starting_point;			// starting point diffusion model not to estimate
}
parameters {
	real drift;
	real threshold;
	real ndt;
}
transformed parameters {
	array[N] real drift_ll;							// trial-by-trial drift rate for likelihood (incorporates accuracy)
	array[N] real drift_t;							// trial-by-trial drift rate for predictions
	array[N] real<lower=0> threshold_t;				// trial-by-trial threshold
	array[N] real<lower=0> ndt_t;					// trial-by-trial ndt

	real transf_drift;
	real transf_threshold;
	real transf_ndt;

	transf_drift = drift;							// for the output
	transf_threshold = log(1 + exp(threshold));
	transf_ndt = log(1 + exp(ndt));

	for (n in 1:N) {
		drift_t[n] = transf_drift;
		drift_ll[n] = drift_t[n]*accuracy[n];
		threshold_t[n] = transf_threshold;
		ndt_t[n] = transf_ndt;
	}
}
model {
	drift ~ normal(drift_priors[1], drift_priors[2]);
	threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	ndt ~ normal(ndt_priors[1], ndt_priors[2]);

	rt ~ wiener(threshold_t, ndt_t, starting_point, drift_ll);
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = wiener_lpdf(rt[n] | threshold_t[n], ndt_t[n], starting_point, drift_ll[n]);
	}
}
}
