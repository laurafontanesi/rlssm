data {
	int<lower=1> N;									// number of data items
	int<lower=1> L;									// number of levels
	int<lower=1> C_drift;							// number of conditions for drift rate
	int<lower=1> C_threshold;						// number of conditions for threshold
	int<lower=1> C_ndt;								// number of conditions for ndt
	int<lower=1> C_sp;								// number of conditions for starting point
	int<lower=1, upper=L> participant[N];			// level (participant)

	row_vector[C] x_drift[N];						// matrix[N, C] predictor matrix drift rate
	row_vector[C] x_threshold[N];					// matrix[N, C] predictor matrix threshold
	row_vector[C] x_ndt[N];							// matrix[N, C] predictor matrix ndt
	row_vector[C] x_sp[N];							// matrix[N, C] predictor matrix starting point

	int<lower=-1,upper=1> accuracy[N];				// accuracy (-1, 1)
	real<lower=0> rt[N];							// rt

	vector[4] drift_priors;							// mean and sd of the mu_ prior and sd_ prior
	vector[4] threshold_priors;						// mean and sd of the mu_ prior and sd_ prior
	vector[4] ndt_priors;							// mean and sd of the mu_ prior and sd_ prior
	vector[4] sp_priors;							// mean and sd of the mu_ prior and sd_ prior
}
parameters {
	real drift_intercept;
	real threshold_intercept;
	real ndt_intercept;
	real drift_coeff[K];
	real threshold_coeff[K];
	real ndt_coeff[K];

	real<lower=0> sd_drift_intercept;
	real<lower=0> sd_threshold_intercept;
	real<lower=0> sd_ndt_intercept;

	real<lower=0> sd_drift_coeff[K];
	real<lower=0> sd_threshold_coeff[K];
	real<lower=0> sd_ndt_coeff[K];

	real z_drift_intercept[L];
	real z_threshold_intercept[L];
	real z_ndt_intercept[L];

	vector[K] z_drift_coeff[L];
	vector[K] z_threshold_coeff[L];
	vector[K] z_ndt_coeff[L];
}
transformed parameters {
	real drift_t[N];								// trial-by-trial drift-rate for likelihood (incorporates accuracy)
	real drift_p[N];								// trial-by-trial drift-rate for predictions
	real<lower=0> thr_t[N];							// trial-by-trial threshold
	real<lower=0> ndt_t[N];							// trial-by-trial ndt

	real drift_intercept_sbj[L];
	real threshold_intercept_sbj[L];
	real ndt_intercept_sbj[L];
	vector[K] drift_coeff_sbj[L];
	vector[K] threshold_coeff_sbj[L];
	vector[K] ndt_coeff_sbj[L];

	for (l in 1:L) {
		drift_intercept_sbj[l] = drift_intercept + z_drift_intercept[l]*sd_drift_intercept;
		threshold_intercept_sbj[l] = threshold_intercept + z_threshold_intercept[l]*sd_threshold_intercept;
		ndt_intercept_sbj[l] = ndt_intercept + z_ndt_intercept[l]*sd_ndt_intercept;
		for (k in 1:K) {
			drift_coeff_sbj[l][k] = drift_coeff[k] + z_drift_coeff[l][k]*sd_drift_coeff[k];
			threshold_coeff_sbj[l][k] = threshold_coeff[k] + z_threshold_coeff[l][k]*sd_threshold_coeff[k];
			ndt_coeff_sbj[l][k] = ndt_coeff[k] + z_ndt_coeff[l][k]*sd_ndt_coeff[k];
		}
	}
	for (n in 1:N) {
		drift_p[n] = x[n]*drift_coeff_sbj[participant[n]] + drift_intercept_sbj[participant[n]];
		drift_t[n] = drift_p[n]*accuracy[n];
		thr_t[n] = log(1 + exp(x[n]*threshold_coeff_sbj[participant[n]] + threshold_intercept_sbj[participant[n]]));
		ndt_t[n] = log(1 + exp(x[n]*ndt_coeff_sbj[participant[n]] + ndt_intercept_sbj[participant[n]]));
	}
}
model {
	drift_intercept ~ cauchy(0, 5);
	threshold_intercept ~ cauchy(0, 5);
	ndt_intercept ~ cauchy(0, 5);

	sd_drift_intercept ~ cauchy(0, 5);
	sd_threshold_intercept ~ cauchy(0, 5);
	sd_ndt_intercept ~ cauchy(0, 5);

	drift_coeff ~ cauchy(0, 5); 
	threshold_coeff ~ cauchy(0, 5); 
	ndt_coeff ~ cauchy(0, 5); 

	sd_drift_coeff ~ cauchy(0, 5);
	sd_threshold_coeff ~ cauchy(0, 5);
	sd_ndt_coeff ~ cauchy(0, 5);

	z_drift_intercept ~ normal(0, 1);
	z_threshold_intercept ~ normal(0, 1);
	z_ndt_intercept ~ normal(0, 1);

	for (l in 1:L) {
		z_drift_coeff[l] ~ normal(0, 1);
		z_threshold_coeff[l] ~ normal(0, 1);
		z_ndt_coeff[l] ~ normal(0, 1);
	}

	rt ~ wiener(thr_t, ndt_t, .5, drift_t);
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = wiener_lpdf(rt[n] | thr_t[n], ndt_t[n], .5, drift_t[n]);
	}
}
}