data {
	int<lower=1> N;									// number of data items
	int<lower=1> n_cor_par;							// number of correlated parameters

	int<lower=-1,upper=1> accuracy[N];				// accuracy (-1, 1)
	int<lower=0,upper=1> accuracy_flipped[N];		// flipped accuracy (1, 0)
	real<lower=0> rt[N];							// rt

	vector[2] drift_trialmu_priors;					// mean and sd of the prior
	vector[2] threshold_priors;						// mean and sd of the prior
	vector[2] ndt_priors;							// mean and sd of the prior
	vector[2] rel_sp_trialmu_priors;				// mean and sd of the prior
	vector[2] drift_trialsd_priors;					// mean and sd of the cauchy prior
	vector[2] rel_sp_trialsd_priors;				// mean and sd of the cauchy prior
	real<lower=0> corr_matrix_prior;				// eta parameter of the LKJ prior
}
parameters {
	corr_matrix[n_cor_par] omega;					// correlation matrix for trial-specific drift and rel_sp
	vector[n_cor_par] par_mean_vector;				// drift and rel_sp means
	vector<lower=0>[n_cor_par] tau;					// drift and rel_sp SD
	vector[n_cor_par] z_par_trial[N];				// trial-specific (deviations from the mean) drift-rate and rel_sp
	real threshold;
	real ndt;
}
transformed parameters {
	real drift_ll[N];								// trial-by-trial drift rate for likelihood (incorporates accuracy)
	real drift_t[N];								// trial-by-trial drift rate for predictions
	real<lower=0> threshold_t[N];					// trial-by-trial threshold   
	real<lower=0> ndt_t[N];							// trial-by-trial ndt
	real<lower=0, upper=1> rel_sp_ll[N];			// trial-by-trial relative starting point for likelihood (incorporates accuracy)
	real<lower=0, upper=1> rel_sp_t[N];				// trial-by-trial relative starting point

	real drift_trialmu;
	real rel_sp_trialmu;
	real drift_trialsd;
	real rel_sp_trialsd;
	real corr_drift_rel_sp;
	real transf_drift_trialmu;
	real transf_rel_sp_trialmu;
	real<lower=0> transf_drift_trialsd;
	real<lower=0> transf_rel_sp_trialsd;
	real transf_corr_drift_rel_sp;
	real transf_threshold;
	real transf_ndt;

	drift_trialmu = par_mean_vector[1];			// for the output
	rel_sp_trialmu = Phi(par_mean_vector[2]);
	drift_trialsd = tau[1];
	rel_sp_trialsd = tau[2];
	corr_drift_rel_sp = omega[2,1];
	transf_drift_trialmu = drift_trialmu;
	transf_rel_sp_trialmu = rel_sp_trialmu;
	transf_drift_trialsd = drift_trialsd;
	transf_rel_sp_trialsd = rel_sp_trialsd;
	transf_corr_drift_rel_sp = corr_drift_rel_sp;
	transf_threshold = log(1 + exp(threshold));
	transf_ndt = log(1 + exp(ndt));

	for (n in 1:N) {
		drift_t[n] = z_par_trial[n][1];
		drift_ll[n] = drift_t[n]*accuracy[n];
		threshold_t[n] = transf_threshold;
		ndt_t[n] = transf_ndt;
		rel_sp_t[n] = Phi(z_par_trial[n][2]);
		rel_sp_ll[n] = accuracy_flipped[n] + accuracy[n]*rel_sp_t[n];
	}
}
model {
	omega ~ lkj_corr(corr_matrix_prior);
	par_mean_vector[1] ~ normal(drift_trialmu_priors[1], drift_trialmu_priors[2]);
	par_mean_vector[2] ~ normal(rel_sp_trialmu_priors[1], rel_sp_trialmu_priors[2]);
	tau[1] ~ cauchy(drift_trialsd_priors[1], drift_trialsd_priors[2]);
	tau[2] ~ cauchy(rel_sp_trialsd_priors[1], rel_sp_trialsd_priors[2]);
	threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	ndt ~ normal(ndt_priors[1], ndt_priors[2]);
	
	for (n in 1:N)
		z_par_trial[n] ~ multi_normal(par_mean_vector, quad_form_diag(omega, tau));

	rt ~ wiener(threshold_t, ndt_t, rel_sp_ll, drift_ll);
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = wiener_lpdf(rt[n] | threshold_t[n], ndt_t[n], rel_sp_ll[n], drift_ll[n]);
	}
}
}