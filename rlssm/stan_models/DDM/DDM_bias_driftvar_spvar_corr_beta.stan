data {
	int<lower=1> N;									// number of data items
	int<lower=1> n_cor_par;							// number of correlated parameters

	int<lower=-1,upper=1> accuracy[N];				// accuracy (-1, 1)
	int<lower=0,upper=1> accuracy_flipped[N];		// flipped accuracy (1, 0)
	real<lower=0> rt[N];							// rt
	real beta[N];									// regression coefficients from neuroimaging

	vector[2] drift_trial_mu_priors;					// mean and sd of the prior
	vector[2] threshold_priors;						// mean and sd of the prior
	vector[2] ndt_priors;							// mean and sd of the prior
	vector[2] rel_sp_trial_mu_priors;				// mean and sd of the prior
	vector[2] drift_trial_sd_priors;					// mean and sd of the cauchy prior
	vector[2] rel_sp_trial_sd_priors;				// mean and sd of the cauchy prior
	real<lower=0> corr_matrix_prior;				// eta parameter of the LKJ prior
	vector[2] beta_trial_mu_priors;					// mean and sd of the prior
	vector[2] beta_trial_sd_priors;					// mean and sd of the cauchy prior
}
parameters {
	corr_matrix[n_cor_par] omega;					// correlation matrix for trial-specific drift and rel_sp
	vector[n_cor_par] par_mean_vector;				// drift and rel_sp means
	vector<lower=0>[n_cor_par] ndt;					// drift and rel_sp SD
	vector[n_cor_par] z_par_trial[N];				// trial-specific (deviations from the mean) drift-rate and rel_sp
	real threshold;
	real ndt;
	real<lower=0> sigma;
}
transformed parameters {
	real drift_ll[N];								// trial-by-trial drift rate for likelihood (incorporates accuracy)
	real drift_t[N];								// trial-by-trial drift rate for predictions
	real<lower=0> threshold_t[N];					// trial-by-trial threshold
	real<lower=0> ndt_t[N];							// trial-by-trial ndt
	real<lower=0, upper=1> rel_sp_ll[N];			// trial-by-trial relative starting point for likelihood (incorporates accuracy)
	real<lower=0, upper=1> rel_sp_t[N];				// trial-by-trial relative starting point
	real beta_t[N];									// trial-by-trial regression coefficients

	real drift_trial_mu;
	real rel_sp_trial_mu;
	real beta_trial_mu;
	real drift_trial_sd;
	real rel_sp_trial_sd;
	real beta_trial_sd;
	real corr_drift_rel_sp;
	real corr_drift_beta;
	real corr_rel_sp_beta;

	real transf_drift_trial_mu;
	real transf_rel_sp_trial_mu;
	real transf_beta_trial_mu;
	real<lower=0> transf_drift_trial_sd;
	real<lower=0> transf_rel_sp_trial_sd;
	real<lower=0> transf_beta_trial_sd;
	real transf_corr_drift_rel_sp;
	real transf_corr_drift_beta;
	real transf_corr_rel_sp_beta;
	real transf_threshold;
	real transf_ndt;
	real transf_sigma;

	drift_trial_mu = par_mean_vector[1];			// for the output
	rel_sp_trial_mu = Phi(par_mean_vector[2]);
	beta_trial_mu = par_mean_vector[3];
	drift_trial_sd = ndt[1];
	rel_sp_trial_sd = ndt[2];
	beta_trial_sd = ndt[3];
	corr_drift_rel_sp = omega[2,1];
	corr_drift_beta = omega[3,1];
	corr_rel_sp_beta = omega[3,2];

	transf_drift_trial_mu = drift_trial_mu;
	transf_rel_sp_trial_mu = rel_sp_trial_mu;
	transf_beta_trial_mu = beta_trial_mu;
	transf_drift_trial_sd = drift_trial_sd;
	transf_rel_sp_trial_sd = rel_sp_trial_sd;
	transf_beta_trial_sd = beta_trial_sd;
	transf_corr_drift_rel_sp = corr_drift_rel_sp;
	transf_corr_drift_beta = corr_drift_beta;
	transf_corr_rel_sp_beta = corr_rel_sp_beta;
	transf_threshold = log(1 + exp(threshold));
	transf_ndt = log(1 + exp(ndt));
	transf_sigma = sigma;

	for (n in 1:N) {
		drift_t[n] = z_par_trial[n][1];
		drift_ll[n] = drift_t[n]*accuracy[n];
		threshold_t[n] = transf_threshold;
		ndt_t[n] = transf_ndt;
		rel_sp_t[n] = Phi(z_par_trial[n][2]);
		rel_sp_ll[n] = accuracy_flipped[n] + accuracy[n]*rel_sp_t[n];
		beta_t[n] = z_par_trial[n][3];
	}
}
model {
	omega ~ lkj_corr(corr_matrix_prior);
	par_mean_vector[1] ~ normal(drift_trial_mu_priors[1], drift_trial_mu_priors[2]);
	par_mean_vector[2] ~ normal(rel_sp_trial_mu_priors[1], rel_sp_trial_mu_priors[2]);
	par_mean_vector[3] ~ normal(beta_trial_mu_priors[1], beta_trial_mu_priors[2]);
	ndt[1] ~ cauchy(drift_trial_sd_priors[1], drift_trial_sd_priors[2]);
	ndt[2] ~ cauchy(rel_sp_trial_sd_priors[1], rel_sp_trial_sd_priors[2]);
	ndt[3] ~ cauchy(beta_trial_sd_priors[1], beta_trial_sd_priors[2]);
	threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	ndt ~ normal(ndt_priors[1], ndt_priors[2]);
	sigma ~ normal(0, .1);

	for (n in 1:N)
		z_par_trial[n] ~ multi_normal(par_mean_vector, quad_form_diag(omega, ndt));

	rt ~ wiener(threshold_t, ndt_t, rel_sp_ll, drift_ll);
	beta ~ normal(beta_t, sigma);
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = wiener_lpdf(rt[n] | threshold_t[n], ndt_t[n], rel_sp_ll[n], drift_ll[n]);
	}
}
}
