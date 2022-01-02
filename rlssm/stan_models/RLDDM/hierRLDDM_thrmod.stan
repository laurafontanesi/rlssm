data {
	int<lower=1> N;									// number of data items
	int<lower=1> K;									// number of options
	int<lower=1> L;									// number of levels
	int<lower=1, upper=L> participant[N];			// level (participant)
	int<lower=1> trial_block[N];					// trial within block
	vector[N] f_cor;								// feedback correct option
	vector[N] f_inc;								// feedback incorrect option
	int<lower=1, upper=K> cor_option[N];			// correct option
	int<lower=1, upper=K> inc_option[N];			// incorrect option
	int<lower=1> block_label[N];					// block label

	int<lower=-1,upper=1> accuracy[N];				// accuracy (-1, 1)
	real<lower=0> rt[N];							// rt

	real initial_value;								// intial value for learning in the first block

	vector[4] alpha_priors;							// mean and sd of the prior
	vector[4] drift_scaling_priors;					// mean and sd of the prior
	vector[4] threshold_priors;						// mean and sd of the prior
	vector[4] threshold_modulation_priors;			// mean and sd of the prior
	vector[4] ndt_priors;							// mean and sd of the prior
	real<lower=0, upper=1> starting_point;			// starting point diffusion model not to estimate
}
transformed data {
	vector[K] Q0;
	Q0 = rep_vector(initial_value, K);
}
parameters {
	real mu_alpha;
	real mu_drift_scaling;
	real mu_threshold;
	real mu_threshold_modulation;
	real mu_ndt;

	real<lower=0> sd_alpha;
	real<lower=0> sd_drift_scaling;
	real<lower=0> sd_threshold;
	real<lower=0> sd_threshold_modulation;
	real<lower=0> sd_ndt;

	real z_alpha[L];
	real z_drift_scaling[L];
	real z_threshold[L];
	real z_threshold_modulation[L];
	real z_ndt[L];
}
transformed parameters {
	real drift_ll[N];								// trial-by-trial drift rate for likelihood (incorporates accuracy)
	real drift_t[N];								// trial-by-trial drift rate for predictions
	real<lower=0> threshold_t[N];					// trial-by-trial threshold   
	real<lower=0> ndt_t[N];							// trial-by-trial ndt

	vector[K] Q;									// Q state values

	real Q_mean;									// mean across all options
	real Q_mean_pres[N];							// mean Q presented options
	real delta_Q[N];								// Qcor - Qinc
	real PE_cor;									// predicion error correct option
	real PE_inc;									// predicion error incorrect option

	real<lower=0, upper=1> alpha_sbj[L];
	real<lower=0> drift_scaling_sbj[L];
	real threshold_sbj[L];
	real threshold_modulation_sbj[L];
	real<lower=0> ndt_sbj[L];

	real transf_mu_alpha;
	real transf_mu_drift_scaling;
	real transf_mu_threshold;
	real transf_mu_threshold_modulation;
	real transf_mu_ndt;

	transf_mu_alpha = Phi(mu_alpha);				// for the output
	transf_mu_drift_scaling = log(1 + exp(mu_drift_scaling));
	transf_mu_threshold = log(1 + exp(mu_threshold));
	transf_mu_threshold_modulation = mu_threshold_modulation;
	transf_mu_ndt = log(1 + exp(mu_ndt));

	for (l in 1:L) {
		alpha_sbj[l] = Phi(mu_alpha + z_alpha[l]*sd_alpha);
		drift_scaling_sbj[l] = log(1 + exp(mu_drift_scaling + z_drift_scaling[l]*sd_drift_scaling));
		threshold_sbj[l] = mu_threshold + z_threshold[l]*sd_threshold;
		threshold_modulation_sbj[l] = mu_threshold_modulation + z_threshold_modulation[l]*sd_threshold_modulation;
		ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l]*sd_ndt));
	}

	for (n in 1:N) {
		if (trial_block[n] == 1) {
			if (block_label[n] == 1) {
				Q = Q0;
			} else {
				Q_mean = mean(Q);
				Q = rep_vector(Q_mean, K);
			}
		}
		Q_mean_pres[n] = (Q[cor_option[n]] + Q[inc_option[n]])/2;
		delta_Q[n] = Q[cor_option[n]] - Q[inc_option[n]];
		PE_cor = f_cor[n] - Q[cor_option[n]];
		PE_inc = f_inc[n] - Q[inc_option[n]];

		drift_t[n] = drift_scaling_sbj[participant[n]]*delta_Q[n];
		drift_ll[n] = drift_t[n]*accuracy[n];
		threshold_t[n] = log(1 + exp(threshold_sbj[participant[n]] + threshold_modulation_sbj[participant[n]]*Q_mean_pres[n]));
		ndt_t[n] = ndt_sbj[participant[n]];

		Q[cor_option[n]] = Q[cor_option[n]] + alpha_sbj[participant[n]]*PE_cor;
		Q[inc_option[n]] = Q[inc_option[n]] + alpha_sbj[participant[n]]*PE_inc;
	}
}
model {
	mu_alpha ~ normal(alpha_priors[1], alpha_priors[2]);
	mu_drift_scaling ~ normal(drift_scaling_priors[1], drift_scaling_priors[2]);
	mu_threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	mu_threshold_modulation ~ normal(threshold_modulation_priors[1], threshold_modulation_priors[2]);
	mu_ndt ~ normal(ndt_priors[1], ndt_priors[2]);

	sd_alpha ~ normal(alpha_priors[3], alpha_priors[4]);
	sd_drift_scaling ~ normal(drift_scaling_priors[3], drift_scaling_priors[4]);
	sd_threshold ~ normal(threshold_priors[3], threshold_priors[4]);
	sd_threshold_modulation ~ normal(threshold_modulation_priors[3], threshold_modulation_priors[4]);
	sd_ndt ~ normal(ndt_priors[3], ndt_priors[4]);

	z_alpha ~ normal(0, 1);
	z_drift_scaling ~ normal(0, 1);
	z_threshold ~ normal(0, 1);
	z_threshold_modulation ~ normal(0, 1);
	z_ndt ~ normal(0, 1);

	rt ~ wiener(threshold_t, ndt_t, starting_point, drift_ll);
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = wiener_lpdf(rt[n] | threshold_t[n], ndt_t[n], starting_point, drift_ll[n]);
	}
	}
}