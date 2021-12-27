data {
	int<lower=1> N;									// number of data items
	int<lower=1> L;									// number of levels
	int<lower=1> K;									// number of options
	int<lower=1, upper=L> participant[N];			// level (participant)
	int<lower=1> trial_block[N];					// trial within block
	vector[N] f_cor;								// feedback correct option
	vector[N] f_inc;								// feedback incorrect option
	int<lower=1, upper=K> cor_option[N];			// correct option
	int<lower=1, upper=K> inc_option[N];			// incorrect option
	int<lower=1> block_label[N];					// block label
	int<lower=-1,upper=1> accuracy[N];				// accuracy (0, 1)
	real initial_value;								// intial value for learning in the first block
	vector[4] alpha_pos_priors;						// mean and sd of the mu_alpha_pos prior and sd_alpha_pos prior
	vector[4] alpha_neg_priors;						// mean and sd of the mu_alpha_neg prior and sd_alpha_neg prior
	vector[4] sensitivity_priors;					// mean and sd of the mu_sensitivity prior and sd_sensitivity prior
}
transformed data {
	vector[K] Q0;
	Q0 = rep_vector(initial_value, K);
}
parameters {
	real mu_alpha_pos;
	real mu_alpha_neg;
	real mu_sensitivity;

	real<lower=0> sd_alpha_pos;
	real<lower=0> sd_alpha_neg;
	real<lower=0> sd_sensitivity;

	real z_alpha_pos[L];
	real z_alpha_neg[L];
	real z_sensitivity[L];
}
transformed parameters {
	real log_p_t[N];								// trial-by-trial probability
	vector[K] Q;									// Q state values

	real PE_cor;
	real PE_inc;
	real Q_mean;

	real<lower=0, upper=1> alpha_pos_sbj[L];
	real<lower=0, upper=1> alpha_neg_sbj[L];
	real<lower=0> sensitivity_sbj[L];

	real transf_mu_alpha_pos;
	real transf_mu_alpha_neg;
	real transf_mu_sensitivity;

	transf_mu_alpha_pos = Phi(mu_alpha_pos);		// for the output
	transf_mu_alpha_neg = Phi(mu_alpha_neg);
	transf_mu_sensitivity = log(1 + exp(mu_sensitivity));

	for (l in 1:L) {
		alpha_pos_sbj[l] = Phi(mu_alpha_pos + z_alpha_pos[l]*sd_alpha_pos);
		alpha_neg_sbj[l] = Phi(mu_alpha_neg + z_alpha_neg[l]*sd_alpha_neg);
		sensitivity_sbj[l] = log(1 + exp(mu_sensitivity + z_sensitivity[l]*sd_sensitivity));
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
		PE_cor = f_cor[n] - Q[cor_option[n]];
		PE_inc = f_inc[n] - Q[inc_option[n]];

		log_p_t[n] = sensitivity_sbj[participant[n]]*Q[cor_option[n]] - log(exp(sensitivity_sbj[participant[n]]*Q[cor_option[n]]) + exp(sensitivity_sbj[participant[n]]*Q[inc_option[n]]));

		if (PE_cor >= 0) {
			Q[cor_option[n]] = Q[cor_option[n]] + alpha_pos_sbj[participant[n]]*PE_cor;
		} else {
			Q[cor_option[n]] = Q[cor_option[n]] + alpha_neg_sbj[participant[n]]*PE_cor;
		}
		if (PE_inc >= 0) {
			Q[inc_option[n]] = Q[inc_option[n]] + alpha_pos_sbj[participant[n]]*PE_inc;
		} else {
			Q[inc_option[n]] = Q[inc_option[n]] + alpha_neg_sbj[participant[n]]*PE_inc;
		}
	}
}
model {
	mu_alpha_pos ~ normal(alpha_pos_priors[1], alpha_pos_priors[2]);
	mu_alpha_neg ~ normal(alpha_neg_priors[1], alpha_neg_priors[2]);
	mu_sensitivity ~ normal(sensitivity_priors[1], sensitivity_priors[2]);

	sd_alpha_pos ~ normal(alpha_pos_priors[3], alpha_pos_priors[4]);
	sd_alpha_neg ~ normal(alpha_neg_priors[3], alpha_neg_priors[4]);
	sd_sensitivity ~ normal(sensitivity_priors[3], sensitivity_priors[4]);

	z_alpha_pos ~ normal(0, 1);
	z_alpha_neg ~ normal(0, 1);
	z_sensitivity ~ normal(0, 1);

	accuracy ~ bernoulli(exp(log_p_t));
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = bernoulli_lpmf(accuracy[n] | exp(log_p_t[n]));
	}
	}
}