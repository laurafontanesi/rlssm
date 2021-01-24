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
	vector[4] alpha_priors;							// mean and sd of the mu_alpha prior and sd_alpha prior
	vector[4] sensitivity_priors;					// mean and sd of the mu_sensitivity prior and sd_sensitivity prior
}
transformed data {
	vector[K] Q0;
	Q0 = rep_vector(initial_value, K);
}
parameters {
	real mu_alpha;
	real mu_sensitivity;

	real<lower=0> sd_alpha;
	real<lower=0> sd_sensitivity;

	real z_alpha[L];
	real z_sensitivity[L];
}
transformed parameters {
	real log_p_t[N];								// trial-by-trial probability
	vector[K] Q;									// Q state values

	real PE_cor;
	real PE_inc;
	real Q_mean;

	real<lower=0, upper=1> alpha_sbj[L];
	real<lower=0> sensitivity_sbj[L];

	real transf_mu_alpha;
	real transf_mu_sensitivity;

	transf_mu_alpha = Phi(mu_alpha);				// for the output
	transf_mu_sensitivity = log(1 + exp(mu_sensitivity));

	for (l in 1:L) {
		alpha_sbj[l] = Phi(mu_alpha + z_alpha[l]*sd_alpha);
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

		Q[cor_option[n]] = Q[cor_option[n]] + alpha_sbj[participant[n]]*PE_cor;
		Q[inc_option[n]] = Q[inc_option[n]] + alpha_sbj[participant[n]]*PE_inc;
	}
}
model {
	mu_alpha ~ normal(alpha_priors[1], alpha_priors[2]);
	mu_sensitivity ~ normal(sensitivity_priors[1], sensitivity_priors[2]);

	sd_alpha ~ normal(alpha_priors[3], alpha_priors[4]);
	sd_sensitivity ~ normal(sensitivity_priors[3], sensitivity_priors[4]);

	z_alpha ~ normal(0, 1);
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