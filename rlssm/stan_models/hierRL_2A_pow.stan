data {
	int<lower=1> N;									// number of data items
	int<lower=1> L;									// number of levels
	int<lower=1> K;									// number of options
	int<lower=1, upper=L> participant[N];			// level (participant)
	int<lower=1> trial_block[N];					// trial within block
	vector[N] times_seen;							// times an option is seen in a block
	vector[N] f_cor;								// feedback correct option
	vector[N] f_inc;								// feedback incorrect option
	int<lower=1, upper=K> cor_option[N];			// correct option
	int<lower=1, upper=K> inc_option[N];			// incorrect option
	int<lower=1> block_label[N];					// block label
	int<lower=-1,upper=1> accuracy[N];				// accuracy (0, 1)
	real initial_value;								// intial value for learning in the first block
	vector[4] alpha_priors;							// mean and sd of the mu_alpha prior and sd_alpha prior
	vector[4] consistency_priors;					// mean and sd of the mu_consistency prior and sd_consistency prior
	vector[4] scaling_priors;						// mean and sd of the mu_scaling prior and sd_scaling prior
}
transformed data {
	vector[K] Q0;
	Q0 = rep_vector(initial_value, K);
}
parameters {
	real mu_alpha;
	real mu_consistency;
	real mu_scaling;

	real<lower=0> sd_alpha;
	real<lower=0> sd_consistency;
	real<lower=0> sd_scaling;

	real z_alpha[L];
	real z_consistency[L];
	real z_scaling[L];
}
transformed parameters {
	real log_p_t[N];								// trial-by-trial probability
	real<lower=0> sensitivity_t[N];					// trial-by-trial sensitivity
	vector[K] Q;									// Q state values

	real PE_cor;
	real PE_inc;
	real Q_mean;

	real<lower=0, upper=1> alpha_sbj[L];
	real<lower=0> consistency_sbj[L];
	real<lower=0> scaling_sbj[L];

	real transf_mu_alpha;
	real transf_mu_consistency;
	real transf_mu_scaling;

	transf_mu_alpha = Phi(mu_alpha);				// for the output
	transf_mu_consistency = log(1 + exp(mu_consistency));
	transf_mu_scaling = log(1 + exp(mu_scaling));

	for (l in 1:L) {
		alpha_sbj[l] = Phi(mu_alpha + z_alpha[l]*sd_alpha);
		consistency_sbj[l] = log(1 + exp(mu_consistency + z_consistency[l]*sd_consistency));
		scaling_sbj[l] = log(1 + exp(mu_scaling + z_scaling[l]*sd_scaling));
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

		sensitivity_t[n] = (times_seen[n]/scaling_sbj[participant[n]])^consistency_sbj[participant[n]];
		log_p_t[n] = sensitivity_t[n]*Q[cor_option[n]] - log(exp(sensitivity_t[n]*Q[cor_option[n]]) + exp(sensitivity_t[n]*Q[inc_option[n]]));

		Q[cor_option[n]] = Q[cor_option[n]] + alpha_sbj[participant[n]]*PE_cor;
		Q[inc_option[n]] = Q[inc_option[n]] + alpha_sbj[participant[n]]*PE_inc;
	}
}
model {
	mu_alpha ~ normal(alpha_priors[1], alpha_priors[2]);
	mu_consistency ~ normal(consistency_priors[1], consistency_priors[2]);
	mu_scaling ~ normal(scaling_priors[1], scaling_priors[2]);

	sd_alpha ~ normal(alpha_priors[3], alpha_priors[4]);
	sd_consistency ~ normal(consistency_priors[3], consistency_priors[4]);
	sd_scaling ~ normal(scaling_priors[3], scaling_priors[4]);

	z_alpha ~ normal(0, 1);
	z_consistency ~ normal(0, 1);
	z_scaling ~ normal(0, 1);

	accuracy ~ bernoulli(exp(log_p_t));
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = bernoulli_lpmf(accuracy[n] | exp(log_p_t[n]));
	}
	}
}