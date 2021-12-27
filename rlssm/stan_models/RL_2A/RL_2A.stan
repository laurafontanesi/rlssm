data {
	int<lower=1> N;									// number of data items
	int<lower=1> K;									// number of options
	int<lower=1> trial_block[N];					// trial within block
	vector[N] f_cor;								// feedback correct option
	vector[N] f_inc;								// feedback incorrect option
	int<lower=1, upper=K> cor_option[N];			// correct option
	int<lower=1, upper=K> inc_option[N];			// incorrect option
	int<lower=1> block_label[N];					// block label
	int<lower=-1,upper=1> accuracy[N];				// accuracy (0, 1)
	real initial_value;								// intial value for learning in the first block
	vector[2] alpha_priors;							// mean and sd of the alpha prior
	vector[2] sensitivity_priors;					// mean and sd of the sensitivity prior
}
transformed data {
	vector[K] Q0;
	Q0 = rep_vector(initial_value, K);
}
parameters {
	real alpha;
	real sensitivity;
}
transformed parameters {
	real log_p_t[N];								// trial-by-trial probability
	vector[K] Q;									// Q state values

	real PE_cor;
	real PE_inc;
	real Q_mean;

	real transf_alpha;
	real transf_sensitivity;

	transf_alpha = Phi(alpha);
	transf_sensitivity = log(1 + exp(sensitivity));

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

		log_p_t[n] = transf_sensitivity*Q[cor_option[n]] - log(exp(transf_sensitivity*Q[cor_option[n]]) + exp(transf_sensitivity*Q[inc_option[n]]));

		Q[cor_option[n]] = Q[cor_option[n]] + transf_alpha*PE_cor;
		Q[inc_option[n]] = Q[inc_option[n]] + transf_alpha*PE_inc;
	}
}
model {
	alpha ~ normal(alpha_priors[1], alpha_priors[2]);
	sensitivity ~ normal(sensitivity_priors[1], sensitivity_priors[2]);

	accuracy ~ bernoulli(exp(log_p_t));
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = bernoulli_lpmf(accuracy[n] | exp(log_p_t[n]));
	}
	}
}