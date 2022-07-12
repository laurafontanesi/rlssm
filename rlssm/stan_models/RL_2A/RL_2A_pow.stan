data {
	int<lower=1> N;									// number of data items
	int<lower=1> K;									// number of options
	int<lower=1> trial_block[N];					// trial within block
	vector[N] times_seen;							// times an option is seen in a block
	vector[N] f_cor;								// feedback correct option
	vector[N] f_inc;								// feedback incorrect option
	int<lower=1, upper=K> cor_option[N];			// correct option
	int<lower=1, upper=K> inc_option[N];			// incorrect option
	int<lower=1> block_label[N];					// block label
	int<lower=-1,upper=1> accuracy[N];				// accuracy (0, 1)
	int<lower=0, upper=1> feedback_type[N]; // feedback_type = 0 -> full feedback, feedback_type = 1 -> partial feedback
	real initial_value;								// intial value for learning in the first block
	vector[2] alpha_priors;							// mean and sd of the alpha prior
	vector[2] consistency_priors;					// mean and sd of the consistency prior
	vector[2] scaling_priors;						// mean and sd of the scaling prior
}
transformed data {
	vector[K] Q0;
	Q0 = rep_vector(initial_value, K);
}
parameters {
	real alpha;
	real consistency;
	real scaling;
}
transformed parameters {
	real log_p_t[N];								// trial-by-trial probability
	real<lower=0> sensitivity_t[N];					// trial-by-trial sensitivity
	vector[K] Q;									// Q state values

	real PE_cor;
	real PE_inc;
	real Q_mean;

	real transf_alpha;
	real transf_consistency;
	real transf_scaling;

	transf_alpha = Phi(alpha);
	transf_consistency = log(1 + exp(consistency));
	transf_scaling = log(1 + exp(scaling));

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

		sensitivity_t[n] = (times_seen[n]/transf_scaling)^transf_consistency;
		log_p_t[n] = sensitivity_t[n]*Q[cor_option[n]] - log(exp(sensitivity_t[n]*Q[cor_option[n]]) + exp(sensitivity_t[n]*Q[inc_option[n]]));

		if (feedback_type[n] == 1){
      if(accuracy[n] == 1){
        Q[cor_option[n]] = Q[cor_option[n]] + transf_alpha*PE_cor;
      }
      else{
        Q[inc_option[n]] = Q[inc_option[n]] + transf_alpha*PE_inc;
      }
    }
    else{
      Q[cor_option[n]] = Q[cor_option[n]] + transf_alpha*PE_cor;
      Q[inc_option[n]] = Q[inc_option[n]] + transf_alpha*PE_inc;
    }
	}
}
model {
	alpha ~ normal(alpha_priors[1], alpha_priors[2]);
	consistency ~ normal(consistency_priors[1], consistency_priors[2]);
	scaling ~ normal(scaling_priors[1], scaling_priors[2]);

	accuracy ~ bernoulli(exp(log_p_t));
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = bernoulli_lpmf(accuracy[n] | exp(log_p_t[n]));
	}
	}
}